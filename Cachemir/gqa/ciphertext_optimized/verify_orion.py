"""Verify the new persistent-layout ciphertext implementation with Orion.

Run from the directory containing the ``gqa`` package:
    python -m gqa.ciphertext.verify_orion
"""

from __future__ import annotations
from pathlib import Path
import statistics
from collections import defaultdict

import orion
import yaml
import gc 

from ..plaintext_optimized.counter import counter
from .he_attention_orion import (
    execute_prepared_attention_gqa_he,
    prepare_attention_gqa_he,
    verify_prepared_attention_gqa_he,
)


LEVEL = 7
TOLERANCE = 5e-3
WARMUP_RUNS = 1
MEASURED_RUNS = 5
CASES = [
    # logN, d, H, n_kv, n_prefill, projection method
    (10, 256, 16, 16, 16, "bsgs"),
    (10, 128, 16, 16, 16, "bsgs"),
    (11, 512, 8, 8, 16, "bsgs"),
    (13, 512, 8, 8, 64, "bsgs"),
    (13, 512, 8, 8, 16, "bsgs"),
]


def get_config_path() -> str:
    return str(Path(__file__).resolve().parent / "configs" / "gqa.yml")


def get_scheme_config(logN: int) -> dict:
    """Load the base Orion configuration and override its ring dimension."""
    with open(get_config_path(), encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    config["ckks_params"]["LogN"] = logN
    return config


def _stage_metrics() -> dict[str, dict[str, float | int | None]]:
    """Copy stage data before the next execution resets the counter."""
    return {
        name: {
            "runtime_seconds": float(values["runtime_seconds"]),
            "level_before": values["level_before"],
            "level_after": values["level_after"],
            "rotations": int(values["rotations"]),
            "unique_rotation_amounts": int(
                values["unique_rotation_amounts"]
            ),
            "ct_pt_mult": int(values["ct_pt_mult"]),
            "ct_ct_mult": int(values["ct_ct_mult"]),
            "levels_consumed": (
                values["level_before"] - values["level_after"]
                if (
                    values["level_before"] is not None
                    and values["level_after"] is not None
                )
                else None
            ),
        }
        for name, values in counter.stages.items()
    }


def _level_text(
    level_before: int | None,
    level_after: int | None,
) -> str:
    if level_before is None or level_after is None:
        return "level=n/a"
    return (
        f"level={level_before}->{level_after}  "
        f"consumed={level_before - level_after}"
    )


def _print_run_report(
    run: int,
    stage_metrics: dict[str, dict[str, float | int | None]],
    total_time: float,
    errors: tuple[float, float, float],
) -> None:
    projection_error, qkt_error, output_error = errors
    error_by_stage = {
        "Q/K/V projection": projection_error,
        "QK^T": qkt_error,
        "QK^T V": output_error,
        "Softmax(QK^T)V": output_error,
    }

    print(f"\nRound {run}")
    for name, metrics in stage_metrics.items():
        error = error_by_stage.get(name)
        error_text = "" if error is None else f"  error={error:.3e}"
        print(
            f"  {name + ':':<22}"
            f"runtime={metrics['runtime_seconds']:.6f} s  "
            f"{_level_text(metrics['level_before'], metrics['level_after'])}"
            f"{error_text}"
        )

    first_stage = next(iter(stage_metrics.values()))
    last_stage = next(reversed(stage_metrics.values()))
    print(
        f"  {'total:':<22}"
        f"runtime={total_time:.6f} s  "
        f"{_level_text(first_stage['level_before'], last_stage['level_after'])}  "
        f"max_error={max(errors):.3e}"
    )


def _print_average_report(
    stage_history: dict[str, list[float]],
    stage_levels: dict[str, tuple[int | None, int | None]],
    total_times: list[float],
    error_history: dict[str, list[float]],
) -> None:
    print("\nAverages")
    for name, runtimes in stage_history.items():
        errors = error_history[name]
        level_before, level_after = stage_levels[name]
        print(
            f"  {name + ':':<22}"
            f"runtime={statistics.mean(runtimes):.6f} s  "
            f"{_level_text(level_before, level_after)}  "
            f"error={statistics.mean(errors):.3e}"
        )

    first_stage = next(iter(stage_levels.values()))
    last_stage = next(reversed(stage_levels.values()))
    print(
        f"  {'total:':<22}"
        f"runtime={statistics.mean(total_times):.6f} s  "
        f"{_level_text(first_stage[0], last_stage[1])}  "
        f"max_error={max(max(values) for values in error_history.values()):.3e}"
    )


def _print_operation_report(
    stage_metrics: dict[str, dict[str, float | int | None]],
) -> None:
    print("\nOperation counts (identical for each measured round)")
    for name, metrics in stage_metrics.items():
        print(
            f"  {name + ':':<22}"
            f"rotations={metrics['rotations']:<4}  "
            f"unique={metrics['unique_rotation_amounts']:<4}  "
            f"ct-pt={metrics['ct_pt_mult']:<4}  "
            f"ct-ct={metrics['ct_ct_mult']:<4}"
        )
    print(f"  {'total:':<22}{counter.summary()}")


def main() -> None:
    scheme = None
    for logN, d, H, n_kv, n_prefill, method in CASES:
        # Orion accepts either a YAML path or a configuration dictionary.
        # Loading the YAML and overriding LogN lets every case select its
        # own ring dimension before the scheme and keys are initialized.
        if scheme is not None:
            # Delete ciphertext wrappers before clearing the backend.
            if "result" in locals():
                del result
            if "prepared" in locals():
                del prepared

            gc.collect()
            scheme.delete_scheme()
            scheme.backend = None
            gc.collect()
        scheme = orion.init_scheme(get_scheme_config(logN))
        actual_logN = scheme.params.get_logn()
        if actual_logN != logN:
            raise RuntimeError(
                f"Requested logN={logN}, but Orion initialized logN={actual_logN}."
            )
        n_he = 1 << (actual_logN - 1)

        print(
            f"\nlogN={logN}, slots={n_he}, starting level={LEVEL}, "
            f"d={d}, H={H}, n_kv={n_kv}, "
            f"n_prefill={n_prefill}, method={method}"
        )

        # One-time work: dimensions, random weights, plaintext weight
        # encoding, input encryption, and encrypted prefill-cache creation.
        prepared = prepare_attention_gqa_he(
            n_he=n_he,
            d=d,
            H=H,
            n_kv=n_kv,
            n_prefill=n_prefill,
            level=LEVEL,
            qkv_method=method,
            skip_softmax=True,
        )

        for _ in range(WARMUP_RUNS):
            execute_prepared_attention_gqa_he(prepared)

        total_times: list[float] = []
        stage_history: dict[str, list[float]] = defaultdict(list)
        stage_levels: dict[str, tuple[int | None, int | None]] = {}
        error_history: dict[str, list[float]] = defaultdict(list)
        last_stage_metrics: dict[
            str, dict[str, float | int | None]
        ] = {}

        for run in range(MEASURED_RUNS):
            result = execute_prepared_attention_gqa_he(prepared)
            total_time = float(counter.total_runtime_seconds)
            stage_metrics = _stage_metrics()
            last_stage_metrics = stage_metrics

            # The attention timer has already stopped, so decrypt/decode and
            # dense-reference verification are excluded from every runtime.
            errors = verify_prepared_attention_gqa_he(
                prepared,
                result,
                verbose=False,
            )

            total_times.append(total_time)
            for name, metrics in stage_metrics.items():
                stage_history[name].append(metrics["runtime_seconds"])
                levels = (
                    metrics["level_before"],
                    metrics["level_after"],
                )
                if name in stage_levels and stage_levels[name] != levels:
                    raise RuntimeError(
                        f"Stage {name!r} changed level schedule across runs: "
                        f"{stage_levels[name]} then {levels}."
                    )
                stage_levels[name] = levels

            projection_error, qkt_error, output_error = errors
            error_history["Q/K/V projection"].append(projection_error)
            error_history["QK^T"].append(qkt_error)
            output_stage = (
                "QK^T V"
                if prepared.skip_softmax
                else "Softmax(QK^T)V"
            )
            error_history[output_stage].append(output_error)

            _print_run_report(
                run + 1,
                stage_metrics,
                total_time,
                errors,
            )

        _print_average_report(
            stage_history,
            stage_levels,
            total_times,
            error_history,
        )
        _print_operation_report(last_stage_metrics)

        maximum = max(
            max(values)
            for values in error_history.values()
        )
        print(
            f"verification: {'PASSED' if maximum < TOLERANCE else 'FAILED'} "
            f"(max_error={maximum:.3e}, tolerance={TOLERANCE:.1e})\n"
        )


if __name__ == "__main__":
    main()