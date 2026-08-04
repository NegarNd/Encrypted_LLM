"""Verify the persistent-layout ciphertext implementation with Orion.

Run from the directory containing the gqa package:
    python -m gqa.ciphertext.verify_orion
"""

from __future__ import annotations

import argparse
import gc
import statistics
from collections import defaultdict
from pathlib import Path

import orion
import yaml

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-softmax",
        action="store_true",
        help="Compute QK^T V directly instead of approximate encrypted softmax.",
    )
    return parser.parse_args()


def get_config_path() -> str:
    return str(Path(__file__).resolve().parent / "configs" / "gqa.yml")


def get_scheme_config(logN: int) -> dict:
    """Load the Orion configuration and override its ring dimension."""
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
) -> None:
    print(f"\nRound {run}")

    for name, metrics in stage_metrics.items():
        print(
            f"  {name + ':':<22}"
            f"runtime={metrics['runtime_seconds']:.6f} s  "
            f"{_level_text(metrics['level_before'], metrics['level_after'])}"
        )

    first_stage = next(iter(stage_metrics.values()))
    last_stage = next(reversed(stage_metrics.values()))

    print(
        f"  {'total:':<22}"
        f"runtime={total_time:.6f} s  "
        f"{_level_text(first_stage['level_before'], last_stage['level_after'])}"
    )


def _print_average_report(
    stage_history: dict[str, list[float]],
    stage_levels: dict[str, tuple[int | None, int | None]],
    total_times: list[float],
) -> None:
    print("\nAverages")

    for name, runtimes in stage_history.items():
        level_before, level_after = stage_levels[name]

        print(
            f"  {name + ':':<22}"
            f"runtime={statistics.mean(runtimes):.6f} s  "
            f"{_level_text(level_before, level_after)}"
        )

    first_stage = next(iter(stage_levels.values()))
    last_stage = next(reversed(stage_levels.values()))

    print(
        f"  {'total:':<22}"
        f"runtime={statistics.mean(total_times):.6f} s  "
        f"{_level_text(first_stage[0], last_stage[1])}"
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
    args = parse_args()
    scheme = None

    for logN, d, H, n_kv, n_prefill, method in CASES:
        if scheme is not None:
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
                f"Requested logN={logN}, "
                f"but Orion initialized logN={actual_logN}."
            )

        n_he = 1 << (actual_logN - 1)

        print(
            f"\nlogN={logN}, slots={n_he}, starting level={LEVEL}, "
            f"d={d}, H={H}, n_kv={n_kv}, "
            f"n_prefill={n_prefill}, method={method}"
        )

        prepared = prepare_attention_gqa_he(
            n_he=n_he,
            d=d,
            H=H,
            n_kv=n_kv,
            n_prefill=n_prefill,
            level=LEVEL,
            qkv_method=method,
            skip_softmax=args.skip_softmax,
        )

        # Warmup runs are not verified.
        for _ in range(WARMUP_RUNS):
            execute_prepared_attention_gqa_he(prepared)

        total_times: list[float] = []
        stage_history: dict[str, list[float]] = defaultdict(list)
        stage_levels: dict[
            str,
            tuple[int | None, int | None],
        ] = {}

        last_stage_metrics: dict[
            str,
            dict[str, float | int | None],
        ] = {}

        result = None

        # Only execute and collect timing information inside this loop.
        for run in range(MEASURED_RUNS):
            result = execute_prepared_attention_gqa_he(prepared)

            total_time = float(counter.total_runtime_seconds)
            stage_metrics = _stage_metrics()
            last_stage_metrics = stage_metrics

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

            _print_run_report(
                run + 1,
                stage_metrics,
                total_time,
            )

        _print_average_report(
            stage_history,
            stage_levels,
            total_times,
        )

        _print_operation_report(last_stage_metrics)

        if result is None:
            raise RuntimeError("No measured output was generated.")

        # Verify only the output from the final measured run.
        # Verification is outside the measured loop, so decryption and
        # plaintext-reference evaluation are not included in runtime.
        projection_error, qkt_error, output_error = (
            verify_prepared_attention_gqa_he(
                prepared,
                result,
                verbose=False,
            )
        )

        output_name = (
            "QK^T V"
            if prepared.skip_softmax
            else "Softmax(QK^T)V"
        )

        max_error = max(
            projection_error,
            qkt_error,
            output_error,
        )

        passed = max_error < TOLERANCE

        print("\nVerification (final measured output only)")
        print(
            f"  {'Q/K/V projection error:':<28}"
            f"{projection_error:.3e}"
        )
        print(
            f"  {'QK^T error:':<28}"
            f"{qkt_error:.3e}"
        )
        print(
            f"  {(output_name + ' error:'):<28}"
            f"{output_error:.3e}"
        )
        print(
            f"  {'max_error:':<28}"
            f"{max_error:.3e}"
        )
        print(
            f"  {'result:':<28}"
            f"{'PASSED' if passed else 'FAILED'} "
            f"(tolerance={TOLERANCE:.1e})"
        )


if __name__ == "__main__":
    main()