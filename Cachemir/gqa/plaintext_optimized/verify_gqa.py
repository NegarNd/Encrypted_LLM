"""Run and compare direct and BSGS GQA verification cases."""

from __future__ import annotations

import argparse

from .attention import run_attention_gqa


CASES = [
    (2**13, 4096, 8, 4, 1),
    # (32, 16, 4, 2, 3),
    # (64, 32, 8, 4, 5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv-method", choices=("direct", "bsgs"), default="direct")
    parser.add_argument("--baby-steps", type=int, default=None)
    parser.add_argument("--all-cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = CASES if args.all_cases else CASES[:1]
    for N, d, H, n_kv, n_prefill in cases:
        projection_error, qkt_error, output_error, _ = run_attention_gqa(
            N,
            d,
            H,
            n_kv,
            n_prefill,
            qkv_method=args.qkv_method,
            bsgs_baby_steps=args.baby_steps,
        )
        tolerance = 1e-9
        max_error = max(projection_error, qkt_error, output_error)
        passed = max_error < tolerance

        print(
            f"verification: {'PASSED' if passed else 'FAILED'} "
            f"(projection_error={projection_error:.3e}, "
            f"qkt_error={qkt_error:.3e}, "
            f"output_error={output_error:.3e}, "
            f"max_error={max_error:.3e}, "
            f"tolerance={tolerance:.1e})"
        )

        if not passed:
            raise AssertionError(
                f"Verification failed: max_error={max_error:.3e}"
            )

        print()


if __name__ == "__main__":
    main()
