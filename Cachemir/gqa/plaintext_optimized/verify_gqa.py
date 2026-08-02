"""Run direct or BSGS PyTorch GQA verification cases."""

from __future__ import annotations

import argparse

from .attention import run_attention_gqa


CASES = [(32, 16, 4, 2, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv-method", choices=("direct", "bsgs"), default="direct")
    parser.add_argument("--baby-steps", type=int, default=None)
    parser.add_argument("--skip-softmax", action="store_true")
    parser.add_argument(
        "--softmax-method", choices=("exact", "approx"), default="approx"
    )
    parser.add_argument("--all-cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tolerance = 2e-5
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
            skip_softmax=args.skip_softmax,
            softmax_method=args.softmax_method,
        )
        max_error = max(projection_error, qkt_error, output_error)
        passed = max_error < tolerance
        print(
            f"verification: {'PASSED' if passed else 'FAILED'} "
            f"(projection_error={projection_error:.3e}, "
            f"qkt_error={qkt_error:.3e}, output_error={output_error:.3e}, "
            f"max_error={max_error:.3e}, tolerance={tolerance:.1e})"
        )
        if not passed:
            raise AssertionError(f"Verification failed: max_error={max_error:.3e}")


if __name__ == "__main__":
    main()
