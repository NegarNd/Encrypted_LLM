"""Run compact GQA verification cases."""
from .attention import run_attention_gqa
from .counter import counter

CASES = [
    # (16, 8, 4, 2, 6),
    # (16, 8, 4, 2, 3),
    # (16, 8, 4, 2, 4),
    # (16, 8, 4, 2, 10),
    # (32, 16, 8, 4, 5),
    # (16, 8, 4, 4, 3),
    # (16, 8, 8, 1, 3),
    # (16, 8, 8, 2, 4),
    # (32, 16, 8, 1, 5),
    # (32, 16, 16, 2, 4),
    # (16, 8, 4, 2, 1),
    # (16, 8, 4, 2, 2),
    # (16, 8, 4, 2, 5),
    # (16, 8, 4, 2, 8),
    # (64, 16, 8, 1, 6),
    # (64, 32, 8, 4, 7),
    # (64, 32, 16, 2, 5),
    # (16, 8, 1, 1, 4),
    # (32, 8, 2, 2, 6),
    (512,  256, 8, 4, 5)
]


def main() -> None:
    for N, d, H, n_kv, n_prefill in CASES:
        err_qkt, err_v = run_attention_gqa(N, d, H, n_kv, n_prefill)
        print(
            f"ops: rotations={counter.rotations}, "
            f"ct-pt={counter.ct_pt_mult}, "
            f"ct-ct={counter.ct_ct_mult}"
        )
        assert max(err_qkt, err_v) < 1e-9


if __name__ == "__main__":
    main()
