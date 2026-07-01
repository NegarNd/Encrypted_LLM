"""Verify the ciphertext-domain GQA attention implementation with Orion.
Run from the parent directory of the `gqa` package, for example:
python -m gqa.ciphertext.verify_orion"""

from __future__ import annotations

from pathlib import Path
import orion
from ..plaintext.counter import counter
from .he_attention_orion import run_attention_gqa_he

# ── Encoding level defined once ─────────────────────────────────────────────
LEVEL = 7

CASES = [
    # Checked for small d values - PASS
    #d, H, n_kv, n_prefill
    (8, 4, 2, 5),
    # (8, 4, 2, 6),
    # (8, 4, 2, 3),
    # (8, 4, 2, 4),
    # (8, 4, 2, 10),
    # (16, 8, 4, 5),
    # (8, 4, 4, 3),
    # (8, 8, 1, 3),
    # (8, 8, 2, 4),
    # (16, 8, 1, 5),
    # (16, 16, 2, 4),
    # (8, 4, 2, 1),
    # (8, 4, 2, 2),
    # (8, 4, 2, 5),
    # (8, 4, 2, 8),
    # (16, 8, 1, 6),
    # (32, 8, 4, 7),
    # (32, 16, 2, 5),
    # (8, 1, 1, 4),
    # (8, 2, 2, 6),
    # Not Passing for larger d value! CHECK
    (128, 8, 4, 5)
]


def get_config_path(yml_name: str) -> str:
    here = Path(__file__).resolve().parent
    return str(here / "configs" / yml_name)


def main() -> None:
    config_path = get_config_path("gqa.yml")

    print(f"Encoding level: {LEVEL}")
    print()
    scheme = orion.init_scheme(config_path)
    n_he = 1 << (scheme.params.get_logn() - 1)


    for d, H, n_kv, n_prefill in CASES:
        counter.reset()
        err_qkt, err_v = run_attention_gqa_he(
            config_path=config_path,
            n_he = n_he,
            d=d,
            H=H,
            n_kv=n_kv,
            n_prefill=n_prefill,
            level=LEVEL,
            verify=True,
            verbose=True,
        )
        assert max(err_qkt, err_v) < 5e-3


if __name__ == "__main__":
    main()
