"""Validation tests: complex lane packing vs. the original real-only layout.

These tests run the same encrypted GQA decoding step twice -- once with
`pack_complex=True` (2 tokens/ciphertext via real+imaginary slot packing)
and once with `pack_complex=False` (the original 1 token/ciphertext,
real-only layout) -- using identical seeds/config, and assert that the
decrypted outputs match. Packing is purely an implementation-level
optimization: it must not change the attention math being computed.

Run with pytest:
    cd Cachemir && python -m pytest gqa/ciphertext/test_complex_packing.py -v

Or standalone:
    cd Cachemir && python -m gqa.ciphertext.test_complex_packing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orion
import pytest

from ..plaintext.counter import counter
from .he_attention_orion import run_attention_gqa_he

# Chosen so that the K/V cache spans multiple ciphertexts under the
# real-only layout (t_p=8 for this config) but collapses to fewer under
# complex packing (2*t_p=16) -- i.e. large enough to actually exercise the
# optimization, not just the degenerate single-ciphertext case.
CASE = dict(d=128, H=8, n_kv=4, n_prefill=9, level=7)


def _config_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here / "configs" / "gqa.yml")


def _init_scheme():
    config_path = _config_path()
    scheme = orion.init_scheme(config_path)
    n_he = 1 << (scheme.params.get_logn() - 1)
    return config_path, n_he


@pytest.fixture(scope="module")
def scheme_info():
    return _init_scheme()


def test_complex_vs_noncomplex_outputs_match(scheme_info):
    """Decrypted attention scores and outputs must match between the two
    packing modes (within CKKS noise), and the complex-packed run must use
    strictly fewer (or equal) cache ciphertexts."""
    config_path, n_he = scheme_info

    counter.reset()
    err_qkt_c, err_v_c, stats_complex = run_attention_gqa_he(
        config_path=config_path, n_he=n_he, verbose=False,
        return_stats=True, pack_complex=True, **CASE,
    )

    counter.reset()
    err_qkt_r, err_v_r, stats_real = run_attention_gqa_he(
        config_path=config_path, n_he=n_he, verbose=False,
        return_stats=True, pack_complex=False, **CASE,
    )

    # Both modes must independently agree with the plaintext reference.
    assert err_qkt_c < 5e-3 and err_v_c < 2e-2
    assert err_qkt_r < 5e-3 and err_v_r < 2e-2

    # The two modes must agree with EACH OTHER, not just with the
    # reference (packing must not change the computed values).
    att_diff = np.max(np.abs(stats_complex["att_dense"] - stats_real["att_dense"]))
    O_diff = np.max(np.abs(stats_complex["O_dense"] - stats_real["O_dense"]))
    assert att_diff < 2e-2, f"attention scores diverged between packing modes: {att_diff:.2e}"
    assert O_diff < 2e-2, f"attention output diverged between packing modes: {O_diff:.2e}"

    # The optimization should actually engage for this config: fewer cache
    # ciphertexts (memory) and at least one conjugation under packing, none
    # without it.
    cc_complex = stats_complex["ciphertext_counts"]
    cc_real = stats_real["ciphertext_counts"]
    assert cc_complex["kcache"] < cc_real["kcache"]
    assert cc_complex["vcache"] < cc_real["vcache"]
    assert stats_complex["ops_total"]["conjugations"] > 0
    assert stats_real["ops_total"]["conjugations"] == 0

    # ct-ct multiplications (the dominant cost) must not increase.
    assert stats_complex["ops_total"]["ct_ct_mult"] <= stats_real["ops_total"]["ct_ct_mult"]


def test_noncomplex_matches_real_only_baseline(scheme_info):
    """pack_complex=False must reproduce exactly the historical real-only
    behavior: zero conjugations, and cache capacity `n_he // d_kv` tokens
    per ciphertext (not doubled)."""
    config_path, n_he = scheme_info

    counter.reset()
    err_qkt, err_v, stats = run_attention_gqa_he(
        config_path=config_path, n_he=n_he, verbose=False,
        return_stats=True, pack_complex=False, **CASE,
    )
    assert stats["ops_total"]["conjugations"] == 0
    assert err_qkt < 5e-3 and err_v < 2e-2


if __name__ == "__main__":
    config_path, n_he = _init_scheme()
    test_complex_vs_noncomplex_outputs_match((config_path, n_he))
    test_noncomplex_matches_real_only_baseline((config_path, n_he))
    print("All complex-packing validation tests PASSED.")
