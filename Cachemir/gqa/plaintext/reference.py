"""Dense reference checks for the compact GQA simulation."""

from __future__ import annotations

from typing import Callable, List
import numpy as np
from .dims import GQADims


def decode_attention_map(
    att_cts: np.ndarray, n_tokens: int, dims: GQADims, pack_complex: bool = False
) -> np.ndarray:
    """Decode structural attention scores into a dense H x n_tokens map.

    `pack_complex` must reflect how the ciphertexts were actually produced
    (not merely whether `att_cts` happens to have a complex dtype -- a
    real-only run still round-trips through complex128 decoding, with a
    near-zero imaginary part that must NOT be interpreted as a second
    packed token). When True, each cache ciphertext (index `b`) is complex
    lane-packed and holds scores for *two* tokens per physical lane -- the
    real part for token `b*2*t_p + tok_in_ct` and the imaginary part for
    token `b*2*t_p + t_p + tok_in_ct` (see qkt_gqa_he / vmm_kv).
    """
    decoded = np.zeros((dims.H, n_tokens), dtype=np.float64)
    is_complex_dtype = np.iscomplexobj(att_cts)
    block = 2 * dims.t_p if pack_complex else dims.t_p

    for c in range(dims.ratio):
        for b in range(att_cts.shape[1]):
            for s in range(dims.n_he):
                within = s % dims.B
                kv = within // dims.t_p
                tok_in_ct = within % dims.t_p
                val = att_cts[c, b, s]

                tok = b * block + tok_in_ct
                if tok < n_tokens:
                    decoded[kv * dims.ratio + c, tok] = val.real if is_complex_dtype else val

                if pack_complex:
                    tok_odd = b * block + dims.t_p + tok_in_ct
                    if tok_odd < n_tokens:
                        decoded[kv * dims.ratio + c, tok_odd] = val.imag

    return decoded


def decode_output(O: np.ndarray, dims: GQADims) -> np.ndarray:
    """Decode output ciphertexts into dense H x d_h values."""
    O_dense = np.zeros((dims.H, dims.d_h), dtype=np.float64)

    for c in range(dims.ratio):
        for kv in range(dims.n_kv):
            for dim in range(dims.d_h):
                h = kv * dims.ratio + c
                O_dense[h, dim] = O[c][dim * dims.B + kv * dims.t_p]

    return O_dense


def reference_attention(
    toks: List[np.ndarray],
    x_new: np.ndarray,
    Wq_raw: np.ndarray,
    Wk_raw: np.ndarray,
    Wv_raw: np.ndarray,
    dims: GQADims,
    head_perm: Callable[[int, int], list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dense reference QK^T and scores*V results."""
    all_toks = toks + [x_new]
    n_tokens = len(all_toks)
    perm_kv = head_perm(dims.d_kv, dims.n_kv)

    K_matrix = np.array([x @ Wk_raw[:, perm_kv] for x in all_toks]).T
    V_matrix = np.array([x @ Wv_raw[:, perm_kv] for x in all_toks])

    Qf = x_new @ Wq_raw

    ref_map = np.zeros((dims.H, n_tokens), dtype=np.float64)
    for h in range(dims.H):
        kv = h // dims.ratio
        for tok in range(n_tokens):
            for i in range(dims.d_h):
                ref_map[h, tok] += Qf[h * dims.d_h + i] * K_matrix[i * dims.n_kv + kv, tok]

    def score(h: int, tok: int) -> float:
        kv = h // dims.ratio
        return sum(
            Qf[h * dims.d_h + i] * K_matrix[i * dims.n_kv + kv, tok]
            for i in range(dims.d_h)
        )

    ref_O = np.zeros((dims.H, dims.d_h), dtype=np.float64)
    for h in range(dims.H):
        kv = h // dims.ratio
        for dim in range(dims.d_h):
            g = dim * dims.n_kv + kv
            ref_O[h, dim] = sum(V_matrix[tok, g] * score(h, tok) for tok in range(n_tokens))

    return ref_map, ref_O


def compare_attention_outputs(
    toks: List[np.ndarray],
    x_new: np.ndarray,
    Wq_raw: np.ndarray,
    Wk_raw: np.ndarray,
    Wv_raw: np.ndarray,
    att_cts: np.ndarray,
    O: np.ndarray,
    dims: GQADims,
    head_perm: Callable[[int, int], list[int]],
    pack_complex: bool = False,
) -> tuple[float, float]:
    """Return max absolute errors for QK^T and scores*V."""
    n_tokens = len(toks) + 1
    decoded_map = decode_attention_map(att_cts, n_tokens, dims, pack_complex=pack_complex)
    O_dense = decode_output(O, dims)
    ref_map, ref_O = reference_attention(toks, x_new, Wq_raw, Wk_raw, Wv_raw, dims, head_perm)
    return float(np.max(np.abs(decoded_map - ref_map))), float(np.max(np.abs(O_dense - ref_O)))
