"""Dense reference checks for persistent-layout GQA."""

from __future__ import annotations

from typing import List

import numpy as np

from .dims import GQADims


def decode_q(q_cts: np.ndarray, dims: GQADims) -> np.ndarray:
    q = np.zeros(dims.d, dtype=np.float64)
    for c in range(dims.ratio):
        for dim in range(dims.d_h):
            for kv in range(dims.n_kv):
                compact = dim * dims.n_kv + kv
                head = kv * dims.ratio + c
                q[head * dims.d_h + dim] = q_cts[c, compact * dims.t_p]
    return q


def decode_positioned_kv(value: np.ndarray, pos: int, dims: GQADims) -> np.ndarray:
    out = np.zeros(dims.d_kv, dtype=np.float64)
    for dim in range(dims.d_h):
        for kv in range(dims.n_kv):
            compact = dim * dims.n_kv + kv
            out[kv * dims.d_h + dim] = value[compact * dims.t_p + pos]
    return out


def decode_output(o_cts: np.ndarray, dims: GQADims) -> np.ndarray:
    out = np.zeros(dims.d, dtype=np.float64)
    for c in range(dims.ratio):
        for dim in range(dims.d_h):
            for kv in range(dims.n_kv):
                compact = dim * dims.n_kv + kv
                head = kv * dims.ratio + c
                out[head * dims.d_h + dim] = o_cts[c, compact * dims.t_p]
    return out


def decode_qkt(
    attention: np.ndarray,
    n_tokens: int,
    dims: GQADims,
) -> np.ndarray:
    """Decode packed QK^T scores into the dense [H, n_tokens] layout."""
    scores = np.zeros((dims.H, n_tokens), dtype=np.float64)
    for c in range(dims.ratio):
        for kv in range(dims.n_kv):
            head = kv * dims.ratio + c
            for token in range(n_tokens):
                block = token // dims.t_p
                lane = token % dims.t_p
                slot = kv * dims.t_p + lane
                scores[head, token] = attention[c, block, slot]
    return scores


def reference_attention(
    tokens: List[np.ndarray],
    x_new: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    dims: GQADims,
) -> tuple[np.ndarray, np.ndarray]:
    all_tokens = tokens + [x_new]
    q = x_new @ Wq
    keys = [x @ Wk for x in all_tokens]
    values = [x @ Wv for x in all_tokens]

    scores = np.zeros((dims.H, len(all_tokens)), dtype=np.float64)
    output = np.zeros(dims.d, dtype=np.float64)
    for h in range(dims.H):
        kv = h // dims.ratio
        qh = q[h * dims.d_h : (h + 1) * dims.d_h]
        kh = np.stack(
            [k[kv * dims.d_h : (kv + 1) * dims.d_h] for k in keys]
        )
        vh = np.stack(
            [v[kv * dims.d_h : (kv + 1) * dims.d_h] for v in values]
        )
        scores[h] = kh @ qh
        shifted = scores[h] - np.max(scores[h])
        probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
        output[h * dims.d_h : (h + 1) * dims.d_h] = probabilities @ vh
    return scores, output
