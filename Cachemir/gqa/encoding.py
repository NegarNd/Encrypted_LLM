"""Client-side slot encodings for X and projection weights."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .dims import (
    GQAConfig,
    GQADims,
    check_dims,
    gqa_group_input_index,
    gqa_kv_group_col,
    make_gqa_dims,
    normalize_heads,
)


def init_input(d: int, seed: int = 0) -> np.ndarray:
    """Generate a random dense input vector X of length d."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(d)


def init_weights(d: int, seed: int = 1) -> np.ndarray:
    """Generate a random dense square weight matrix."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, d))


def head_perm(d: int, H: int) -> List[int]:
    """Head-interleaving permutation at group granularity."""
    H = normalize_heads(H)
    d_h = d // H
    return [(g % H) * d_h + (g // H) for g in range(d)]


def gqa_lane_offsets(dims: GQADims) -> List[int]:
    """Lane offsets used by the compact slide encoding."""
    return head_perm(dims.d_kv, dims.n_kv)


def preprocess_input(X: np.ndarray, N: int, d: int) -> np.ndarray:
    """Encode X into the original interleaved-replicated layout."""
    check_dims(N, d)
    t = N // d
    enc = np.zeros(N, dtype=np.float64)
    enc[::t] = X

    i = 0
    while (1 << i) < t:
        enc = enc + np.roll(enc, -((1 << i) * (t - 1)))
        i += 1
    return enc


def preprocess_weights(B: np.ndarray, N: int, d: int) -> List[np.ndarray]:
    """Encode a dxd matrix into interleaved generalized diagonals."""
    check_dims(N, d)
    t = N // d
    n_iters = d // t
    return [
        np.array(
            [B[(s // t + s % t + it * t) % d][(s // t) % d] for s in range(N)],
            dtype=np.float64,
        )
        for it in range(n_iters)
    ]


def make_input(N: int, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dense input and its original Cachemir encoding."""
    X = init_input(d, seed)
    return X, preprocess_input(X, N, d)


def make_weights(N: int, d: int, seed: int = 1, H: int = 1) -> tuple[np.ndarray, List[np.ndarray]]:
    """Generate a square projection matrix and its encoded diagonals."""
    W = init_weights(d, seed)
    perm = head_perm(d, H)
    return W, preprocess_weights(W[:, perm], N, d)


def gqa_q_perm(dims: GQADims, c: int) -> List[int]:
    """Query-column permutation for one query group c."""
    return [
        ((g % dims.n_kv) * dims.ratio + c) * dims.d_h + (g // dims.n_kv)
        for g in range(dims.d_kv)
    ]


def preprocess_input_kv(X: np.ndarray, dims: GQADims) -> List[np.ndarray]:
    """Encode X into compact GQA chunks.

    There are ratio logical sparse inputs, one for each query group within a
    KV head. Each logical input may need R ciphertext chunks, so the returned
    list length is ratio * R.
    """
    lane_offsets = gqa_lane_offsets(dims)
    chunks: List[np.ndarray] = []

    for c in range(dims.ratio):
        for start in range(0, dims.d_kv, dims.t_p):
            offsets = lane_offsets[start : start + dims.t_p]
            enc = np.zeros(dims.N, dtype=np.float64)
            for g in range(dims.d_kv):
                base = g * dims.t_p
                for lane, off in enumerate(offsets):
                    row_g = (g + off) % dims.d_kv
                    row = gqa_group_input_index(c, row_g, dims)
                    enc[base + lane] = X[row]
            chunks.append(enc)

    return chunks


def preprocess_input_kv_from_params(
    X: np.ndarray, N: int, d: int, H: int, n_kv: int
) -> List[np.ndarray]:
    """Backward-compatible parameter-based compact input encoder."""
    return preprocess_input_kv(X, make_gqa_dims(GQAConfig(N=N, d=d, H=H, n_kv=n_kv)))


def make_weights_kv(
    dims: GQADims, seed: int = 1
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate and encode a compact GQA K/V projection matrix."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((dims.d, dims.d_kv))
    lane_offsets = gqa_lane_offsets(dims)

    chunks_enc: List[np.ndarray] = []
    for c in range(dims.ratio):
        for start in range(0, dims.d_kv, dims.t_p):
            offsets = lane_offsets[start : start + dims.t_p]
            enc = np.zeros(dims.N, dtype=np.float64)
            for g in range(dims.d_kv):
                col = gqa_kv_group_col(g, dims)
                base = g * dims.t_p
                for lane, off in enumerate(offsets):
                    row_g = (g + off) % dims.d_kv
                    row = gqa_group_input_index(c, row_g, dims)
                    enc[base + lane] = W[row, col]
            chunks_enc.append(enc)

    return W, chunks_enc


def make_weights_q_gqa(
    dims: GQADims, seed: int = 1
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """Generate and encode the full Q projection for GQA."""
    Wq = init_weights(dims.d, seed)
    lane_offsets = gqa_lane_offsets(dims)

    encs: List[List[np.ndarray]] = []
    for c_out in range(dims.ratio):
        chunks_enc: List[np.ndarray] = []
        q_perm = gqa_q_perm(dims, c_out)
        for c_in in range(dims.ratio):
            for start in range(0, dims.d_kv, dims.t_p):
                offsets = lane_offsets[start : start + dims.t_p]
                enc = np.zeros(dims.N, dtype=np.float64)
                for g in range(dims.d_kv):
                    col = q_perm[g]
                    base = g * dims.t_p
                    for lane, off in enumerate(offsets):
                        row_g = (g + off) % dims.d_kv
                        row = gqa_group_input_index(c_in, row_g, dims)
                        enc[base + lane] = Wq[row, col]
                chunks_enc.append(enc)
        encs.append(chunks_enc)

    return Wq, encs
