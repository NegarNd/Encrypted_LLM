"""Client-side encodings for persistent X and Q/K/V weight diagonals."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .dims import GQADims


def init_input(d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(d)


def init_weights(rows: int, cols: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, cols))


def encode_input(x: np.ndarray, dims: GQADims) -> np.ndarray:
    """Encode X persistently as repeated dense d-slot copies."""
    value = np.asarray(x, dtype=np.float64)
    if tuple(value.shape) != (dims.d,):
        raise ValueError(f"x must have shape ({dims.d},).")
    return np.tile(value, dims.n_he // dims.d)


def q_column(group: int, dims: GQADims) -> int:
    """Dense Q column stored at compact output group ``group``."""
    compact = group // dims.ratio
    query_in_group = group % dims.ratio
    dim = compact // dims.n_kv
    kv_head = compact % dims.n_kv
    query_head = kv_head * dims.ratio + query_in_group
    return query_head * dims.d_h + dim


def kv_column(compact_group: int, dims: GQADims) -> int:
    dim = compact_group // dims.n_kv
    kv_head = compact_group % dims.n_kv
    return kv_head * dims.d_h + dim


def _encode_projection_diagonals(
    weights: np.ndarray,
    output_columns: List[int],
    dims: GQADims,
) -> List[np.ndarray]:
    """Encode the m_p partial-dot-product diagonals.

    Slot ``compact_group*t_p + lane`` receives the contribution from input
    row ``(slot + r*t_p) mod d`` for direct rotation r.
    """
    diagonals = [
        np.zeros(dims.n_he, dtype=np.float64)
        for _ in range(dims.m_p)
    ]

    for compact_group, column in enumerate(output_columns):
        base = compact_group * dims.t_p
        for lane in range(dims.t_p):
            slot = base + lane
            for r in range(dims.m_p):
                row = (slot + r * dims.t_p) % dims.d
                diagonals[r][slot] = weights[row, column]

    return diagonals


def make_weights_q_gqa(
    dims: GQADims,
    seed: int = 1,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    Wq = init_weights(dims.d, dims.d, seed)
    encoded: List[List[np.ndarray]] = []

    for c in range(dims.ratio):
        columns = [
            q_column(compact * dims.ratio + c, dims)
            for compact in range(dims.d_kv)
        ]
        encoded.append(_encode_projection_diagonals(Wq, columns, dims))

    return Wq, encoded


def make_weights_kv(
    dims: GQADims,
    seed: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    W = init_weights(dims.d, dims.d_kv, seed)
    columns = [kv_column(compact, dims) for compact in range(dims.d_kv)]
    return W, _encode_projection_diagonals(W, columns, dims)
