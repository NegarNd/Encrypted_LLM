"""PyTorch encodings for persistent X and Q/K/V weight diagonals."""

from __future__ import annotations

from typing import List, Tuple

import torch

from .dims import GQADims


def init_input(d: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=generator)


def init_weights(
    rows: int,
    cols: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((rows, cols), generator=generator)


def encode_input(x: torch.Tensor, dims: GQADims) -> torch.Tensor:
    if tuple(x.shape) != (dims.d,):
        raise ValueError(f"x must have shape ({dims.d},).")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor.")
    return x.repeat(dims.n_he // dims.d)


def q_column(group: int, dims: GQADims) -> int:
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
    weights: torch.Tensor,
    output_columns: List[int],
    dims: GQADims,
) -> List[torch.Tensor]:
    diagonals = [weights.new_zeros(dims.n_he) for _ in range(dims.m_p)]
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
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    Wq = init_weights(dims.d, dims.d, seed)
    encoded = []
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
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    W = init_weights(dims.d, dims.d_kv, seed)
    columns = [kv_column(compact, dims) for compact in range(dims.d_kv)]
    return W, _encode_projection_diagonals(W, columns, dims)
