"""Client-side slot encodings for X and projection weights."""

from __future__ import annotations
from typing import List, Tuple
import torch

from .dims import (
    GQADims,
    gqa_group_input_index,
    gqa_kv_group_col,
    normalize_heads,
)

from .counter import counter

def init_input(d: int, seed: int = 0) -> torch.Tensor:
    """Generate a random dense input vector X of length d."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=gen, dtype=torch.float64)


def init_weights(d: int, seed: int = 1) -> torch.Tensor:
    """Generate a random dense square weight matrix."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((d, d), generator=gen, dtype=torch.float64)


def head_perm(d: int, H: int) -> List[int]:
    """Head-interleaving permutation at group granularity."""
    H = normalize_heads(H)
    d_h = d // H
    return [(g % H) * d_h + (g // H) for g in range(d)]


def gqa_lane_offsets(dims: GQADims) -> List[int]:
    """Lane offsets used by the compact slide encoding."""
    return head_perm(dims.d_kv, dims.n_kv)


def gqa_q_perm(dims: GQADims, c: int) -> List[int]:
    """Query-column permutation for one query group c."""
    return [
        ((g % dims.n_kv) * dims.ratio + c) * dims.d_h + (g // dims.n_kv)
        for g in range(dims.d_kv)
    ]



def make_sparse_input_kv(X: torch.Tensor, dims: GQADims) -> List[torch.Tensor]:
    """Create sparse GQA input vectors.

    This is the client-side plaintext packing only.

    Returns one sparse vector per query group c.
    These sparse vectors should be encoded/encrypted before HE rotations.

    For each c:
        sparse[g * t_p] = X[gqa_group_input_index(c, g, dims)]
        all other slots are zero.
    """
    sparse_inputs: List[torch.Tensor] = []

    for c in range(dims.ratio):
        enc = torch.zeros(dims.n_he, dtype=torch.float64)

        for g in range(dims.d_kv):
            slot = g * dims.t_p
            row = gqa_group_input_index(c, g, dims)
            enc[slot] = X[row]

        sparse_inputs.append(enc)

    return sparse_inputs


def expand_sparse_input_kv_plain(
    sparse_inputs: List[torch.Tensor], dims: GQADims
) -> List[torch.Tensor]:
    """Plaintext simulator for the HE rotations used to expand sparse X.
    """
    lane_offsets = gqa_lane_offsets(dims)
    chunks: List[torch.Tensor] = []

    for sparse in sparse_inputs:
        for start in range(0, dims.d_kv, dims.t_p):
            offsets = lane_offsets[start : start + dims.t_p]

            acc = torch.zeros(dims.n_he, dtype=torch.float64)

            for lane, off in enumerate(offsets):
                shift = off * dims.t_p - lane
                counter.rotations +=1
                acc += torch.roll(sparse, -shift)

            chunks.append(acc)

    return chunks

def make_weights_kv(
    dims: GQADims, seed: int = 1
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Generate and encode a compact GQA K/V projection matrix."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn((dims.d, dims.d_kv), generator=gen, dtype=torch.float64)
    lane_offsets = gqa_lane_offsets(dims)

    chunks_enc: List[torch.Tensor] = []
    for c in range(dims.ratio):
        for start in range(0, dims.d_kv, dims.t_p):
            offsets = lane_offsets[start : start + dims.t_p]
            enc = torch.zeros(dims.n_he, dtype=torch.float64)
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
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """Generate and encode the full Q projection for GQA."""
    Wq = init_weights(dims.d, seed)
    lane_offsets = gqa_lane_offsets(dims)

    encs: List[List[torch.Tensor]] = []
    for c_out in range(dims.ratio):
        chunks_enc: List[torch.Tensor] = []
        q_perm = gqa_q_perm(dims, c_out)
        for c_in in range(dims.ratio):
            for start in range(0, dims.d_kv, dims.t_p):
                offsets = lane_offsets[start : start + dims.t_p]
                enc = torch.zeros(dims.n_he, dtype=torch.float64)
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
