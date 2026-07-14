"""HE-like primitive operations implemented with torch for simulation."""

from __future__ import annotations
from typing import List
import torch
from .dims import GQADims, check_dims
from .counter import counter


def vmm_kv(
    X_enc_chunks: List[torch.Tensor], W_enc_chunks: List[torch.Tensor], dims: GQADims, pos: int = 0
) -> torch.Tensor:
    """Compact GQA K/V projection over ratio * R encoded chunks.
    Output is positioned at slots pos, pos + t_p, pos + 2*t_p, ... so it
    can be accumulated directly into the packed cache.
    """
    out = torch.zeros(dims.n_he, dtype=torch.float64)

    mask = torch.zeros(dims.n_he, dtype=torch.float64)
    mask[pos::dims.t_p] = 1.0

    for Xc, Wc in zip(X_enc_chunks, W_enc_chunks):
        acc = Xc * Wc
        counter.ct_pt_mult += 1
        step, i = 1, 0
        while step < dims.t_p:
            counter.rotations += 1
            acc = acc + torch.roll(acc, +step if (pos >> i) & 1 else -step)
            step *= 2
            i += 1
        out += acc * mask

    return out


def block_replicate(v: torch.Tensor, t_p: int) -> torch.Tensor:
    """Replicate structural blocks with power-of-two rotate-add steps."""
    out = v.clone()
    step = 1
    while step < t_p:
        counter.rotations += 1
        out = out + torch.roll(out, step)
        step *= 2
    return out
