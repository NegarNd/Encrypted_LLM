"""HE-like primitive operations implemented with NumPy for simulation."""

from __future__ import annotations
from typing import List
import numpy as np
from .dims import GQADims, check_dims
from .counter import counter


def vmm_kv(
    X_enc_chunks: List[np.ndarray], W_enc_chunks: List[np.ndarray], dims: GQADims, pos: int = 0
) -> np.ndarray:
    """Compact GQA K/V projection over ratio * R encoded chunks.
    Output is positioned at slots pos, pos + t_p, pos + 2*t_p, ... so it
    can be accumulated directly into the packed cache.
    """
    out = np.zeros(dims.n_he, dtype=np.float64)

    mask = np.zeros(dims.n_he, dtype=np.float64)
    mask[pos::dims.t_p] = 1.0

    for Xc, Wc in zip(X_enc_chunks, W_enc_chunks):
        acc = Xc * Wc
        counter.ct_pt_mult += 1
        step, i = 1, 0
        while step < dims.t_p:
            counter.rotations += 1
            acc += np.roll(acc, +step if (pos >> i) & 1 else -step)
            step *= 2
            i += 1
        out += acc * mask

    return out


def block_replicate(v: np.ndarray, t_p: int) -> np.ndarray:
    """Replicate structural blocks with power-of-two rotate-add steps."""
    out = v.copy()
    step = 1
    while step < t_p:
        counter.rotations += 1 
        out = out + np.roll(out, step)
        step *= 2
    return out
