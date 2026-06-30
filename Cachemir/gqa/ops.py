"""HE-like primitive operations implemented with NumPy for simulation."""

from __future__ import annotations
from typing import List
import numpy as np
from .dims import GQADims, check_dims
from .counter import counter


def rotate(ct: np.ndarray, step: int) -> np.ndarray:
    """Rotate slots using NumPy's roll convention."""
    return np.roll(ct, step)


def vmm(X_enc: np.ndarray, W_list: List[np.ndarray], N: int, d: int, pos: int = 0) -> np.ndarray:
    """Original interleaved-replicated vector-matrix multiply."""
    check_dims(N, d)
    t = N // d
    acc = np.zeros(N, dtype=np.float64)

    for it, W_it in enumerate(W_list):
        if it != 0:
            counter.rotations += 1
        rotated = np.roll(X_enc, -(it * t * t))
        counter.ct_pt_mults += 1
        acc += rotated * W_it

    stride, i = 1, 0
    while stride < t:
        counter.rotations += 1
        acc += np.roll(acc, +stride if (pos >> i) & 1 else -stride)
        stride *= 2
        i += 1

    mask = np.zeros(N, dtype=np.float64)
    mask[pos::t] = 1.0
    return acc * mask


def extract(out_interleaved: np.ndarray, N: int, d: int) -> np.ndarray:
    """Extract a dense length-d output vector from interleaved slots."""
    check_dims(N, d)
    t = N // d
    return out_interleaved[::t][:d].copy()


def vmm_kv(
    X_enc_chunks: List[np.ndarray], W_enc_chunks: List[np.ndarray], dims: GQADims, pos: int = 0
) -> np.ndarray:
    """Compact GQA K/V projection over ratio * R encoded chunks."""
    out = np.zeros(dims.N, dtype=np.float64)

    mask = np.zeros(dims.N, dtype=np.float64)
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
