"""Packed GQA K/V cache classes."""

from __future__ import annotations

from typing import List

import numpy as np

from .dims import GQADims


class PackedCache:
    """Pack t_p token positions into each ciphertext."""

    name = "PackedCache"

    def __init__(self, dims: GQADims):
        self.dims = dims
        self.ciphertexts: List[np.ndarray] = []
        self.length = 0

    def append(self, positioned: np.ndarray) -> None:
        value = np.asarray(positioned, dtype=np.float64)
        if tuple(value.shape) != (self.dims.n_he,):
            raise ValueError(f"value must have shape ({self.dims.n_he},).")
        if self.length % self.dims.t_p == 0:
            self.ciphertexts.append(np.zeros_like(value))
        self.ciphertexts[-1] += value
        self.length += 1

    def __len__(self) -> int:
        return self.length

    @property
    def num_ciphertexts(self) -> int:
        return len(self.ciphertexts)

    def __repr__(self) -> str:
        return (
            f"{self.name}(tokens={self.length}, "
            f"ciphertexts={self.num_ciphertexts}, t_p={self.dims.t_p})"
        )


class KCache(PackedCache):
    name = "KCache"


class VCache(PackedCache):
    name = "VCache"
