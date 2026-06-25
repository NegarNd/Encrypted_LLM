"""
k_cache.py
==========
Interleaved K cache for Cachemir.

The cache packs t = N/d tokens into a single ciphertext, interleaved.
This module is stateful by design to grow across decoding steps.
"""

from typing import List
import numpy as np
from cachemir_attention import (
    check_dims,
    preprocess_input,
    preprocess_weights,
    vmm,
)


class KCache:
    """
    Interleaved K cache that packs t = N/d tokens per ciphertext.

    Attributes:
        N (int): Polynomial degree / slot count (power of two).
        d (int): Feature dimension.
    """

    def __init__(self, N: int, d: int):
        check_dims(N, d)
        self.N = N
        self.d = d
        self.t = N // d
        self.ciphertexts: List[np.ndarray] = []
        self.length: int = 0

    @property
    def num_ciphertexts(self) -> int:
        """Number of ciphertexts holding the cache."""
        return len(self.ciphertexts)

    def append(self, k_positioned: np.ndarray) -> None:
        """
        Append one token's K vector to the cache.
        """
        k = np.asarray(k_positioned, dtype=np.float64)
        if k.shape != (self.N,):
            raise ValueError(f"k must have shape ({self.N},), got {k.shape}.")

        pos = self.length % self.t
        if pos == 0:
            self.ciphertexts.append(np.zeros(self.N, dtype=np.float64))
        
        self.ciphertexts[-1] += k
        self.length += 1

    def append_from_input(self, X: np.ndarray, Wk: np.ndarray) -> None:
        """Project input X through weight Wk at the correct group offset and append."""
        pos = self.length % self.t
        k = vmm(
            preprocess_input(X, self.N, self.d),
            preprocess_weights(Wk, self.N, self.d),
            self.N,
            self.d,
            pos=pos,
        )
        self.append(k)

    def get_token(self, i: int) -> np.ndarray:
        """Reconstruct the dense length-d K vector of cached token i."""
        if not 0 <= i < self.length:
            raise IndexError(f"token {i} out of range [0, {self.length}).")
        
        ct = self.ciphertexts[i // self.t]
        off = i % self.t
        return np.array([ct[j * self.t + off] for j in range(self.d)], dtype=np.float64)

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (
            f"KCache(N={self.N}, d={self.d}, t={self.t}, "
            f"tokens={self.length}, ciphertexts={self.num_ciphertexts})"
        )