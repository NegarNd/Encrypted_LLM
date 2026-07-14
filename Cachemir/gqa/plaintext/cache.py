"""Packed K/V cache classes."""

from __future__ import annotations

from typing import List

import torch

from .dims import check_dims, normalize_heads


class PackedCache:
    """Cache that packs t tokens into one ciphertext."""

    name = "PackedCache"

    def __init__(self, N: int, d: int):
        check_dims(N, d)
        self.N = N
        self.d = d
        self.t = N // d
        self.ciphertexts: List[torch.Tensor] = []
        self.length = 0

    @property
    def num_ciphertexts(self) -> int:
        return len(self.ciphertexts)

    @property
    def blocks(self) -> List[torch.Tensor]:
        return self.ciphertexts

    def append(self, positioned: torch.Tensor) -> None:
        value = torch.as_tensor(positioned, dtype=torch.float64)
        if tuple(value.shape) != (self.N,):
            raise ValueError(f"value must have shape ({self.N},), got {tuple(value.shape)}.")

        pos = self.length % self.t
        if pos == 0:
            self.ciphertexts.append(torch.zeros(self.N, dtype=torch.float64))

        self.ciphertexts[-1] += value
        self.length += 1

    def get_token(self, i: int) -> torch.Tensor:
        if not 0 <= i < self.length:
            raise IndexError(f"token {i} out of range [0, {self.length}).")

        ct = self.ciphertexts[i // self.t]
        tok_in_ct = i % self.t
        return torch.stack([ct[g * self.t + tok_in_ct] for g in range(self.d)])

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (
            f"{self.name}(N={self.N}, d={self.d}, t={self.t}, "
            f"tokens={self.length}, ciphertexts={self.num_ciphertexts})"
        )


class KCache(PackedCache):
    """K cache. Kept as a separate class for semantic clarity."""

    name = "KCache"


class VCache(PackedCache):
    """V cache with GQA naming aliases."""

    name = "VCache"

    def __init__(self, N: int, d: int, H: int = 1):
        super().__init__(N, d)
        self.H = normalize_heads(H)
        self.t_p = self.t
        self.d_h = d // self.H
        self.B = self.H * self.t_p

    @property
    def n_k_ct(self) -> int:
        return len(self.ciphertexts)

    def __repr__(self) -> str:
        return (
            f"VCache(N={self.N}, d_kv={self.d}, n_kv={self.H}, "
            f"t_p={self.t_p}, d_h={self.d_h}, B={self.B}, "
            f"tokens={self.length}, n_k_ct={self.n_k_ct}, "
            f"ciphertexts={self.num_ciphertexts})"
        )
