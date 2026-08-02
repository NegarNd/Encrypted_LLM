"""Packed K/V cache classes for ciphertexts in the new GQA layout."""

from __future__ import annotations

from typing import Any

from ..plaintext_optimized.dims import GQADims


class PackedHECache:
    name = "PackedHECache"

    def __init__(self, dims: GQADims):
        self.dims = dims
        self.ciphertexts: list[Any] = []
        self.length = 0

    def append(self, positioned_ct: Any) -> None:
        """Add a ciphertext containing values only in the current token lane."""
        if self.length % self.dims.t_p == 0:
            self.ciphertexts.append(positioned_ct)
        else:
            self.ciphertexts[-1] = self.ciphertexts[-1] + positioned_ct
        self.length += 1

    def append_packed_block(self, block_ct: Any, token_count: int) -> None:
        """Append one prepacked client-generated cache block."""
        if token_count <= 0 or token_count > self.dims.t_p:
            raise ValueError(
                f"token_count must be in [1, {self.dims.t_p}]."
            )
        if self.length % self.dims.t_p:
            raise ValueError("A packed block must start at a cache boundary.")
        self.ciphertexts.append(block_ct)
        self.length += token_count

    def __len__(self) -> int:
        return self.length

    @property
    def num_ciphertexts(self) -> int:
        return len(self.ciphertexts)

    @property
    def blocks(self) -> list[Any]:
        return self.ciphertexts

    def __repr__(self) -> str:
        return (
            f"{self.name}(tokens={self.length}, "
            f"ciphertexts={self.num_ciphertexts}, t_p={self.dims.t_p})"
        )


class HEKCache(PackedHECache):
    name = "HEKCache"


class HEVCache(PackedHECache):
    name = "HEVCache"
