"""Packed PyTorch GQA K/V cache classes."""

from __future__ import annotations

import torch

from .dims import GQADims


class PackedCache:
    name = "PackedCache"

    def __init__(self, dims: GQADims):
        self.dims = dims
        self.ciphertexts: list[torch.Tensor] = []
        self.length = 0

    def append(self, positioned: torch.Tensor) -> None:
        if not isinstance(positioned, torch.Tensor):
            raise TypeError("positioned must be a torch.Tensor.")
        if tuple(positioned.shape) != (self.dims.n_he,):
            raise ValueError(f"value must have shape ({self.dims.n_he},).")
        if self.length % self.dims.t_p == 0:
            self.ciphertexts.append(torch.zeros_like(positioned))
        elif (
            positioned.device != self.ciphertexts[-1].device
            or positioned.dtype != self.ciphertexts[-1].dtype
        ):
            raise ValueError("All cache tensors must share one dtype and device.")
        self.ciphertexts[-1] = self.ciphertexts[-1] + positioned
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
