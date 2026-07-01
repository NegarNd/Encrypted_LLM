"""Packed K/V cache classes for Orion ciphertexts."""

from __future__ import annotations

from typing import Any, List
from ..plaintext.dims import check_dims, normalize_heads


class PackedHECache:
    """Cache that packs t tokens into one ciphertext.

    The plaintext cache used np.zeros(N) for a new packed block.  In HE, a new
    block starts as the first positioned ciphertext itself; later tokens are
    homomorphically added into the same packed block.
    """

    name = "PackedHECache"

    def __init__(self, n_he: int, d: int):
        check_dims(n_he, d)
        self.n_he = n_he
        self.d = d
        self.t = n_he // d
        self.ciphertexts: List[Any] = []
        self.length = 0

    @property
    def num_ciphertexts(self) -> int:
        return len(self.ciphertexts)

    @property
    def blocks(self) -> List[Any]:
        return self.ciphertexts

    def append(self, positioned_ct: Any) -> None:
        pos = self.length % self.t
        if pos == 0:
            self.ciphertexts.append(positioned_ct)
        else:
            self.ciphertexts[-1] = self.ciphertexts[-1] + positioned_ct
        # print(self.ciphertexts[-1].type())
        self.length += 1

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (
            f"{self.name}(N={self.N}, d={self.d}, t={self.t}, "
            f"tokens={self.length}, ciphertexts={self.num_ciphertexts})"
        )


class HEKCache(PackedHECache):
    name = "HEKCache"


class HEVCache(PackedHECache):
    name = "HEVCache"

    def __init__(self, n_he: int, d: int, H: int = 1):
        super().__init__(n_he, d)
        self.H = normalize_heads(H)
        self.t_p = self.t
        self.d_h = d // self.H
        self.B = self.H * self.t_p

    @property
    def n_k_ct(self) -> int:
        return len(self.ciphertexts)
