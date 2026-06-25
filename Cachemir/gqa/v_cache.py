"""
v_cache.py
==========
Interleaved V cache for Cachemir GQA.
"""

from typing import List
import numpy as np
from cachemir_attention import check_dims, normalize_heads


class VCache:
    """
    V cache for Cachemir Grouped-Query Attention.

    Attributes:
        N (int): Polynomial degree / slot count.
        d (int): K/V projection output dimension.
        H (int): Number of KV heads.
    """

    def __init__(self, N: int, d: int, H: int = 1):
        check_dims(N, d)
        self.N = N
        self.d = d
        self.H = normalize_heads(H)
        self.t_p = N // d
        self.d_h = d // self.H
        self.B = self.H * self.t_p
        self.blocks: List[List[np.ndarray]] = []
        self.length: int = 0

    @property
    def n_k_ct(self) -> int:
        """Number of K-ct blocks."""
        return len(self.blocks)

    @property
    def num_ciphertexts(self) -> int:
        """Total ciphertexts stored."""
        return self.d_h * len(self.blocks)

    def append(self, v_new: np.ndarray) -> None:
        """Append a new V block to the cache streams."""
        v = np.asarray(v_new, dtype=np.float64)
        if v.shape != (self.N,):
            raise ValueError(f"v must have shape ({self.N},), got {v.shape}.")

        att_group_size = self.d_h * self.t_p
        tok_in_att_group = self.length % att_group_size
        k_local = tok_in_att_group // self.t_p
        tok_in_kct = tok_in_att_group % self.t_p

        if tok_in_att_group == 0:
            self.blocks.append(
                [np.zeros(self.N, dtype=np.float64) for _ in range(self.d_h)]
            )

        for g in range(self.d):
            dim = g // self.H
            kv = g % self.H
            slot = k_local * self.B + kv * self.t_p + tok_in_kct
            self.blocks[-1][dim][slot] += v[g * self.t_p]

        self.length += 1

    def get_token(self, i: int) -> np.ndarray:
        """Reconstruct the token value vector from localized slots."""
        if not 0 <= i < self.length:
            raise IndexError(f"token {i} out of range [0, {self.length}).")

        att_group_size = self.d_h * self.t_p
        block_id = i // att_group_size
        tok_in_att_group = i % att_group_size
        k_local = tok_in_att_group // self.t_p
        tok_in_kct = tok_in_att_group % self.t_p

        out = np.zeros(self.d, dtype=np.float64)
        for g in range(self.d):
            dim = g // self.H
            kv = g % self.H
            slot = k_local * self.B + kv * self.t_p + tok_in_kct
            out[g] = self.blocks[block_id][dim][slot]

        return out

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (
            f"VCache(N={self.N}, d_kv={self.d}, n_kv={self.H}, "
            f"t_p={self.t_p}, d_h={self.d_h}, B={self.B}, "
            f"tokens={self.length}, n_k_ct={self.n_k_ct}, "
            f"ciphertexts={self.num_ciphertexts})"
        )