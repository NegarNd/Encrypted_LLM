"""Dimension checks and derived GQA parameters. """

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


def check_dims(n_he: int, d: int) -> None:
    """Validate the base Cachemir dimensions."""
    if n_he <= 0 or n_he & (n_he - 1) != 0:
        raise ValueError(f"n_he ({n_he}) must be a positive power of two.")
    if d <= 0:
        raise ValueError(f"d ({d}) must be positive.")
    if n_he % d != 0:
        raise ValueError(f"n_he ({n_he}) must be divisible by d ({d}).")


def normalize_heads(H: int) -> int:
    """Treat H=0 and H=1 as the single-head case."""
    return 1 if H in (0, 1) else H


@dataclass(frozen=True)
class GQAConfig:
    """User-facing GQA configuration."""

    n_he: int
    d: int
    H: int
    n_kv: int
    n_prefill: int = 0


@dataclass(frozen=True)
class GQADims:
    """Derived dimensions for compact GQA encoding.

    ratio:
        Number of query heads that share each KV head, ratio = H / n_kv.
    R:
        Number of chunks per query-group input encoding. The total encoded
        input chunks are ratio * R.
    """

    n_he: int
    d: int
    H: int
    n_kv: int
    d_h: int
    ratio: int
    d_kv: int
    t_p: int
    B: int
    R: int

    @property
    def input_chunks(self) -> int:
        return self.ratio * self.R


def make_gqa_dims(config: GQAConfig) -> GQADims:
    """Calculate and validate all compact GQA dimensions."""
    check_dims(config.n_he, config.d)
    H = normalize_heads(config.H)

    if config.n_kv <= 0:
        raise ValueError(f"n_kv ({config.n_kv}) must be positive.")
    if config.d % H != 0:
        raise ValueError(f"d ({config.d}) must be divisible by H ({H}).")
    if H % config.n_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by n_kv ({config.n_kv}).")

    d_h = config.d // H
    ratio = H // config.n_kv
    d_kv = config.n_kv * d_h

    if config.n_he % d_kv != 0:
        raise ValueError(f"N ({config.N}) must be divisible by d_kv ({d_kv}).")

    t_p = config.n_he // d_kv
    B = config.n_kv * t_p
    if config.n_he != d_h * B:
        raise ValueError(f"N ({config.N}) must equal d_h*B = {d_h * B}.")

    R = max(ceil(d_kv / t_p), 1)
    return GQADims(
        n_he=config.n_he,
        d=config.d,
        H=H,
        n_kv=config.n_kv,
        d_h=d_h,
        ratio=ratio,
        d_kv=d_kv,
        t_p=t_p,
        B=B,
        R=R,
    )


def gqa_group_input_index(c: int, g: int, dims: GQADims) -> int:
    """Input row for query-group chunk c and compact group g."""
    kv = g % dims.n_kv
    dim = g // dims.n_kv
    h = kv * dims.ratio + c
    return h * dims.d_h + dim


def gqa_kv_group_col(g: int, dims: GQADims) -> int:
    """Raw K/V column index for compact group g."""
    kv = g % dims.n_kv
    dim = g // dims.n_kv
    return kv * dims.d_h + dim
