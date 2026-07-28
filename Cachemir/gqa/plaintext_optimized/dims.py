"""Dimension checks and derived GQA parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Optional


ProjectionMethod = Literal["direct", "bsgs"]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class GQAConfig:
    """User-facing configuration.

    qkv_method selects the projection evaluator without changing the encoding.
    bsgs_baby_steps=None asks the simulator to choose the cheapest factor.
    """

    n_he: int
    d: int
    H: int
    n_kv: int
    n_prefill: int = 0
    qkv_method: ProjectionMethod = "direct"
    bsgs_baby_steps: Optional[int] = None


@dataclass(frozen=True)
class GQADims:
    n_he: int
    d: int
    H: int
    n_kv: int
    d_h: int
    ratio: int
    d_kv: int
    t_p: int
    B: int
    m_p: int
    qkv_method: ProjectionMethod
    bsgs_baby_steps: int
    bsgs_giant_steps: int

    @property
    def num_shared_projections(self) -> int:
        return self.ratio + 2


def _choose_bsgs_shape(m: int, projections: int, requested_b: Optional[int]) -> tuple[int, int]:
    if requested_b is not None:
        if requested_b <= 0:
            raise ValueError("bsgs_baby_steps must be positive.")
        b = min(requested_b, m)
        return b, ceil(m / b)

    candidates = range(1, m + 1)
    b = min(
        candidates,
        key=lambda x: (x - 1) + projections * (ceil(m / x) - 1),
    )
    return b, ceil(m / b)


def make_gqa_dims(config: GQAConfig) -> GQADims:
    if not _is_power_of_two(config.n_he):
        raise ValueError(f"n_he ({config.n_he}) must be a positive power of two.")
    if config.d <= 0 or config.n_he % config.d != 0:
        raise ValueError("d must be positive and divide n_he.")
    if config.H <= 0 or config.d % config.H != 0:
        raise ValueError("H must be positive and divide d.")
    if config.n_kv <= 0 or config.H % config.n_kv != 0:
        raise ValueError("n_kv must be positive and divide H.")
    if config.qkv_method not in ("direct", "bsgs"):
        raise ValueError("qkv_method must be 'direct' or 'bsgs'.")

    d_h = config.d // config.H
    ratio = config.H // config.n_kv
    d_kv = config.n_kv * d_h
    if config.n_he % d_kv != 0:
        raise ValueError("n_he must be divisible by d_kv.")

    t_p = config.n_he // d_kv
    if not _is_power_of_two(t_p):
        raise ValueError("t_p=n_he/d_kv must be a power of two.")
    if config.d % t_p != 0:
        raise ValueError("d must be divisible by t_p.")

    m_p = config.d // t_p
    projections = ratio + 2
    b, g = _choose_bsgs_shape(m_p, projections, config.bsgs_baby_steps)

    return GQADims(
        n_he=config.n_he,
        d=config.d,
        H=config.H,
        n_kv=config.n_kv,
        d_h=d_h,
        ratio=ratio,
        d_kv=d_kv,
        t_p=t_p,
        B=config.n_kv * t_p,
        m_p=m_p,
        qkv_method=config.qkv_method,
        bsgs_baby_steps=b,
        bsgs_giant_steps=g,
    )
