"""Compact GQA attention simulation with Cachemir-style slot layouts."""

from .dims import GQAConfig, GQADims, make_gqa_dims
from .attention import attention_gqa, run_attention_gqa

__all__ = [
    "GQAConfig",
    "GQADims",
    "make_gqa_dims",
    "attention_gqa",
    "run_attention_gqa",
]
