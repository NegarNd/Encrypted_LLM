"""Persistent-dense GQA attention simulator."""

from .attention import attention_gqa, run_attention_gqa
from .dims import GQAConfig, GQADims, make_gqa_dims

__all__ = [
    "GQAConfig",
    "GQADims",
    "make_gqa_dims",
    "attention_gqa",
    "run_attention_gqa",
]
