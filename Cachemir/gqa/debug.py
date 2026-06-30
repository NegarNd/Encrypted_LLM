"""Small debug helpers for dimensions and operation accounting."""

from __future__ import annotations

from dataclasses import dataclass

from .dims import GQADims


@dataclass
class OpCounter:
    rotations: int = 0
    ct_pt_mults: int = 0
    ct_ct_mults: int = 0

    def reset(self) -> None:
        self.rotations = 0
        self.ct_pt_mults = 0
        self.ct_ct_mults = 0


def describe_dims(dims: GQADims) -> str:
    """Human-readable compact GQA dimension summary."""
    return (
        f"N={dims.N}, d={dims.d}, H={dims.H}, n_kv={dims.n_kv}, "
        f"d_h={dims.d_h}, ratio={dims.ratio}, d_kv={dims.d_kv}, "
        f"t_p={dims.t_p}, B={dims.B}, R={dims.R}, "
        f"input_chunks={dims.input_chunks}"
    )
