"""HE-like primitive operations implemented with PyTorch."""

from __future__ import annotations

from typing import Sequence

import torch

from .counter import counter
from .dims import GQADims


def rotate_left(value: torch.Tensor, amount: int) -> torch.Tensor:
    counter.record_rotation(amount, value.numel())
    return torch.roll(value, shifts=-amount, dims=-1)


def multiply_plain(value: torch.Tensor, plaintext: torch.Tensor) -> torch.Tensor:
    counter.ct_pt_mult += 1
    return value * plaintext


def multiply_cipher(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    counter.ct_ct_mult += 1
    return left * right


def reduce_lanes(value: torch.Tensor, dims: GQADims, pos: int = 0) -> torch.Tensor:
    out = value
    step = 1
    bit = 0
    while step < dims.t_p:
        direction = -step if (pos >> bit) & 1 else step
        out = out + rotate_left(out, direction)
        step *= 2
        bit += 1

    mask = value.new_zeros(dims.n_he)
    mask[pos :: dims.t_p] = 1.0
    return multiply_plain(out, mask)


def block_replicate(value: torch.Tensor, block_size: int) -> torch.Tensor:
    """Replicate the first slot of every block across that block."""
    out = value
    step = 1
    while step < block_size:
        out = out + rotate_left(out, -step)
        step *= 2
    return out


def replicate_lanes(value: torch.Tensor, dims: GQADims) -> torch.Tensor:
    return block_replicate(value, dims.t_p)


def prepare_shared_projection_inputs(
    x_ct: torch.Tensor,
    dims: GQADims,
) -> list[torch.Tensor]:
    if dims.qkv_method == "direct":
        amounts = [r * dims.t_p for r in range(1, dims.m_p)]
    else:
        amounts = [
            baby * dims.t_p for baby in range(1, dims.bsgs_baby_steps)
        ]
    return [x_ct, *(rotate_left(x_ct, amount) for amount in amounts)]


def evaluate_direct_projection(
    x_rotations: Sequence[torch.Tensor],
    encoded_diagonals: Sequence[torch.Tensor],
) -> torch.Tensor:
    acc = torch.zeros_like(x_rotations[0])
    for x_rot, diagonal in zip(x_rotations, encoded_diagonals):
        acc = acc + multiply_plain(x_rot, diagonal)
    return acc


def evaluate_bsgs_projection(
    babies: Sequence[torch.Tensor],
    encoded_diagonals: Sequence[torch.Tensor],
    dims: GQADims,
) -> torch.Tensor:
    b = dims.bsgs_baby_steps
    out = torch.zeros_like(babies[0])
    for giant in range(dims.bsgs_giant_steps):
        giant_amount = giant * b * dims.t_p
        group = torch.zeros_like(babies[0])
        for baby in range(b):
            diagonal_index = giant * b + baby
            if diagonal_index >= dims.m_p:
                break
            diagonal = encoded_diagonals[diagonal_index]
            adjusted = torch.roll(diagonal, shifts=giant_amount, dims=-1)
            group = group + multiply_plain(babies[baby], adjusted)
        if giant:
            group = rotate_left(group, giant_amount)
        out = out + group
    return out


def evaluate_projection(
    shared_inputs: Sequence[torch.Tensor],
    encoded_diagonals: Sequence[torch.Tensor],
    dims: GQADims,
) -> torch.Tensor:
    if dims.qkv_method == "direct":
        return evaluate_direct_projection(shared_inputs, encoded_diagonals)
    return evaluate_bsgs_projection(shared_inputs, encoded_diagonals, dims)
