"""Orion primitives mirroring ``gqa.plaintext.ops``.

No function in this module decrypts a ciphertext or indexes its slots.
Orion's ``ct.roll(k)`` is used as the ciphertext equivalent of the plaintext
simulator's ``np.roll(x, -k)`` (a left rotation by ``k`` slots).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import orion
import torch

from ..plaintext_optimized.counter import counter
from ..plaintext_optimized.dims import GQADims


def _encode(values: np.ndarray, level: int) -> Any:
    return orion.encode(torch.as_tensor(values, dtype=torch.float64), level)


def rotate_left(value: Any, amount: int, dims: GQADims) -> Any:
    """Rotate a ciphertext left and record the required Galois key."""
    counter.record_rotation(amount, dims.n_he)
    if amount % dims.n_he == 0:
        return value
    return value.roll(amount)


def multiply_plain(value: Any, plaintext: Any) -> Any:
    counter.ct_pt_mult += 1
    return value * plaintext


def multiply_cipher(left: Any, right: Any) -> Any:
    counter.ct_ct_mult += 1
    return left * right

def multiply_cipher_raw(left: Any, right: Any) -> Any:
    """Ciphertext multiplication without relinearization or rescaling."""
    counter.ct_ct_mult += 1
    return left.mul_raw(right)

def reduce_lanes(value: Any, dims: GQADims, pos: int = 0) -> Any:
    """Sum each t_p-slot block into lane ``pos``, then mask other lanes."""
    if not 0 <= pos < dims.t_p:
        raise ValueError(f"pos must be in [0, {dims.t_p}); got {pos}.")

    out = value
    step = 1
    bit = 0
    while step < dims.t_p:
        direction = -step if (pos >> bit) & 1 else step
        out = out + rotate_left(out, direction, dims)
        step *= 2
        bit += 1

    mask = np.zeros(dims.n_he, dtype=np.float64)
    mask[pos :: dims.t_p] = 1.0
    return multiply_plain(out, _encode(mask, out.level()))


def replicate_lanes(value: Any, dims: GQADims) -> Any:
    """Replicate lane zero into every lane of each t_p-slot block."""
    out = value
    step = 1
    while step < dims.t_p:
        out = out + rotate_left(out, -step, dims)
        step *= 2
    return out

def prepare_shared_projection_inputs(x_ct: Any, dims: GQADims) -> list[Any]:
    if dims.qkv_method == "direct":
        amounts = [
            r * dims.t_p
            for r in range(1, dims.m_p)
        ]
    else:
        amounts = [
            baby * dims.t_p
            for baby in range(1, dims.bsgs_baby_steps)
        ]

    # No rotations are needed; return only the original ciphertext.
    if not amounts:
        return [x_ct]
    
    for amount in amounts:
        counter.record_rotation(amount, dims.n_he)

    # All rotations use the same x_ct, so decomposition can be hoisted.
    rotated = x_ct.roll_many(amounts)

    return [x_ct, *rotated]


def _sum_products(inputs: Sequence[Any], diagonals: Sequence[Any]) -> Any:
    if not inputs or not diagonals:
        raise ValueError("Projection inputs and diagonals cannot be empty.")
    if len(inputs) != len(diagonals):
        raise ValueError(
            f"Inputs ({len(inputs)}) and diagonals ({len(diagonals)}) differ."
        )

    out = multiply_plain(inputs[0], diagonals[0])
    for x_rot, diagonal in zip(inputs[1:], diagonals[1:]):
        out = out + multiply_plain(x_rot, diagonal)
    return out


def evaluate_direct_projection(
    x_rotations: Sequence[Any],
    encoded_diagonals: Sequence[Any],
    dims: GQADims,
) -> Any:
    if len(x_rotations) != dims.m_p:
        raise ValueError(
            f"Expected {dims.m_p} shared rotations; got {len(x_rotations)}."
        )
    return _sum_products(x_rotations, encoded_diagonals)


def evaluate_bsgs_projection(
    babies: Sequence[Any],
    encoded_diagonal_groups: Sequence[Sequence[Any]],
    dims: GQADims,
) -> Any:
    """Evaluate BSGS using client-pre-adjusted plaintext diagonals.

    ``he_encoding_orion`` right-rotates each diagonal by its giant-step
    amount before encoding. Therefore only ciphertext rotations occur here.
    """
    if len(babies) != dims.bsgs_baby_steps:
        raise ValueError(
            f"Expected {dims.bsgs_baby_steps} baby rotations; got {len(babies)}."
        )

    out = None
    for giant, group_diagonals in enumerate(encoded_diagonal_groups):
        group = _sum_products(babies[: len(group_diagonals)], group_diagonals)
        if giant:
            group = rotate_left(
                group,
                giant * dims.bsgs_baby_steps * dims.t_p,
                dims,
            )
        out = group if out is None else out + group

    if out is None:
        raise ValueError("Encoded BSGS diagonal groups cannot be empty.")
    return out


def evaluate_projection(
    shared_inputs: Sequence[Any],
    encoded_diagonals: Sequence[Any],
    dims: GQADims,
) -> Any:
    if dims.qkv_method == "direct":
        return evaluate_direct_projection(shared_inputs, encoded_diagonals, dims)
    return evaluate_bsgs_projection(shared_inputs, encoded_diagonals, dims)
