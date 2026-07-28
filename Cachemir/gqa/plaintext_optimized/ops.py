"""HE-like primitive operations implemented with NumPy."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .counter import counter
from .dims import GQADims


def rotate_left(value: np.ndarray, amount: int) -> np.ndarray:
    """Simulate one ciphertext left rotation and record its key amount."""
    counter.record_rotation(amount, value.size)
    return np.roll(value, -amount)


def multiply_plain(value: np.ndarray, plaintext: np.ndarray) -> np.ndarray:
    counter.ct_pt_mult += 1
    return value * plaintext


def multiply_cipher(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    counter.ct_ct_mult += 1
    return left * right


def reduce_lanes(value: np.ndarray, dims: GQADims, pos: int = 0) -> np.ndarray:
    """Sum each t_p-slot partial-dot-product block into lane ``pos``."""
    out = value
    step = 1
    bit = 0
    while step < dims.t_p:
        direction = -step if (pos >> bit) & 1 else step
        out = out + rotate_left(out, direction)
        step *= 2
        bit += 1

    mask = np.zeros(dims.n_he, dtype=np.float64)
    mask[pos :: dims.t_p] = 1.0
    return multiply_plain(out, mask)


def replicate_lanes(value: np.ndarray, dims: GQADims) -> np.ndarray:
    """Replicate lane zero to all t_p token lanes."""
    out = value
    step = 1
    while step < dims.t_p:
        out = out + rotate_left(out, -step)
        step *= 2
    return out


def prepare_shared_projection_inputs(
    x_ct: np.ndarray,
    dims: GQADims,
) -> List[np.ndarray]:
    """Create the input rotations shared by all Q/K/V projections."""
    if dims.qkv_method == "direct":
        inputs = [x_ct]
        for r in range(1, dims.m_p):
            inputs.append(rotate_left(x_ct, r * dims.t_p))
        return inputs

    inputs = [x_ct]
    for baby in range(1, dims.bsgs_baby_steps):
        inputs.append(rotate_left(x_ct, baby * dims.t_p))
    return inputs


def evaluate_direct_projection(
    x_rotations: Sequence[np.ndarray],
    encoded_diagonals: Sequence[np.ndarray],
) -> np.ndarray:
    """Evaluate one projection from a previously shared set of X rotations."""
    acc = np.zeros_like(x_rotations[0])
    for x_rot, diagonal in zip(x_rotations, encoded_diagonals):
        acc += multiply_plain(x_rot, diagonal)
    return acc


def evaluate_bsgs_projection(
    babies: Sequence[np.ndarray],
    encoded_diagonals: Sequence[np.ndarray],
    dims: GQADims,
) -> np.ndarray:
    """Evaluate one BSGS projection from shared baby-step rotations."""
    b = dims.bsgs_baby_steps
    g = dims.bsgs_giant_steps
    out = np.zeros_like(babies[0])

    for giant in range(g):
        giant_amount = giant * b * dims.t_p
        group = np.zeros_like(babies[0])

        for baby in range(b):
            diagonal_index = giant * b + baby
            if diagonal_index >= dims.m_p:
                break

            diagonal = encoded_diagonals[diagonal_index]
            # RotL(baby * RotR(P, giant), giant)
            # equals RotL(X, diagonal_index*t_p) * P.
            adjusted = np.roll(diagonal, giant_amount)
            group += multiply_plain(babies[baby], adjusted)

        if giant:
            group = rotate_left(group, giant_amount)
        out += group

    return out


def evaluate_projection(
    shared_inputs: Sequence[np.ndarray],
    encoded_diagonals: Sequence[np.ndarray],
    dims: GQADims,
) -> np.ndarray:
    """Evaluate one Q, K, or V projection using prepared shared inputs."""
    if dims.qkv_method == "direct":
        return evaluate_direct_projection(shared_inputs, encoded_diagonals)
    return evaluate_bsgs_projection(shared_inputs, encoded_diagonals, dims)
