"""Client-side Orion encoding for the persistent-dense GQA layout."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import orion
import torch

from ..plaintext_optimized.dims import GQADims
from ..plaintext_optimized.encoding import (
    encode_input,
    make_weights_kv,
    make_weights_q_gqa,
)


def encode_plain_vector(values: np.ndarray, *, n_he: int, level: int) -> Any:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != n_he:
        raise ValueError(
            f"Expected exactly {n_he} encoded slots; got {vector.size}."
        )
    return orion.encode(torch.as_tensor(vector, dtype=torch.float64), level)


def encrypt_input(x: np.ndarray, dims: GQADims, *, level: int) -> Any:
    """Encode persistent ``[x | x | ...]`` and encrypt one ciphertext."""
    slots = encode_input(x, dims)
    return orion.encrypt(
        encode_plain_vector(slots, n_he=dims.n_he, level=level)
    )


def decrypt_decode_ct(ct: Any) -> np.ndarray:
    """Client-only helper used by verification, never by the HE pipeline."""
    decoded = ct.decrypt().decode()
    if hasattr(decoded, "real"):
        decoded = decoded.real
    return np.asarray(decoded, dtype=np.float64)[:]


def encrypt_slot_vector(
    values: np.ndarray,
    dims: GQADims,
    *,
    level: int,
) -> Any:
    """Encrypt an already packed vector (used for plaintext cache prefill)."""
    return orion.encrypt(
        encode_plain_vector(values, n_he=dims.n_he, level=level)
    )


def _encode_direct(
    diagonals: Sequence[np.ndarray],
    dims: GQADims,
    level: int,
) -> list[Any]:
    return [
        encode_plain_vector(diagonal, n_he=dims.n_he, level=level)
        for diagonal in diagonals
    ]


def _encode_bsgs(
    diagonals: Sequence[np.ndarray],
    dims: GQADims,
    level: int,
) -> list[list[Any]]:
    """Pre-adjust plaintext diagonals for the BSGS giant rotations."""
    b = dims.bsgs_baby_steps
    groups: list[list[Any]] = []
    for giant in range(dims.bsgs_giant_steps):
        amount = giant * b * dims.t_p
        group: list[Any] = []
        for baby in range(b):
            index = giant * b + baby
            if index >= dims.m_p:
                break
            adjusted = np.roll(diagonals[index], amount)
            group.append(
                encode_plain_vector(adjusted, n_he=dims.n_he, level=level)
            )
        groups.append(group)
    return groups


def encode_projection_diagonals(
    diagonals: Sequence[np.ndarray],
    dims: GQADims,
    *,
    level: int,
) -> list[Any]:
    if len(diagonals) != dims.m_p:
        raise ValueError(
            f"Expected {dims.m_p} diagonals; got {len(diagonals)}."
        )
    if dims.qkv_method == "direct":
        return _encode_direct(diagonals, dims, level)
    return _encode_bsgs(diagonals, dims, level)


def make_he_weights_q_gqa(
    dims: GQADims,
    *,
    seed: int = 1,
    level: int,
) -> tuple[np.ndarray, list[list[Any]], list[list[np.ndarray]]]:
    Wq, raw = make_weights_q_gqa(dims, seed)
    encoded = [
        encode_projection_diagonals(group, dims, level=level)
        for group in raw
    ]
    return Wq, encoded, raw


def make_he_weights_kv(
    dims: GQADims,
    *,
    seed: int,
    level: int,
) -> tuple[np.ndarray, list[Any], list[np.ndarray]]:
    W, raw = make_weights_kv(dims, seed)
    return W, encode_projection_diagonals(raw, dims, level=level), raw
