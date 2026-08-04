"""Approximate GQA softmax evaluated entirely on Orion ciphertexts.

The polynomial coefficients, masks, public maximum, and scalar constants are
plaintext.  Attention scores, exponentials, denominators, reciprocals, and
probabilities remain encrypted throughout this module.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

from ..plaintext_optimized.dims import GQADims
from .he_encoding_orion import encode_plain_vector
from .he_ops_orion import (
    multiply_cipher,
    multiply_plain,
    rotate_left,
)


def _make_app_exp_poly(
    fit_lo: float,
    fit_hi: float,
    deg: int = 16,
    n_samples: int = 200_001,
) -> tuple[float, tuple[float, ...]]:
    """Fit the public degree-16 AppExp polynomial before HE evaluation."""
    xs = np.linspace(fit_lo, fit_hi, n_samples, dtype=np.float64)
    ys = np.exp(xs)

    cheb = Chebyshev.fit(xs, ys, deg=deg, domain=[fit_lo, fit_hi])
    coeff = cheb.convert(kind=Polynomial).coef.astype(np.float64)
    coeff[0] = 1.0

    exp_input_scale = (1.0 / coeff[deg]) ** (1.0 / deg)
    powers = np.arange(coeff.shape[0], dtype=np.float64)
    coeff = coeff * (exp_input_scale ** powers)
    coeff[0] = 1.0
    coeff[deg] = 1.0

    return float(exp_input_scale), tuple(float(value) for value in coeff)


def _encode_vector(values: np.ndarray, ct: Any, dims: GQADims) -> Any:
    return encode_plain_vector(
        values,
        n_he=dims.n_he,
        level=ct.level(),
    )


def _constant_vector(value: float, dims: GQADims) -> np.ndarray:
    return np.full(dims.n_he, float(value), dtype=np.float64)


def _add_plain_vector(ct: Any, values: np.ndarray, dims: GQADims) -> Any:
    return ct + _encode_vector(values, ct, dims)


def _sub_plain_vector(ct: Any, values: np.ndarray, dims: GQADims) -> Any:
    return ct - _encode_vector(values, ct, dims)


def _add_plain_constant(ct: Any, value: float, dims: GQADims) -> Any:
    return _add_plain_vector(ct, _constant_vector(value, dims), dims)


def _sub_plain_constant(ct: Any, value: float, dims: GQADims) -> Any:
    return _sub_plain_vector(ct, _constant_vector(value, dims), dims)


def _mul_plain_vector(ct: Any, values: np.ndarray, dims: GQADims) -> Any:
    return multiply_plain(ct, _encode_vector(values, ct, dims))


def _mul_plain_constant(ct: Any, value: float, dims: GQADims) -> Any:
    return _mul_plain_vector(ct, _constant_vector(value, dims), dims)


def _align_levels(left: Any, right: Any, dims: GQADims) -> tuple[Any, Any]:
    """Bring two ciphertexts to the same level before binary arithmetic."""
    target = min(left.level(), right.level())
    while left.level() > target:
        left = _mul_plain_constant(left, 1.0, dims)
    while right.level() > target:
        right = _mul_plain_constant(right, 1.0, dims)
    return left, right


def _add_cipher(left: Any, right: Any, dims: GQADims) -> Any:
    left, right = _align_levels(left, right, dims)
    return left + right


def _mul_cipher(left: Any, right: Any, dims: GQADims) -> Any:
    left, right = _align_levels(left, right, dims)
    return multiply_cipher(left, right)


def _polyval_he(
    x: Any,
    coeff: Sequence[float],
    dims: GQADims,
) -> Any:
    """Evaluate the degree-16 AppExp polynomial with baby/giant steps."""
    if len(coeff) != 17:
        raise ValueError(f"Expected 17 polynomial coefficients; got {len(coeff)}.")

    x2 = _mul_cipher(x, x, dims)
    # added
    x3 = _mul_cipher(x2, x, dims)  
    x4 = _mul_cipher(x2, x2, dims)
    x8 = _mul_cipher(x4, x4, dims)
    x16 = _mul_cipher(x8, x8, dims)

    def baby_poly(cs: Sequence[float]) -> Any:
        result = _mul_plain_constant(x, cs[1], dims)
        result = _add_cipher(result, _mul_plain_constant(x2, cs[2], dims), dims)
        # cubic = _mul_cipher(
        #     x2,
        #     _mul_plain_constant(x, cs[3], dims),
        #     dims,
        # )
        result = _add_cipher(result, _mul_plain_constant(x3, cs[3], dims), dims)
        # result = _add_cipher(result, cubic, dims)
        return _add_plain_constant(result, cs[0], dims)

    b0 = baby_poly(coeff[0:4])
    b1 = baby_poly(coeff[4:8])
    b2 = baby_poly(coeff[8:12])
    b3 = baby_poly(coeff[12:16])

    gs0 = _add_cipher(b0, _mul_cipher(b1, x4, dims), dims)
    gs1 = _add_cipher(b2, _mul_cipher(b3, x4, dims), dims)
    result = _add_cipher(gs0, _mul_cipher(gs1, x8, dims), dims)
    if coeff[16] == 1.0:
        return _add_cipher(result, x16, dims)
    return _add_cipher(
        result,
        _mul_plain_constant(x16, coeff[16], dims),
        dims,
    )


def _power_of_two_he(x: Any, exponent: int, dims: GQADims) -> Any:
    if exponent < 1 or exponent & (exponent - 1):
        raise ValueError("The ciphertext exponent must be a positive power of two.")
    result = x
    for _ in range(int(math.log2(exponent))):
        result = _mul_cipher(result, result, dims)
    return result


def _app_inv_he(
    x: Any,
    dims: GQADims,
    *,
    input_max: float = 1.0,
    iters: int = 10,
) -> Any:
    """Approximate 1/x using the ciphertext AppInv iteration."""
    if input_max <= 0.0:
        raise ValueError("input_max must be positive.")
    if iters < 1:
        raise ValueError("iters must be positive.")

    b = x
    if input_max != 1.0:
        b = _mul_plain_constant(b, 1.0 / float(input_max), dims)

    # Create encrypted one at b's level without decrypting or separately
    # encrypting a new ciphertext.
    a = _add_plain_constant(b - b, 1.0, dims)
    en = 1e-4

    for _ in range(iters):
        kn = 2.0 / (en + 1.0)
        kn_sq_int = int(kn * kn)
        kn_int_sqrt = math.sqrt(kn_sq_int)
        factor = 2.0 / kn_int_sqrt

        # u = b - factor, so multiplying by -kn_sq_int implements
        # (factor - b) * kn_sq_int without ciphertext negation.
        u = _sub_plain_constant(b, factor, dims)
        a = _mul_cipher(a, u, dims)
        a = _mul_plain_constant(a, -float(kn_sq_int), dims)

        b = _mul_cipher(b, u, dims)
        b = _mul_plain_constant(b, -float(kn_sq_int), dims)
        en = en * (factor - en) * kn_sq_int

    if input_max != 1.0:
        a = _mul_plain_constant(a, 1.0 / float(input_max), dims)
    return a


def _fold_token_lanes(x: Any, dims: GQADims) -> Any:
    """Sum the t_p token lanes independently inside every packed lane."""
    acc = x
    step = 1
    while step < dims.t_p:
        acc = _add_cipher(acc, rotate_left(acc, step, dims), dims)
        step *= 2
    return acc


def _replicate_token_lane_zero(x: Any, dims: GQADims) -> Any:
    """Replicate lane zero across each t_p-slot token lane."""
    acc = x
    step = 1
    while step < dims.t_p:
        acc = _add_cipher(acc, rotate_left(acc, -step, dims), dims)
        step *= 2
    return acc


def _block_valid_masks(
    n_blocks: int,
    n_tokens: int,
    dims: GQADims,
) -> list[np.ndarray]:
    expected_blocks = math.ceil(n_tokens / dims.t_p)
    if n_blocks != expected_blocks:
        raise ValueError(
            f"Expected {expected_blocks} attention blocks for {n_tokens} tokens; "
            f"got {n_blocks}."
        )

    masks = [np.ones(dims.n_he, dtype=np.float64) for _ in range(n_blocks)]
    remainder = n_tokens - (n_blocks - 1) * dims.t_p
    if remainder < dims.t_p:
        base = np.zeros(dims.B, dtype=np.float64)
        for kv in range(dims.n_kv):
            start = kv * dims.t_p
            base[start : start + remainder] = 1.0
        masks[-1] = np.tile(base, dims.n_he // dims.B)
    return masks


def app_softmax_gqa_he(
    att_cts: list[list[Any]],
    n_tokens: int,
    dims: GQADims,
    att_lo: float = -20.0,
    att_hi: float = 20.0,
    delta1: int = 2,
    delta2: int = 2,
    app_inv_iters: int = 10,
) -> list[list[Any]]:
    """Approximate row softmax for packed Orion GQA attention ciphertexts."""
    if len(att_cts) != dims.ratio:
        raise ValueError(
            f"Expected {dims.ratio} query-group rows; got {len(att_cts)}."
        )
    if not att_cts or not att_cts[0]:
        raise ValueError("Attention ciphertexts cannot be empty.")
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive.")
    if delta1 < 1 or delta1 & (delta1 - 1):
        raise ValueError("delta1 must be a positive power of two.")
    if delta2 < 1 or delta2 & (delta2 - 1):
        raise ValueError("delta2 must be a positive power of two.")

    n_blocks = len(att_cts[0])
    if any(len(row) != n_blocks for row in att_cts):
        raise ValueError("All attention rows must contain the same number of blocks.")

    valid_masks = _block_valid_masks(n_blocks, n_tokens, dims)
    reduce_mask = np.zeros(dims.n_he, dtype=np.float64)
    reduce_mask[:: dims.t_p] = 1.0

    sqrt_dh = math.sqrt(dims.d_h)
    buffer_ratio = 0.05
    att_width = float(att_hi) - float(att_lo)
    public_max_value = float(att_hi) + att_width * buffer_ratio
    public_max = _constant_vector(public_max_value, dims)

    fit_lo = (float(att_lo) - public_max_value) / (
        sqrt_dh * float(delta1 * delta2)
    )
    exp_input_scale, exp_coeff = _make_app_exp_poly(fit_lo, 0.0)
    input_divisor = sqrt_dh * float(delta1 * delta2) * exp_input_scale

    output: list[list[Any]] = []
    for row in att_cts:
        exp_cts: list[Any] = []
        for block, att_ct in enumerate(row):
            diff = _sub_plain_vector(att_ct, public_max, dims)
            diff = _mul_plain_constant(diff, 1.0 / input_divisor, dims)

            # Mask every block so the full and partial blocks follow the same
            # level schedule. Full blocks simply multiply by an all-one mask.
            diff = _mul_plain_vector(diff, valid_masks[block], dims)
            exp_ct = _polyval_he(diff, exp_coeff, dims)
            exp_ct = _power_of_two_he(exp_ct, delta1, dims)
            exp_ct = _mul_plain_vector(exp_ct, valid_masks[block], dims)
            exp_cts.append(exp_ct)

        denom_isolated = None
        for exp_ct in exp_cts:
            folded = _fold_token_lanes(exp_ct, dims)
            folded = _mul_plain_vector(folded, reduce_mask, dims)
            denom_isolated = (
                folded
                if denom_isolated is None
                else _add_cipher(denom_isolated, folded, dims)
            )

        denom = _replicate_token_lane_zero(denom_isolated, dims)
        inv_denom = _app_inv_he(
            denom,
            dims,
            input_max=float(n_tokens),
            iters=app_inv_iters,
        )

        y_cts: list[Any] = []
        for block, exp_ct in enumerate(exp_cts):
            y = _mul_cipher(exp_ct, inv_denom, dims)
            y = _mul_plain_vector(y, valid_masks[block], dims)
            y_cts.append(y)

        for _ in range(int(math.log2(delta2))):
            z_cts: list[Any] = []
            for block, y in enumerate(y_cts):
                z = _mul_cipher(y, y, dims)
                z = _mul_plain_vector(z, valid_masks[block], dims)
                z_cts.append(z)

            denom_isolated = None
            for z in z_cts:
                folded = _fold_token_lanes(z, dims)
                folded = _mul_plain_vector(folded, reduce_mask, dims)
                denom_isolated = (
                    folded
                    if denom_isolated is None
                    else _add_cipher(denom_isolated, folded, dims)
                )

            denom = _replicate_token_lane_zero(denom_isolated, dims)
            inv_denom = _app_inv_he(
                denom,
                dims,
                input_max=1.0,
                iters=app_inv_iters,
            )

            next_y_cts: list[Any] = []
            for block, z in enumerate(z_cts):
                y = _mul_cipher(z, inv_denom, dims)
                y = _mul_plain_vector(y, valid_masks[block], dims)
                next_y_cts.append(y)
            y_cts = next_y_cts

        output.append(y_cts)

    return output