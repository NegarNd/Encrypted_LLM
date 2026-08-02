"""Exact and approximate softmax for packed GQA attention using PyTorch."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from .dims import GQADims
from .ops import block_replicate, rotate_left


def softmax_gqa(
    attention: torch.Tensor,
    n_tokens: int,
    dims: GQADims,
) -> torch.Tensor:
    """Exact plaintext softmax oracle; its internal HE cost is not counted."""
    out = torch.zeros_like(attention)
    for c in range(dims.ratio):
        for kv in range(dims.n_kv):
            head_scores = []
            locations = []
            for token in range(n_tokens):
                block = token // dims.t_p
                lane = token % dims.t_p
                slot = kv * dims.t_p + lane
                head_scores.append(attention[c, block, slot])
                locations.append((block, lane))

            head_scores_tensor = torch.stack(head_scores)
            probabilities = torch.softmax(head_scores_tensor, dim=0)

            for probability, (block, lane) in zip(probabilities, locations):
                for dim in range(dims.d_h):
                    slot = dim * dims.B + kv * dims.t_p + lane
                    out[c, block, slot] = probability
    return out


@lru_cache(maxsize=16)
def _make_app_exp_poly(
    fit_lo: float,
    fit_hi: float,
    deg: int = 16,
    n_samples: int = 200_001,
) -> tuple[float, tuple[float, ...]]:
    """Fit once per range; fitting is public preprocessing, not HE work."""
    xs = torch.linspace(fit_lo, fit_hi, n_samples, dtype=torch.float64)
    normalized = (2.0 * xs - (fit_hi + fit_lo)) / (fit_hi - fit_lo)

    # Fit in the numerically stable Chebyshev basis.
    columns = [torch.ones_like(normalized), normalized]
    for _ in range(2, deg + 1):
        columns.append(2.0 * normalized * columns[-1] - columns[-2])
    design = torch.stack(columns, dim=1)
    cheb_coeff = torch.linalg.lstsq(design, torch.exp(xs)).solution

    # Convert T_k(alpha*x + beta) to ordinary power-basis coefficients.
    alpha = 2.0 / (fit_hi - fit_lo)
    beta = -(fit_hi + fit_lo) / (fit_hi - fit_lo)
    t0 = torch.zeros(deg + 1, dtype=torch.float64)
    t0[0] = 1.0
    t1 = torch.zeros_like(t0)
    t1[0], t1[1] = beta, alpha
    basis = [t0, t1]
    for _ in range(2, deg + 1):
        previous = basis[-1]
        times_t = beta * previous
        times_t[1:] = times_t[1:] + alpha * previous[:-1]
        basis.append(2.0 * times_t - basis[-2])
    coeff = sum(cheb_coeff[k] * basis[k] for k in range(deg + 1))

    coeff[0] = 1.0
    exp_input_scale = (1.0 / coeff[deg]) ** (1.0 / deg)
    coeff *= exp_input_scale ** torch.arange(deg + 1, dtype=torch.float64)
    coeff[0] = coeff[deg] = 1.0
    return exp_input_scale.item(), tuple(v.item() for v in coeff)


def _polyval(x: torch.Tensor, coeff: tuple[float, ...]) -> torch.Tensor:
    """Evaluate the degree-16 AppExp polynomial with baby/giant steps."""
    c = x.new_tensor(coeff)
    x2 = x * x
    x4 = x2 * x2
    x8 = x4 * x4
    x16 = x8 * x8

    def baby_poly(cs: torch.Tensor) -> torch.Tensor:
        if len(cs) == 1:
            return torch.zeros_like(x) + cs[0]
        result = cs[1] * x
        if len(cs) >= 3:
            result = result + cs[2] * x2
        if len(cs) == 4:
            result = result + x2 * (cs[3] * x)
        return result + cs[0]

    b0 = baby_poly(c[0:4])
    b1 = baby_poly(c[4:8])
    b2 = baby_poly(c[8:12])
    b3 = baby_poly(c[12:16])
    return b0 + b1 * x4 + (b2 + b3 * x4) * x8 + c[16] * x16


def _app_inv(
    x: torch.Tensor,
    input_max: float = 1.0,
    iters: int = 10,
) -> torch.Tensor:
    if input_max <= 0:
        raise ValueError("input_max must be positive.")
    if input_max != 1.0:
        x = x / float(input_max)

    a = torch.ones_like(x)
    b = x
    en = 1e-4
    for _ in range(iters):
        kn = 2.0 / (en + 1.0)
        kn_sq_int = int(kn * kn)
        kn_int_sqrt = math.sqrt(kn_sq_int)
        factor = 2.0 / kn_int_sqrt
        a = a * (factor - b) * kn_sq_int
        b = b * (factor - b) * kn_sq_int
        en = en * (factor - en) * kn_sq_int

    return a / float(input_max) if input_max != 1.0 else a


def _is_power_of_two(value: int) -> bool:
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


def _fold_token_lanes(value: torch.Tensor, t_p: int) -> torch.Tensor:
    acc = value
    step = 1
    while step < t_p:
        acc = acc + rotate_left(acc, step)
        step *= 2
    return acc


def app_softmax_gqa(
    att_cts: torch.Tensor,
    dims: GQADims,
    n_tokens: int,
    att_lo: float = -20.0,
    att_hi: float = 20.0,
    delta1: int = 2,
    delta2: int = 2,
) -> torch.Tensor:
    """Approximate row softmax independently per query group and KV head."""
    if not isinstance(att_cts, torch.Tensor) or not att_cts.is_floating_point():
        raise TypeError("att_cts must be a floating-point torch.Tensor.")
    if att_cts.ndim != 3:
        raise ValueError("att_cts must have shape (ratio, cache_blocks, n_he).")
    ratio, n_b, n_he = att_cts.shape
    if (ratio, n_he) != (dims.ratio, dims.n_he):
        raise ValueError(
            f"att_cts must have shape ({dims.ratio}, cache_blocks, {dims.n_he})."
        )
    if n_tokens <= 0 or n_tokens > n_b * dims.t_p:
        raise ValueError("n_tokens is incompatible with the packed cache shape.")
    if n_b != math.ceil(n_tokens / dims.t_p):
        raise ValueError("cache_blocks must equal ceil(n_tokens / t_p).")
    if not _is_power_of_two(delta1) or not _is_power_of_two(delta2):
        raise ValueError("delta1 and delta2 must be positive powers of two.")
    if att_hi <= att_lo:
        raise ValueError("att_hi must be greater than att_lo.")

    t_p, B, n_kv = dims.t_p, dims.B, dims.n_kv
    rem = n_tokens - (n_b - 1) * t_p

    base_valid = att_cts.new_zeros(B)
    for kv in range(n_kv):
        base_valid[kv * t_p : kv * t_p + rem] = 1.0
    valid_last = base_valid.repeat(n_he // B)

    base_reduce = att_cts.new_zeros(B)
    base_reduce[::t_p] = 1.0
    reduce_mask = base_reduce.repeat(n_he // B)

    sqrt_dh = math.sqrt(dims.d_h)
    public_max_value = float(att_hi) + (float(att_hi) - float(att_lo)) * 0.05
    fit_lo = (float(att_lo) - public_max_value) / (
        sqrt_dh * float(delta1 * delta2)
    )
    exp_input_scale, exp_coeff = _make_app_exp_poly(fit_lo, 0.0)
    lane_max = att_cts.new_full((n_kv,), public_max_value)
    public_max = lane_max.repeat_interleave(t_p).repeat(n_he // B)

    group_outputs = []
    for c in range(ratio):
        exp_cts = []
        for block in range(n_b):
            diff = (att_cts[c, block] - public_max) / (
                sqrt_dh * float(delta1 * delta2) * exp_input_scale
            )
            if block == n_b - 1:
                diff = diff * valid_last
            e = _polyval(diff, exp_coeff)
            if delta1 != 1:
                e = e**delta1
            if block == n_b - 1:
                e = e * valid_last
            exp_cts.append(e)

        denom_isolated = att_cts.new_zeros(n_he)
        for e in exp_cts:
            denom_isolated = denom_isolated + _fold_token_lanes(e, t_p) * reduce_mask
        inv_denom = _app_inv(
            block_replicate(denom_isolated, t_p),
            input_max=float(n_tokens),
        )
        y_cts = [
            e * inv_denom * (valid_last if b == n_b - 1 else 1.0)
            for b, e in enumerate(exp_cts)
        ]

        for _ in range(int(math.log2(delta2))):
            z_cts = [
                y.square() * (valid_last if b == n_b - 1 else 1.0)
                for b, y in enumerate(y_cts)
            ]
            denom_isolated = att_cts.new_zeros(n_he)
            for z in z_cts:
                denom_isolated = (
                    denom_isolated + _fold_token_lanes(z, t_p) * reduce_mask
                )
            inv_denom = _app_inv(block_replicate(denom_isolated, t_p))
            y_cts = [
                z * inv_denom * (valid_last if b == n_b - 1 else 1.0)
                for b, z in enumerate(z_cts)
            ]
        group_outputs.append(torch.stack(y_cts))

    return torch.stack(group_outputs)
