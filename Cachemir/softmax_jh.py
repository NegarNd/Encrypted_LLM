"""Approximate softmax for GQA attention."""

from __future__ import annotations

import math
import numpy as np
import torch

from numpy.polynomial import Chebyshev, Polynomial

from .counter import counter
from .dims import GQADims
from .ops import block_replicate


## AppExp/AppInv helper functions for approximate softmax.
def _make_app_exp_poly(
    fit_lo: float,
    fit_hi: float,
    deg: int = 16,
    n_samples: int = 200_001,
):
    xs = np.linspace(fit_lo, fit_hi, n_samples, dtype=np.float64)
    ys = np.exp(xs)

    cheb = Chebyshev.fit(xs, ys, deg=deg, domain=[fit_lo, fit_hi])
    coeff = cheb.convert(kind=Polynomial).coef.astype(np.float64)

    coeff[0] = 1.0

    ## choose exp_input_scale so the highest-degree coefficient becomes 1.
    exp_input_scale = (1.0 / coeff[deg]) ** (1.0 / deg)

    powers = np.arange(coeff.shape[0], dtype=np.float64)
    coeff = coeff * (exp_input_scale ** powers)

    coeff[0] = 1.0
    coeff[deg] = 1.0

    return float(exp_input_scale), tuple(float(v) for v in coeff)


def _polyval(x: torch.Tensor, coeff) -> torch.Tensor:
    ## evaluate degree-16 AppExp polynomial with Stockmeyer-style baby/giant steps.
    c = torch.as_tensor(coeff, dtype=x.dtype, device=x.device)

    x2 = x * x
    x4 = x2 * x2
    x8 = x4 * x4
    x16 = x8 * x8

    def baby_poly(cs):
        if len(cs) == 1:
            return torch.zeros_like(x) + cs[0]

        result = cs[1] * x
        if len(cs) == 2:
            return result + cs[0]

        result = result + cs[2] * x2
        if len(cs) == 3:
            return result + cs[0]

        result = result + x2 * (cs[3] * x)
        return result + cs[0]

    b0 = baby_poly(c[0:4])
    b1 = baby_poly(c[4:8])
    b2 = baby_poly(c[8:12])
    b3 = baby_poly(c[12:16])

    gs0 = b0 + b1 * x4
    gs1 = b2 + b3 * x4

    return gs0 + gs1 * x8 + c[16] * x16


def _app_exp(x: torch.Tensor, coeff) -> torch.Tensor:
    return _polyval(x, coeff)


def _app_inv(
    x: torch.Tensor,
    input_max: float = 1.0,
    iters: int = 10,
) -> torch.Tensor:
    ## the 1/input_max scaling is handled at both the beginning and end
    # of the iteration so AppInv still returns an approximation of 1/x.
    if input_max != 1.0:
        x = x / float(input_max)

    a = torch.ones_like(x)
    b = x
    en = 1e-4

    for _ in range(iters):
        kn = 2.0 / (en + 1.0)
        kn_sq_int = int(kn * kn)
        kn_int_sqrt = math.sqrt(kn_sq_int)
        a = a * (2.0 / kn_int_sqrt - b) * kn_sq_int
        b = b * (2.0 / kn_int_sqrt - b) * kn_sq_int
        en = en * (2.0 / kn_int_sqrt - en) * kn_sq_int

    if input_max != 1.0:
        a = a / float(input_max)

    return a

## end of AppExp/AppInv helper functions.

def app_softmax_gqa(
    att_cts: torch.Tensor,
    dims: GQADims,
    n_tokens: int,
    att_lo: float = -20.0,
    att_hi: float = 20.0,
    delta1: int = 2,
    delta2: int = 2,
) -> torch.Tensor:

    """Row-approx-softmax over the token axis, independently per (query-group c, kv-head)."""
    ratio, n_b, n_he = att_cts.shape
    t_p, B, n_kv = dims.t_p, dims.B, dims.n_kv
    out = torch.zeros_like(att_cts)

    ## validate approximation parameters.
    if delta1 < 1 or delta2 < 1:
        raise ValueError("delta1 and delta2 must be positive integers")
    if 2 ** int(math.log2(delta1)) != delta1:
        raise ValueError("delta1 must be a power of two")
    if 2 ** int(math.log2(delta2)) != delta2:
        raise ValueError("delta2 must be a power of two")

    # valid tokens in the last ciphertext
    rem = n_tokens - (n_b - 1) * t_p

    # valid mask for the last ciphertext for a single block - needs to be replicated in the ciphertext (done by tile)
    base_valid = torch.zeros(B, dtype=torch.float64, device=att_cts.device)
    for kv in range(n_kv):
        base_valid[kv * t_p: kv * t_p + rem] = 1.0
    valid_last = base_valid.tile(n_he // B)

    # mask keeping exactly one base slot per lane (for isolating summation fold results)
    base_reduce = torch.zeros(B, dtype=torch.float64, device=att_cts.device)
    base_reduce[::t_p] = 1.0
    reduce_mask = base_reduce.tile(n_he // B)

    ## constants for public max, scaling, and AppExp fitting.
    sqrt_dh = float(math.sqrt(dims.d_h))
    buffer_ratio = 0.05
    att_width = float(att_hi) - float(att_lo)
    public_max_value = float(att_hi) + att_width * buffer_ratio

    ## AppExp fitting range for (QK^T - public_max) / (sqrt(d_h) * delta1 * delta2).
    fit_lo = (float(att_lo) - public_max_value) / (
        sqrt_dh * float(delta1 * delta2)
    )
    fit_hi = 0.0
    exp_input_scale, exp_coeff = _make_app_exp_poly(fit_lo, fit_hi)
    
    ## use calibrated public max instead of ciphertext-dependent lane-wise max.
    lane_max = torch.full(
        (n_kv,),
        public_max_value,
        dtype=torch.float64,
        device=att_cts.device,
    )

    # broadcast the calibrated public max to each lane's t_p slots, tiled over d_h blocks
    public_max = lane_max.repeat_interleave(t_p).tile(n_he // B)
    
    for c in range(ratio):
        # mask the padding slots of the last block to 0 BEFORE exp: raw
        # attention scores are unscaled dot products and can be large in
        # magnitude, so exp(padding_value - public_max) can overflow to inf if
        # left unmasked; forcing the diff to exactly 0 keeps exp bounded
        # (exp(0) == 1). Masking again AFTER exp zeroes those 1s back out so
        # they don't pollute the denominator sum below.
        exp_cts = []
        for b in range(n_b):
            diff = att_cts[c, b] - public_max
            ## scale raw diff for AppExp using sqrt(d_h), delta1, delta2, and x/k.
            diff = diff / (
                sqrt_dh * float(delta1 * delta2) * exp_input_scale
            )
            if b == n_b - 1:
                diff = diff * valid_last
            e = _app_exp(diff, exp_coeff)
            if delta1 != 1:
                e = e ** delta1
            if b == n_b - 1:
                e = e * valid_last
            exp_cts.append(e)

        # fold: all lanes fold correctly in parallel (verified: lane boundaries
        # align exactly with the rotation steps, so no cross-lane leakage)
        denom_isolated = torch.zeros(n_he, dtype=torch.float64)
        for e in exp_cts:
            acc = e.clone()
            step = 1
            while step < t_p:
                counter.rotations += 1
                acc = acc + torch.roll(acc, -step)
                step *= 2
            denom_isolated += acc * reduce_mask

        denom = block_replicate(denom_isolated, t_p)
        ## replace exact reciprocal with AppInv; denom is normalized by n_tokens inside AppInv.
        inv_denom = _app_inv(denom, input_max=float(n_tokens))

        ## keep first normalized result for delta2 refinement.
        y_cts = []
        for b in range(n_b):
            y = exp_cts[b] * inv_denom
            if b == n_b - 1:
                y = y * valid_last
            y_cts.append(y)

        ## apply square-and-normalize log2(delta2) times.
        num_refine = int(math.log2(delta2))
        for _ in range(num_refine):
            z_cts = []
            for b in range(n_b):
                z = y_cts[b] * y_cts[b]
                if b == n_b - 1:
                    z = z * valid_last
                z_cts.append(z)

            denom_isolated = torch.zeros(n_he, dtype=torch.float64)
            for z in z_cts:
                acc = z.clone()
                step = 1
                while step < t_p:
                    counter.rotations += 1
                    acc = acc + torch.roll(acc, -step)
                    step *= 2
                denom_isolated += acc * reduce_mask

            denom = block_replicate(denom_isolated, t_p)
            ## after normalization, sum(y^2) is already bounded by 1.
            inv_denom = _app_inv(denom, input_max=1.0)

            next_y_cts = []
            for b in range(n_b):
                y = z_cts[b] * inv_denom
                if b == n_b - 1:
                    y = y * valid_last
                next_y_cts.append(y)

            y_cts = next_y_cts

        for b in range(n_b):
            out[c, b] = y_cts[b]

    return out
