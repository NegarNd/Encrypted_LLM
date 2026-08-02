"""Dense PyTorch reference checks for persistent-layout GQA."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .dims import GQADims


def decode_q(q_cts: torch.Tensor, dims: GQADims) -> torch.Tensor:
    q = q_cts.new_zeros(dims.d)
    for c in range(dims.ratio):
        for dim in range(dims.d_h):
            for kv in range(dims.n_kv):
                compact = dim * dims.n_kv + kv
                head = kv * dims.ratio + c
                q[head * dims.d_h + dim] = q_cts[c, compact * dims.t_p]
    return q


def decode_positioned_kv(
    value: torch.Tensor, pos: int, dims: GQADims
) -> torch.Tensor:
    out = value.new_zeros(dims.d_kv)
    for dim in range(dims.d_h):
        for kv in range(dims.n_kv):
            compact = dim * dims.n_kv + kv
            out[kv * dims.d_h + dim] = value[compact * dims.t_p + pos]
    return out


def decode_output(o_cts: torch.Tensor, dims: GQADims) -> torch.Tensor:
    out = o_cts.new_zeros(dims.d)
    for c in range(dims.ratio):
        for dim in range(dims.d_h):
            for kv in range(dims.n_kv):
                compact = dim * dims.n_kv + kv
                head = kv * dims.ratio + c
                out[head * dims.d_h + dim] = o_cts[c, compact * dims.t_p]
    return out


def decode_qkt(
    attention: torch.Tensor,
    n_tokens: int,
    dims: GQADims,
) -> torch.Tensor:
    scores = attention.new_zeros((dims.H, n_tokens))
    for c in range(dims.ratio):
        for kv in range(dims.n_kv):
            head = kv * dims.ratio + c
            for token in range(n_tokens):
                block, lane = divmod(token, dims.t_p)
                scores[head, token] = attention[c, block, kv * dims.t_p + lane]
    return scores


def reference_attention(
    tokens: Sequence[torch.Tensor],
    x_new: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    dims: GQADims,
    *,
    skip_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_tokens = [*tokens, x_new]
    q = x_new @ Wq
    keys = [x @ Wk for x in all_tokens]
    values = [x @ Wv for x in all_tokens]
    scores = x_new.new_zeros((dims.H, len(all_tokens)))
    output = x_new.new_zeros(dims.d)

    for h in range(dims.H):
        kv = h // dims.ratio
        qh = q[h * dims.d_h : (h + 1) * dims.d_h]
        kh = torch.stack([k[kv * dims.d_h : (kv + 1) * dims.d_h] for k in keys])
        vh = torch.stack([v[kv * dims.d_h : (kv + 1) * dims.d_h] for v in values])
        scores[h] = kh @ qh
        weights = (
            scores[h]
            if skip_softmax
            else torch.softmax(scores[h] / math.sqrt(dims.d_h), dim=0)
        )
        output[h * dims.d_h : (h + 1) * dims.d_h] = weights @ vh
    return scores, output


def reference_output_from_weights(
    tokens: Sequence[torch.Tensor],
    x_new: torch.Tensor,
    Wv: torch.Tensor,
    weights: torch.Tensor,
    dims: GQADims,
) -> torch.Tensor:
    """Dense output using already-computed exact or approximate weights."""
    values = [x @ Wv for x in [*tokens, x_new]]
    output = x_new.new_zeros(dims.d)
    for h in range(dims.H):
        kv = h // dims.ratio
        vh = torch.stack([v[kv * dims.d_h : (kv + 1) * dims.d_h] for v in values])
        output[h * dims.d_h : (h + 1) * dims.d_h] = weights[h] @ vh
    return output
