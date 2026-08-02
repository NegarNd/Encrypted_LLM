"""Modular PyTorch GQA decoder-attention pipeline."""

from __future__ import annotations

import math
import time
from typing import Sequence

import torch

from .cache import KCache, VCache
from .counter import counter
from .dims import GQAConfig, GQADims, make_gqa_dims
from .encoding import encode_input, init_input, make_weights_kv, make_weights_q_gqa
from .ops import (
    evaluate_projection,
    multiply_cipher,
    prepare_shared_projection_inputs,
    reduce_lanes,
    replicate_lanes,
    rotate_left,
)
from .reference import (
    decode_output,
    decode_positioned_kv,
    decode_q,
    decode_qkt,
    reference_attention,
    reference_output_from_weights,
)
from .softmax import app_softmax_gqa, softmax_gqa


def project_qkv(
    x_ct: torch.Tensor,
    Wq_encoded: Sequence[Sequence[torch.Tensor]],
    Wk_encoded: Sequence[torch.Tensor],
    Wv_encoded: Sequence[torch.Tensor],
    dims: GQADims,
    pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shared_inputs = prepare_shared_projection_inputs(x_ct, dims)
    q_partials = [
        evaluate_projection(shared_inputs, Wq_group, dims)
        for Wq_group in Wq_encoded
    ]
    k_partial = evaluate_projection(shared_inputs, Wk_encoded, dims)
    v_partial = evaluate_projection(shared_inputs, Wv_encoded, dims)
    q_cts = torch.stack(
        [replicate_lanes(reduce_lanes(q, dims), dims) for q in q_partials]
    )
    return (
        q_cts,
        reduce_lanes(k_partial, dims, pos=pos),
        reduce_lanes(v_partial, dims, pos=pos),
    )


def qkt_gqa(
    q_cts: torch.Tensor,
    k_ciphertexts: Sequence[torch.Tensor],
    dims: GQADims,
) -> torch.Tensor:
    groups = []
    for c in range(dims.ratio):
        blocks = []
        for k_ct in k_ciphertexts:
            folded = multiply_cipher(q_cts[c], k_ct)
            step = dims.B
            while step < dims.n_he:
                folded = folded + rotate_left(folded, step)
                step *= 2
            blocks.append(folded)
        groups.append(torch.stack(blocks))
    return torch.stack(groups)


def softmax_v_gqa(
    probabilities: torch.Tensor,
    vcache: VCache,
    dims: GQADims,
) -> torch.Tensor:
    outputs = []
    for c in range(dims.ratio):
        accumulated = torch.zeros_like(vcache.ciphertexts[0])
        for block, v_ct in enumerate(vcache.ciphertexts):
            accumulated = accumulated + multiply_cipher(probabilities[c, block], v_ct)
        outputs.append(reduce_lanes(accumulated, dims))
    return torch.stack(outputs)


def attention_gqa(
    x_ct: torch.Tensor,
    kcache: KCache,
    vcache: VCache,
    Wq_encoded: Sequence[Sequence[torch.Tensor]],
    Wk_encoded: Sequence[torch.Tensor],
    Wv_encoded: Sequence[torch.Tensor],
    dims: GQADims,
    *,
    skip_softmax: bool = False,
    softmax_method: str = "approx",
    softmax_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = len(kcache) % dims.t_p
    total_start = time.perf_counter()

    before, stage_start = counter.snapshot(), time.perf_counter()
    q_cts, k_new, v_new = project_qkv(
        x_ct, Wq_encoded, Wk_encoded, Wv_encoded, dims, pos
    )
    counter.record_stage("Q/K/V projection", before, time.perf_counter() - stage_start)
    kcache.append(k_new)
    vcache.append(v_new)

    before, stage_start = counter.snapshot(), time.perf_counter()
    attention = qkt_gqa(q_cts, kcache.ciphertexts, dims)
    counter.record_stage("QK^T", before, time.perf_counter() - stage_start)

    if skip_softmax:
        weights = attention
    else:
        before, stage_start = counter.snapshot(), time.perf_counter()
        if softmax_method == "exact":
            weights = softmax_gqa(attention, len(kcache), dims)
        elif softmax_method == "approx":
            weights = app_softmax_gqa(
                attention,
                dims,
                len(kcache),
                **(softmax_kwargs or {}),
            )
        else:
            raise ValueError("softmax_method must be 'exact' or 'approx'.")
        counter.record_stage("Softmax", before, time.perf_counter() - stage_start)

    before, stage_start = counter.snapshot(), time.perf_counter()
    output = softmax_v_gqa(weights, vcache, dims)
    stage_name = "QK^T V" if skip_softmax else "Softmax(QK^T)V"
    counter.record_stage(stage_name, before, time.perf_counter() - stage_start)
    counter.total_runtime_seconds = time.perf_counter() - total_start
    return q_cts, k_new, v_new, attention, weights, output


def _prefill_cache(
    tokens: Sequence[torch.Tensor],
    Wk_encoded: Sequence[torch.Tensor],
    Wv_encoded: Sequence[torch.Tensor],
    dims: GQADims,
) -> tuple[KCache, VCache]:
    kcache, vcache = KCache(dims), VCache(dims)
    zero_q = [
        [torch.zeros_like(Wk_encoded[0]) for _ in range(dims.m_p)]
        for _ in range(dims.ratio)
    ]
    for x in tokens:
        _, k_new, v_new = project_qkv(
            encode_input(x, dims),
            zero_q,
            Wk_encoded,
            Wv_encoded,
            dims,
            pos=len(kcache) % dims.t_p,
        )
        kcache.append(k_new)
        vcache.append(v_new)
    return kcache, vcache


def run_attention_gqa(
    n_he: int = 32,
    d: int = 16,
    H: int = 4,
    n_kv: int = 2,
    n_prefill: int = 0,
    qkv_method: str = "direct",
    bsgs_baby_steps: int | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 99),
    skip_softmax: bool = False,
    softmax_method: str = "approx",
    verbose: bool = True,
    *,
    softmax_kwargs: dict | None = None,
) -> tuple[float, float, float, float]:
    dims = make_gqa_dims(
        GQAConfig(n_he, d, H, n_kv, n_prefill, qkv_method, bsgs_baby_steps)
    )
    Wq, Wq_encoded = make_weights_q_gqa(dims, seeds[0])
    Wk, Wk_encoded = make_weights_kv(dims, seeds[1])
    Wv, Wv_encoded = make_weights_kv(dims, seeds[2])
    tokens = [init_input(d, 10 + i) for i in range(n_prefill)]
    kcache, vcache = _prefill_cache(tokens, Wk_encoded, Wv_encoded, dims)
    x_new = init_input(d, seeds[3])

    counter.reset()
    q_cts, k_new, v_new, attention, weights, output = attention_gqa(
        encode_input(x_new, dims),
        kcache,
        vcache,
        Wq_encoded,
        Wk_encoded,
        Wv_encoded,
        dims,
        skip_softmax=skip_softmax,
        softmax_method=softmax_method,
        softmax_kwargs=softmax_kwargs,
    )

    def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
        return torch.max(torch.abs(left - right)).item()

    q_error = max_error(decode_q(q_cts, dims), x_new @ Wq)
    pos = n_prefill % dims.t_p
    k_error = max_error(decode_positioned_kv(k_new, pos, dims), x_new @ Wk)
    v_error = max_error(decode_positioned_kv(v_new, pos, dims), x_new @ Wv)
    reference_qkt, exact_reference_output = reference_attention(
        tokens, x_new, Wq, Wk, Wv, dims, skip_softmax=skip_softmax
    )
    decoded_qkt = decode_qkt(attention, len(tokens) + 1, dims)
    qkt_error = max_error(decoded_qkt, reference_qkt)
    decoded_weights = decode_qkt(weights, len(tokens) + 1, dims)

    if skip_softmax:
        reference_output = exact_reference_output
        approximation_error = 0.0
    else:
        reference_output = reference_output_from_weights(
            tokens, x_new, Wv, decoded_weights, dims
        )
        if softmax_method == "exact":
            exact_weights = torch.softmax(reference_qkt, dim=1)
        else:
            exact_weights = torch.softmax(reference_qkt / math.sqrt(dims.d_h), dim=1)
        approximation_error = max_error(decoded_weights, exact_weights)
    output_error = max_error(decode_output(output, dims), reference_output)

    if verbose:
        print(
            f"N={dims.n_he} d={dims.d} H={dims.H} n_kv={dims.n_kv} "
            f"(d_h={dims.d_h}, rho={dims.ratio}, d_kv={dims.d_kv}, "
            f"t_p={dims.t_p}, m_p={dims.m_p})"
        )
        print(
            f"Q/K/V method={dims.qkv_method}" +
            (f" (baby={dims.bsgs_baby_steps}, giant={dims.bsgs_giant_steps})"
             if dims.qkv_method == "bsgs" else "")
        )
        print(
            f"errors: Q={q_error:.2e}, K={k_error:.2e}, V={v_error:.2e}, "
            f"QK^T={qkt_error:.2e}, output-layout={output_error:.2e}"
        )
        if not skip_softmax and softmax_method == "approx":
            print(f"approx-softmax vs exact-softmax max error={approximation_error:.2e}")
        for name, values in counter.stages.items():
            print(
                f"{name + ':':<22}rotations={values['rotations']:<4}  "
                f"unique={values['unique_rotation_amounts']:<4}  "
                f"ct-pt={values['ct_pt_mult']:<4}  "
                f"ct-ct={values['ct_ct_mult']:<4}  "
                f"runtime={values['runtime_seconds']:.6f}s"
            )

    return max(q_error, k_error, v_error), qkt_error, output_error, float(counter.rotations)
