"""Modular GQA decoder-attention pipeline."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .cache import KCache, VCache
from .counter import counter
from .dims import GQAConfig, GQADims, make_gqa_dims
from .encoding import (
    encode_input,
    init_input,
    make_weights_kv,
    make_weights_q_gqa,
)
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
)


def project_qkv(
    x_ct: np.ndarray,
    Wq_encoded: List[List[np.ndarray]],
    Wk_encoded: List[np.ndarray],
    Wv_encoded: List[np.ndarray],
    dims: GQADims,
    pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Q/K/V while sharing direct rotations or BSGS baby rotations."""
    shared_inputs = prepare_shared_projection_inputs(x_ct, dims)
    q_partials = [
        evaluate_projection(shared_inputs, Wq_group, dims)
        for Wq_group in Wq_encoded
    ]
    k_partial = evaluate_projection(shared_inputs, Wk_encoded, dims)
    v_partial = evaluate_projection(shared_inputs, Wv_encoded, dims)

    q_cts = np.stack(
        [replicate_lanes(reduce_lanes(q, dims, pos=0), dims) for q in q_partials]
    )
    k_new = reduce_lanes(k_partial, dims, pos=pos)
    v_new = reduce_lanes(v_partial, dims, pos=pos)
    return q_cts, k_new, v_new


def qkt_gqa(
    q_cts: np.ndarray,
    k_ciphertexts: List[np.ndarray],
    dims: GQADims,
) -> np.ndarray:
    """Compute QK^T. Each score remains repeated across d_h blocks."""
    attention = np.zeros(
        (dims.ratio, len(k_ciphertexts), dims.n_he),
        dtype=np.float64,
    )
    for c in range(dims.ratio):
        for block, k_ct in enumerate(k_ciphertexts):
            folded = multiply_cipher(q_cts[c], k_ct)
            step = dims.B
            while step < dims.n_he:
                folded = folded + rotate_left(folded, step)
                step *= 2
            attention[c, block] = folded
    return attention


def softmax_gqa(
    attention: np.ndarray,
    n_tokens: int,
    dims: GQADims,
) -> np.ndarray:
    """Exact plaintext softmax oracle; its internal HE cost is not counted."""
    out = np.zeros_like(attention)
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
            head_scores_array = np.asarray(head_scores)
            shifted = head_scores_array - np.max(head_scores_array)
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
            for probability, (block, lane) in zip(probabilities, locations):
                for dim in range(dims.d_h):
                    slot = dim * dims.B + kv * dims.t_p + lane
                    out[c, block, slot] = probability
    return out


def softmax_v_gqa(
    probabilities: np.ndarray,
    vcache: VCache,
    dims: GQADims,
) -> np.ndarray:
    """Compute softmax(QK^T)V into one sparse ciphertext per query group."""
    outputs = np.zeros((dims.ratio, dims.n_he), dtype=np.float64)

    for c in range(dims.ratio):
        accumulated_products = np.zeros(dims.n_he, dtype=np.float64)

        # Keep one ct-ct multiplication per cache block.
        for block, v_ct in enumerate(vcache.ciphertexts):
            product = multiply_cipher(probabilities[c, block],v_ct,)
            accumulated_products += product

        # Reduce only once per query group.
        outputs[c] = reduce_lanes(
            accumulated_products,
            dims,
            pos=0,
        )

    return outputs


def attention_gqa(
    x_ct: np.ndarray,
    kcache: KCache,
    vcache: VCache,
    Wq_encoded: List[List[np.ndarray]],
    Wk_encoded: List[np.ndarray],
    Wv_encoded: List[np.ndarray],
    dims: GQADims,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = len(kcache) % dims.t_p

    before = counter.snapshot()
    q_cts, k_new, v_new = project_qkv(
        x_ct, Wq_encoded, Wk_encoded, Wv_encoded, dims, pos
    )
    counter.record_stage("Q/K/V projection", before)

    kcache.append(k_new)
    vcache.append(v_new)

    before = counter.snapshot()
    attention = qkt_gqa(q_cts, kcache.ciphertexts, dims)
    counter.record_stage("QK^T", before)

    probabilities = softmax_gqa(attention, len(kcache), dims)

    before = counter.snapshot()
    output = softmax_v_gqa(probabilities, vcache, dims)
    counter.record_stage("softmax(QK^T)V", before)

    return q_cts, k_new, v_new, attention, probabilities, output


def _prefill_cache(
    tokens: List[np.ndarray],
    Wk_encoded: List[np.ndarray],
    Wv_encoded: List[np.ndarray],
    dims: GQADims,
) -> tuple[KCache, VCache]:
    """Build cache using the same projection math, outside measured decoding."""
    kcache, vcache = KCache(dims), VCache(dims)
    zero_q = [
        [np.zeros(dims.n_he, dtype=np.float64) for _ in range(dims.m_p)]
        for _ in range(dims.ratio)
    ]
    for x in tokens:
        x_ct = encode_input(x, dims)
        _, k_new, v_new = project_qkv(
            x_ct,
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
    seeds: Tuple[int, ...] = (1, 2, 3, 99),
    verbose: bool = True,
) -> tuple[float, float, float, float]:
    dims = make_gqa_dims(
        GQAConfig(
            n_he=n_he,
            d=d,
            H=H,
            n_kv=n_kv,
            n_prefill=n_prefill,
            qkv_method=qkv_method,
            bsgs_baby_steps=bsgs_baby_steps,
        )
    )

    Wq, Wq_encoded = make_weights_q_gqa(dims, seeds[0])
    Wk, Wk_encoded = make_weights_kv(dims, seeds[1])
    Wv, Wv_encoded = make_weights_kv(dims, seeds[2])

    tokens = [init_input(d, 10 + i) for i in range(n_prefill)]
    kcache, vcache = _prefill_cache(tokens, Wk_encoded, Wv_encoded, dims)

    x_new = init_input(d, seeds[3])
    x_ct = encode_input(x_new, dims)

    counter.reset()
    q_cts, k_new, v_new, attention, _, output = attention_gqa(
        x_ct,
        kcache,
        vcache,
        Wq_encoded,
        Wk_encoded,
        Wv_encoded,
        dims,
    )

    q_error = float(np.max(np.abs(decode_q(q_cts, dims) - x_new @ Wq)))
    pos = n_prefill % dims.t_p
    k_error = float(
        np.max(np.abs(decode_positioned_kv(k_new, pos, dims) - x_new @ Wk))
    )
    v_error = float(
        np.max(np.abs(decode_positioned_kv(v_new, pos, dims) - x_new @ Wv))
    )

    reference_qkt, reference_output = reference_attention(
        tokens, x_new, Wq, Wk, Wv, dims
    )
    decoded_qkt = decode_qkt(attention, len(tokens) + 1, dims)
    qkt_error = float(np.max(np.abs(decoded_qkt - reference_qkt)))
    output_error = float(
        np.max(np.abs(decode_output(output, dims) - reference_output))
    )

    if verbose:
        print(
            f"N={dims.n_he} d={dims.d} H={dims.H} n_kv={dims.n_kv} "
            f"(d_h={dims.d_h}, rho={dims.ratio}, d_kv={dims.d_kv}, "
            f"t_p={dims.t_p}, m_p={dims.m_p})"
        )
        if dims.qkv_method == "bsgs":
            print(
                f"Q/K/V method=bsgs "
                f"(baby={dims.bsgs_baby_steps}, giant={dims.bsgs_giant_steps})"
            )
        else:
            print("Q/K/V method=direct")
        print(
            f"errors: Q={q_error:.2e}, K={k_error:.2e}, "
            f"V={v_error:.2e}, QK^T={qkt_error:.2e}, "
            f"output={output_error:.2e}"
        )

        def print_counts(name, values):
            amounts = str(sorted(values["rotation_amounts"]))

            print(
                f"{name + ':':<22}"
                f"rotations={values['rotations']:<4}  "
                f"unique={values['unique_rotation_amounts']:<4}  "
                # f"amounts={amounts:<38}  "
                f"ct-pt={values['ct_pt_mult']:<4}  "
                f"ct-ct={values['ct_ct_mult']:<4}"
            )


        for name, values in counter.stages.items():
            print_counts(name, values)

        total_values = {
            "rotations": counter.rotations,
            "unique_rotation_amounts": len(counter.rotation_amounts),
            "rotation_amounts": counter.rotation_amounts,
            "ct_pt_mult": counter.ct_pt_mult,
            "ct_ct_mult": counter.ct_ct_mult,
        }

        print_counts("total", total_values)

    return (
        max(q_error, k_error, v_error),
        qkt_error,
        output_error,
        float(counter.rotations),
    )
