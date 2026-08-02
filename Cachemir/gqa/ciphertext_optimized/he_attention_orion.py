"""Ciphertext-domain Orion implementation of persistent-layout GQA attention."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..plaintext_optimized.attention import _prefill_cache
from ..plaintext_optimized.counter import counter
from ..plaintext_optimized.dims import GQAConfig, GQADims, make_gqa_dims
from ..plaintext_optimized.encoding import init_input
from ..plaintext_optimized.reference import (
    decode_output,
    decode_positioned_kv,
    decode_q,
    decode_qkt,
    reference_attention,
)
from .he_cache_orion import HEKCache, HEVCache
from .he_encoding_orion import (
    decrypt_decode_ct,
    encrypt_input,
    encrypt_slot_vector,
    make_he_weights_kv,
    make_he_weights_q_gqa,
)

from .he_ops_orion import (
    evaluate_projection,
    multiply_cipher,
    multiply_cipher_raw,
    prepare_shared_projection_inputs,
    reduce_lanes,
    replicate_lanes,
    rotate_left,
)

SoftmaxEvaluator = Callable[
    [list[list[Any]], int, GQADims],
    list[list[Any]],
]


@dataclass(frozen=True)
class PreparedAttentionGQAHE:
    """One-time state reused by warm-up, benchmark, and verification runs."""

    dims: GQADims
    level: int
    tokens: list[np.ndarray]
    x_new: np.ndarray
    x_ct: Any
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wq_encoded: Sequence[Sequence[Any]]
    Wk_encoded: Sequence[Any]
    Wv_encoded: Sequence[Any]
    initial_kcache: HEKCache
    initial_vcache: HEVCache
    softmax_evaluator: Optional[SoftmaxEvaluator]
    skip_softmax: bool


@dataclass(frozen=True)
class AttentionGQAHEResult:
    q_cts: list[Any]
    k_new: Any
    v_new: Any
    attention: list[list[Any]]
    weights: list[list[Any]]
    output: list[Any]


def _clone_cache(cache: HEKCache | HEVCache) -> HEKCache | HEVCache:
    """Clone cache metadata/list while safely reusing existing ciphertexts."""
    cloned = type(cache)(cache.dims)
    cloned.ciphertexts = list(cache.ciphertexts)
    cloned.length = cache.length
    return cloned


def project_qkv_he(
    x_ct: Any,
    Wq_encoded: Sequence[Sequence[Any]],
    Wk_encoded: Sequence[Any],
    Wv_encoded: Sequence[Any],
    dims: GQADims,
    pos: int,
) -> tuple[list[Any], Any, Any]:
    """Project Q/K/V while sharing X rotations across every projection."""
    shared_inputs = prepare_shared_projection_inputs(x_ct, dims)
    q_partials = [
        evaluate_projection(shared_inputs, group, dims)
        for group in Wq_encoded
    ]
    k_partial = evaluate_projection(shared_inputs, Wk_encoded, dims)
    v_partial = evaluate_projection(shared_inputs, Wv_encoded, dims)

    q_cts = [
        replicate_lanes(reduce_lanes(q, dims, pos=0), dims)
        for q in q_partials
    ]
    k_new = reduce_lanes(k_partial, dims, pos=pos)
    v_new = reduce_lanes(v_partial, dims, pos=pos)
    return q_cts, k_new, v_new


def qkt_gqa_he(
    q_cts: Sequence[Any],
    k_ciphertexts: Sequence[Any],
    dims: GQADims,
) -> list[list[Any]]:
    """Compute packed QK^T; scores remain repeated across d_h blocks."""
    attention: list[list[Any]] = []
    for q_ct in q_cts:
        row: list[Any] = []
        for k_ct in k_ciphertexts:
            folded = multiply_cipher(q_ct, k_ct)
            step = dims.B
            while step < dims.n_he:
                folded = folded + rotate_left(folded, step, dims)
                step *= 2
            row.append(folded)
        attention.append(row)
    return attention


def softmax_v_gqa_he(
    probabilities: Sequence[Sequence[Any]],
    vcache: HEVCache,
    dims: GQADims,
) -> list[Any]:
    """Compute encrypted softmax(QK^T)V with one lane reduction per group."""
    if len(probabilities) != dims.ratio:
        raise ValueError(
            f"Expected {dims.ratio} probability rows; got {len(probabilities)}."
        )
    if not vcache.ciphertexts:
        raise ValueError("V cache cannot be empty.")

    outputs: list[Any] = []
    for c, row in enumerate(probabilities):
        if len(row) != len(vcache.ciphertexts):
            raise ValueError(
                f"Probability row {c} has {len(row)} blocks, but V cache "
                f"has {len(vcache.ciphertexts)}."
            )

        accumulated = None

        for probability_ct, v_ct in zip(row, vcache.ciphertexts):
            product = multiply_cipher_raw(probability_ct, v_ct)

            accumulated = (
                product
                if accumulated is None
                else accumulated + product
            )

        # One relinearization and rescale for the entire cache row.
        accumulated = accumulated.relinearize()
        accumulated = accumulated.rescale()

        outputs.append(
            reduce_lanes(accumulated, dims, pos=0)
        )
    return outputs


def attention_gqa_he(
    x_ct: Any,
    kcache: HEKCache,
    vcache: HEVCache,
    Wq_encoded: Sequence[Sequence[Any]],
    Wk_encoded: Sequence[Any],
    Wv_encoded: Sequence[Any],
    dims: GQADims,
    *,
    softmax_evaluator: Optional[SoftmaxEvaluator] = None,
    skip_softmax: bool = False,
) -> tuple[
    list[Any],
    Any,
    Any,
    list[list[Any]],
    list[list[Any]],
    list[Any],
]:
    """Run one encrypted decoding step.

    When ``skip_softmax`` is true, the encrypted QK^T scores are passed
    directly to the V multiplication. Otherwise, ``softmax_evaluator`` must
    preserve the packed shape
    ``[ratio][cache_block]`` and return encrypted probabilities. It may use
    any polynomial/normalization implementation, but must not decrypt.
    """
    if not skip_softmax and softmax_evaluator is None:
        raise ValueError(
            "A ciphertext softmax_evaluator is required. "
            "Pass skip_softmax=True to compute QK^T V without softmax."
        )
    if len(kcache) != len(vcache):
        raise ValueError("K and V caches must contain the same number of tokens.")

    pos = len(kcache) % dims.t_p
    total_start = time.perf_counter()

    before = counter.snapshot()
    stage_start = time.perf_counter()
    level_before = x_ct.level()
    q_cts, k_new, v_new = project_qkv_he(
        x_ct, Wq_encoded, Wk_encoded, Wv_encoded, dims, pos
    )
    counter.record_stage(
        "Q/K/V projection",
        before,
        time.perf_counter() - stage_start,
        level_before=level_before,
        level_after=k_new.level(),
    )

    kcache.append(k_new)
    vcache.append(v_new)

    before = counter.snapshot()
    stage_start = time.perf_counter()
    level_before = q_cts[0].level()
    attention = qkt_gqa_he(q_cts, kcache.ciphertexts, dims)
    counter.record_stage(
        "QK^T",
        before,
        time.perf_counter() - stage_start,
        level_before=level_before,
        level_after=attention[0][0].level(),
    )

    if skip_softmax:
        weights = attention
    else:
        before = counter.snapshot()
        stage_start = time.perf_counter()
        level_before = attention[0][0].level()
        weights = softmax_evaluator(attention, len(kcache), dims)
        counter.record_stage(
            "Softmax",
            before,
            time.perf_counter() - stage_start,
            level_before=level_before,
            level_after=weights[0][0].level(),
        )

    before = counter.snapshot()
    stage_start = time.perf_counter()
    level_before = weights[0][0].level()
    output = softmax_v_gqa_he(weights, vcache, dims)
    stage_name = "QK^T V" if skip_softmax else "Softmax(QK^T)V"
    counter.record_stage(
        stage_name,
        before,
        time.perf_counter() - stage_start,
        level_before=level_before,
        level_after=output[0].level(),
    )
    counter.total_runtime_seconds = time.perf_counter() - total_start

    return q_cts, k_new, v_new, attention, weights, output


def _encrypt_plaintext_prefill(
    tokens: list[np.ndarray],
    Wk_raw_encoded: list[np.ndarray],
    Wv_raw_encoded: list[np.ndarray],
    dims: GQADims,
    *,
    output_level: int,
) -> tuple[HEKCache, HEVCache]:
    """Compute optional client-side prefill and encrypt final packed blocks."""
    k_plain, v_plain = _prefill_cache(
        tokens, Wk_raw_encoded, Wv_raw_encoded, dims
    )
    kcache, vcache = HEKCache(dims), HEVCache(dims)
    for block in k_plain.ciphertexts:
        kcache.ciphertexts.append(
            encrypt_slot_vector(block, dims, level=output_level)
        )
    for block in v_plain.ciphertexts:
        vcache.ciphertexts.append(
            encrypt_slot_vector(block, dims, level=output_level)
        )
    kcache.length = k_plain.length
    vcache.length = v_plain.length
    return kcache, vcache


def prepare_attention_gqa_he(
    n_he: int = 32,
    d: int = 16,
    H: int = 4,
    n_kv: int = 2,
    n_prefill: int = 0,
    level: int = 8,
    qkv_method: str = "direct",
    bsgs_baby_steps: Optional[int] = None,
    seeds: tuple[int, ...] = (1, 2, 3, 99),
    softmax_evaluator: Optional[SoftmaxEvaluator] = None,
    skip_softmax: bool = False,
) -> PreparedAttentionGQAHE:
    """Prepare dimensions, weights, input, and prefill caches exactly once."""
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

    Wq, Wq_encoded, _ = make_he_weights_q_gqa(
        dims, seed=seeds[0], level=level
    )
    Wk, Wk_encoded, Wk_raw_encoded = make_he_weights_kv(
        dims, seed=seeds[1], level=level
    )
    Wv, Wv_encoded, Wv_raw_encoded = make_he_weights_kv(
        dims, seed=seeds[2], level=level
    )

    tokens = [init_input(d, 10 + i) for i in range(n_prefill)]
    # Projection: diagonal ct-pt multiplication then lane-mask ct-pt
    projection_output_level = level - 2
    kcache, vcache = _encrypt_plaintext_prefill(
        tokens,
        Wk_raw_encoded,
        Wv_raw_encoded,
        dims,
        output_level=projection_output_level,
    )

    x_new = init_input(d, seeds[3])
    x_ct = encrypt_input(x_new, dims, level=level)
    if not skip_softmax and softmax_evaluator is None:
        raise ValueError(
            "Pass your Orion ciphertext softmax as softmax_evaluator, or set "
            "skip_softmax=True to compute QK^T V."
        )

    return PreparedAttentionGQAHE(
        dims=dims,
        level=level,
        tokens=tokens,
        x_new=x_new,
        x_ct=x_ct,
        Wq=Wq,
        Wk=Wk,
        Wv=Wv,
        Wq_encoded=Wq_encoded,
        Wk_encoded=Wk_encoded,
        Wv_encoded=Wv_encoded,
        initial_kcache=kcache,
        initial_vcache=vcache,
        softmax_evaluator=softmax_evaluator,
        skip_softmax=skip_softmax,
    )


def execute_prepared_attention_gqa_he(
    prepared: PreparedAttentionGQAHE,
) -> AttentionGQAHEResult:
    """Run attention from the same initial cache state.

    Only the cache containers are cloned. Scheme state, keys, encoded weights,
    encrypted input, and encrypted prefill blocks are reused.
    """
    kcache = _clone_cache(prepared.initial_kcache)
    vcache = _clone_cache(prepared.initial_vcache)

    counter.reset()
    q_cts, k_new, v_new, attention, weights, output = attention_gqa_he(
        prepared.x_ct,
        kcache,
        vcache,
        prepared.Wq_encoded,
        prepared.Wk_encoded,
        prepared.Wv_encoded,
        prepared.dims,
        softmax_evaluator=prepared.softmax_evaluator,
        skip_softmax=prepared.skip_softmax,
    )

    return AttentionGQAHEResult(
        q_cts=q_cts,
        k_new=k_new,
        v_new=v_new,
        attention=attention,
        weights=weights,
        output=output,
    )


def print_attention_gqa_he_stats(prepared: PreparedAttentionGQAHE) -> None:
    dims = prepared.dims
    print(
        f"N={dims.n_he} d={dims.d} H={dims.H} n_kv={dims.n_kv} "
        f"d_h={dims.d_h} rho={dims.ratio} d_kv={dims.d_kv} "
        f"t_p={dims.t_p} method={dims.qkv_method}"
    )
    for name, values in counter.stages.items():
        level_str = ""
        if values["level_before"] is not None:
            level_str = (
                f"level={values['level_before']}->{values['level_after']:<4}  "
            )
        print(
            f"{name + ':':<22}"
            f"rotations={values['rotations']:<4}  "
            f"unique={values['unique_rotation_amounts']:<4}  "
            f"ct-pt={values['ct_pt_mult']:<4}  "
            f"ct-ct={values['ct_ct_mult']:<4}  "
            f"{level_str}"
            f"runtime={values['runtime_seconds']:.6f}s"
        )
    print(f"{'total:':<22}{counter.summary()}")


def verify_prepared_attention_gqa_he(
    prepared: PreparedAttentionGQAHE,
    result: AttentionGQAHEResult,
    *,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Decrypt one un-timed result and compare it with the dense reference."""
    dims = prepared.dims
    q_decoded = np.stack([decrypt_decode_ct(ct) for ct in result.q_cts])
    k_decoded = decrypt_decode_ct(result.k_new)
    v_decoded = decrypt_decode_ct(result.v_new)
    attention_decoded = np.asarray(
        [[decrypt_decode_ct(ct) for ct in row] for row in result.attention]
    )
    output_decoded = np.stack(
        [decrypt_decode_ct(ct) for ct in result.output]
    )

    pos = len(prepared.tokens) % dims.t_p
    projection_error = max(
        float(
            np.max(
                np.abs(
                    decode_q(q_decoded, dims)
                    - prepared.x_new @ prepared.Wq
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    decode_positioned_kv(k_decoded, pos, dims)
                    - prepared.x_new @ prepared.Wk
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    decode_positioned_kv(v_decoded, pos, dims)
                    - prepared.x_new @ prepared.Wv
                )
            )
        ),
    )
    reference_qkt, reference_output = reference_attention(
        prepared.tokens,
        prepared.x_new,
        prepared.Wq,
        prepared.Wk,
        prepared.Wv,
        dims,
        skip_softmax=prepared.skip_softmax,
    )
    qkt_error = float(
        np.max(
            np.abs(
                decode_qkt(
                    attention_decoded,
                    len(prepared.tokens) + 1,
                    dims,
                )
                - reference_qkt
            )
        )
    )
    output_error = float(
        np.max(
            np.abs(decode_output(output_decoded, dims) - reference_output)
        )
    )

    if verbose:
        print(
            f"errors: projection={projection_error:.2e}, "
            f"QK^T={qkt_error:.2e}, output={output_error:.2e}"
        )

    return projection_error, qkt_error, output_error


def run_attention_gqa_he(
    n_he: int = 32,
    d: int = 16,
    H: int = 4,
    n_kv: int = 2,
    n_prefill: int = 0,
    level: int = 8,
    qkv_method: str = "direct",
    bsgs_baby_steps: Optional[int] = None,
    seeds: tuple[int, ...] = (1, 2, 3, 99),
    softmax_evaluator: Optional[SoftmaxEvaluator] = None,
    skip_softmax: bool = False,
    verify: bool = True,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Compatibility wrapper for a one-off setup, execution, and check."""
    prepared = prepare_attention_gqa_he(
        n_he=n_he,
        d=d,
        H=H,
        n_kv=n_kv,
        n_prefill=n_prefill,
        level=level,
        qkv_method=qkv_method,
        bsgs_baby_steps=bsgs_baby_steps,
        seeds=seeds,
        softmax_evaluator=softmax_evaluator,
        skip_softmax=skip_softmax,
    )
    result = execute_prepared_attention_gqa_he(prepared)

    if verbose:
        print_attention_gqa_he_stats(prepared)
    if not verify:
        return float("nan"), float("nan"), float("nan")
    return verify_prepared_attention_gqa_he(
        prepared,
        result,
        verbose=verbose,
    )