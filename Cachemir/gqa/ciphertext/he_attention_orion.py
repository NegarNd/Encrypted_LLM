"""Ciphertext-domain GQA attention using Orion.

This mirrors attention.py.  The client-side input encoding and all weight
encodings remain plaintext torch tensors.  Only X chunks, K/V cache entries,
Q/K/V projections, attention scores, and output are Orion ciphertexts.
"""

from __future__ import annotations

import math
import time
from typing import Any, List, Tuple

import orion
import torch

from ..plaintext.dims import GQAConfig, GQADims, make_gqa_dims
from ..plaintext.encoding import head_perm, init_input
from ..plaintext.reference import compare_attention_outputs
from ..plaintext.counter import counter
from .he_cache_orion import HEKCache, HEVCache
from .he_ops_orion import block_replicate, vmm_kv, ct_zero
from .he_encoding_orion import (
    make_he_weights_kv,
    make_he_weights_q_gqa,
    encrypt_sparse_input_kv,
    expand_sparse_input_kv_he,
    decrypt_decode_ct,
)


def vmm_q_gqa_he(
    X_enc_chunks_ct: List[Any],
    Wq_list: List[List[torch.Tensor]],
    dims: GQADims,
    level: int,
) -> Tuple[List[Any], int]:
    """Compute encrypted Q ciphertexts in sparse canonical layout."""
    Q_cts: List[Any] = []
    level_out = level
    for c in range(dims.ratio):
        q_ct, level_out = vmm_kv(X_enc_chunks_ct, Wq_list[c], dims, level, pos=0)
        Q_cts.append(q_ct)
    return Q_cts, level_out


def qkt_gqa_he(
    Q_cts: List[Any],
    k_ciphertexts: List[Any],
    dims: GQADims,
    level: int,
) -> Tuple[List[List[Any]], int]:
    """Compute encrypted QK^T scores against encrypted cached keys."""
    fold_nsteps = int(math.log2(dims.d_h))
    att_cts: List[List[Any]] = []

    for c in range(dims.ratio):
        Q_rep = block_replicate(Q_cts[c], dims.t_p)
        row: List[Any] = []
        for k_ct in k_ciphertexts:
            counter.ct_ct_mult += 1
            folded = Q_rep * k_ct
            step = dims.B
            for _ in range(fold_nsteps):
                counter.rotations += 1
                folded = folded + folded.roll(step)
                step *= 2
            row.append(folded)
        att_cts.append(row)

    return att_cts, level - 1


def softmax_v_gqa_he(
    att_cts: List[List[Any]],
    vcache: HEVCache,
    dims: GQADims,
    level: int,
) -> Tuple[List[Any], int]:
    """Compute scores times encrypted V cache.

    TODO: does NOT evaluate softmax yet, as in the plaintext simulator --
    uses the structural attention scores directly.
    """
    out_mask = torch.zeros(dims.n_he, dtype=torch.float64)
    out_mask[:: dims.t_p] = 1.0

    prod_level = level - 1
    out_mask_pt = orion.encode(out_mask, prod_level)
    out_level = prod_level - 1

    O_ct: List[Any] = []
    for c in range(dims.ratio):
        if len(vcache.ciphertexts) == 0:
            raise ValueError("vcache is empty; append at least one V ciphertext first.")
        out = ct_zero(dims.n_he, out_level)

        for b, att in enumerate(att_cts[c]):
            if b >= len(vcache.ciphertexts):
                break
            counter.ct_ct_mult += 1
            prod = att * vcache.ciphertexts[b]
            step = 1
            while step < dims.t_p:
                counter.rotations += 1
                prod = prod + prod.roll(step)
                step *= 2
            counter.ct_pt_mult += 1
            out = out + prod * out_mask_pt

        O_ct.append(out)

    return O_ct, out_level


# --- op-count / timing instrumentation --------------------------------------

def _ct_level(x: Any) -> int:
    """Get the real Orion ciphertext level, descending into (possibly nested) lists."""
    while isinstance(x, list):
        x = x[0]
    return x.level()


def _op_counts() -> Tuple[int, int, int]:
    """Snapshot the running (rotations, ct-pt, ct-ct) counters."""
    return counter.rotations, counter.ct_pt_mult, counter.ct_ct_mult


def _log_ops(label: str, before: Tuple[int, int, int]) -> None:
    """Print rotations/ct-pt/ct-ct consumed since `before` was snapshotted."""
    rot0, ctpt0, ctct0 = before
    print(
        f"[{label:<40}] "
        f"rotations={counter.rotations - rot0:5d}  "
        f"ct-pt={counter.ct_pt_mult - ctpt0:4d}  "
        f"ct-ct={counter.ct_ct_mult - ctct0:4d}"
    )


def _run_step(name: str, in_ct: Any, fn, *args: Any, **kwargs: Any) -> Tuple[Any, int]:
    """Run one HE step, logging its level transition, wall time, and op counts.

    `fn` must return `(value, _)`; the level is read directly off `value` via
    Orion (`_ct_level`) rather than trusting the int the function computed.
    """
    in_level = _ct_level(in_ct)
    before = _op_counts()
    t_start = time.perf_counter()

    value, _ = fn(*args, **kwargs)

    elapsed = time.perf_counter() - t_start
    out_level = _ct_level(value)
    rot0, ctpt0, ctct0 = before
    print(
        f"[attention_gqa_he] {name:<28} level {in_level:>2} -> {out_level:<2}  "
        f"time={elapsed:8.4f}s  "
        f"rotations={counter.rotations - rot0:5d}  "
        f"ct-pt={counter.ct_pt_mult - ctpt0:4d}  "
        f"ct-ct={counter.ct_ct_mult - ctct0:4d}"
    )
    return value, out_level


# --- attention pipeline ------------------------------------------------------

def attention_gqa_he(
    X_enc_chunks_ct: List[Any],
    kcache: HEKCache,
    vcache: HEVCache,
    Wq_list: List[List[torch.Tensor]],
    Wk_list: List[torch.Tensor],
    Wv_list: List[torch.Tensor],
    dims: GQADims,
    level: int,
) -> Tuple[List[Any], Any, Any, List[List[Any]], List[Any], int]:
    """Execute one encrypted GQA decoding step, logging level/time/op-counts per stage."""
    Q_cts, q_level = _run_step(
        "Q (vmm_q_gqa_he)", X_enc_chunks_ct,
        vmm_q_gqa_he, X_enc_chunks_ct, Wq_list, dims, level,
    )

    pos_k = len(kcache) % dims.t_p
    K_new, k_level = _run_step(
        "K (vmm_kv)", X_enc_chunks_ct,
        vmm_kv, X_enc_chunks_ct, Wk_list, dims, level, pos=pos_k,
    )
    kcache.append(K_new)

    pos_v = len(vcache) % dims.t_p
    V_new, v_level = _run_step(
        "V (vmm_kv)", X_enc_chunks_ct,
        vmm_kv, X_enc_chunks_ct, Wv_list, dims, level, pos=pos_v,
    )
    vcache.append(V_new)

    # Q, K, V all start from the same input `level` and each pass through
    # exactly one vmm_kv call, so they land at the same level.
    assert q_level == k_level == v_level

    att_cts, att_level = _run_step(
        "QK^T (qkt_gqa_he)", Q_cts,
        qkt_gqa_he, Q_cts, kcache.ciphertexts, dims, q_level,
    )

    O, out_level = _run_step(
        "softmax.V (softmax_v_gqa_he)", att_cts,
        softmax_v_gqa_he, att_cts, vcache, dims, att_level,
    )

    return Q_cts, K_new, V_new, att_cts, O, out_level


def run_attention_gqa_he(
    n_he: int,
    d: int,
    H: int,
    n_kv: int,
    n_prefill: int,
    level: int,
    seeds: Tuple[int, ...] = (1, 2, 3, 99),
    verify: bool = True,
    verbose: bool = True,
) -> Tuple[float, float]:
    """End-to-end encrypted test driver."""
    dims = make_gqa_dims(GQAConfig(n_he=n_he, d=d, H=H, n_kv=n_kv, n_prefill=n_prefill))

    Wq_raw, Wq_list = make_he_weights_q_gqa(dims, seed=seeds[0], n_he=n_he, level=level)
    Wk_raw, Wk_list = make_he_weights_kv(dims, seed=seeds[1], n_he=n_he, level=level)
    Wv_raw, Wv_list = make_he_weights_kv(dims, seed=seeds[2], n_he=n_he, level=level)

    kcache = HEKCache(n_he, dims.d_kv)
    vcache = HEVCache(n_he, dims.d_kv, H=dims.n_kv)

    toks = [init_input(dims.d, seed=10 + i) for i in range(n_prefill)]

    before = _op_counts()
    for x in toks:
        x_sparse_ct = encrypt_sparse_input_kv(x, dims, level=level)
        xc_ct = expand_sparse_input_kv_he(x_sparse_ct, dims)

        k_new, _ = vmm_kv(xc_ct, Wk_list, dims, level, pos=len(kcache) % dims.t_p)
        kcache.append(k_new)

        v_new, _ = vmm_kv(xc_ct, Wv_list, dims, level, pos=len(vcache) % dims.t_p)
        vcache.append(v_new)
    _log_ops(f"run_attention_gqa_he: fill caches ({n_prefill} tok)", before)

    before = _op_counts()
    x_new = init_input(dims.d, seed=seeds[3])
    x_sparse_ct = encrypt_sparse_input_kv(x_new, dims, level=level)
    xc_new_ct = expand_sparse_input_kv_he(x_sparse_ct, dims)
    _log_ops("run_attention_gqa_he: generate input", before)

    Q_cts, K_new, V_new, att_cts, O_cts, out_level = attention_gqa_he(
        xc_new_ct, kcache, vcache, Wq_list, Wk_list, Wv_list, dims, level,
    )

    if not verify:
        if verbose:
            print(
                f"encrypted run done (level {level} -> {out_level}): "
                f"rotations={counter.rotations}, ct-pt={counter.ct_pt_mult}, "
                f"ct-ct={counter.ct_ct_mult}"
            )
        return float("nan"), float("nan")

    # Decrypt only for local correctness checking.
    att_torch = torch.stack(
        [torch.stack([decrypt_decode_ct(ct) for ct in row]) for row in att_cts]
    )
    O_torch = torch.stack([decrypt_decode_ct(ct) for ct in O_cts])

    err_qkt, err_v = compare_attention_outputs(
        toks=toks,
        x_new=x_new,
        Wq_raw=Wq_raw,
        Wk_raw=Wk_raw,
        Wv_raw=Wv_raw,
        att_cts=att_torch,
        O=O_torch,
        dims=dims,
        head_perm=head_perm,
        apply_softmax=False,
    )

    if verbose:
        ok = max(err_qkt, err_v) < 5e-3
        print(
            f"\nn_he={dims.n_he} d={dims.d} H={dims.H} n_kv={dims.n_kv} "
            f"(d_h={dims.d_h}, ratio={dims.ratio}, d_kv={dims.d_kv}, "
            f"t_p={dims.t_p}, B={dims.B}, R={dims.R}) "
            f"n_prefill={n_prefill} (n={len(kcache)}): "
            f"qkt_err={err_qkt:.2e}, softmaxv_err={err_v:.2e} -> "
            f"{'PASS' if ok else 'CHECK'}"
        )
        print(
            f"level: {level} -> {out_level}   "
            f"ops: rotations={counter.rotations}, "
            f"ct-pt={counter.ct_pt_mult}, ct-ct={counter.ct_ct_mult}"
        )
        print("------------------------------------")

    return err_qkt, err_v