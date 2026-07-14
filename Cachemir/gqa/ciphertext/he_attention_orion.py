"""Ciphertext-domain GQA attention using Orion.

This mirrors attention.py.  The client-side input encoding and all weight
encodings remain plaintext torch tensors.  Only X chunks, K/V cache entries,
Q/K/V projections, attention scores, and output are Orion ciphertexts.
"""

from __future__ import annotations
import math
import resource
import time

from typing import Any, List, Tuple
import orion
import torch


from ..plaintext.dims  import GQAConfig, GQADims, make_gqa_dims
from ..plaintext.encoding import (
    head_perm,
    init_input,
    make_sparse_input_kv,
    gqa_lane_offsets,
)

from ..plaintext.reference import (
    compare_attention_outputs,
    decode_attention_map,
    decode_output,
)
from ..plaintext.counter import counter
from .he_cache_orion import HEKCache, HEVCache
from .he_ops_orion import block_replicate, ct_roll, vmm_kv, encode_pt, decrypt_cipher_list, ct_zero
from .he_encoding_orion import make_he_weights_kv, make_he_weights_q_gqa
 

def expand_sparse_input_kv_he(sparse_cts: List[Any],dims: GQADims) -> List[Any]:
    """Ciphertext version of expand_sparse_input_kv_plain."""
    lane_offsets = gqa_lane_offsets(dims)
    chunks: List[Any] = []

    for sparse_ct in sparse_cts:
        for start in range(0, dims.d_kv, dims.t_p):
            offsets = lane_offsets[start : start + dims.t_p]
            acc = None
            for lane, off in enumerate(offsets):

                shift = off * dims.t_p - lane

                if shift == 0:
                    term = sparse_ct
                else:
                    term = sparse_ct.roll(shift)
                    counter.rotations += 1

                acc = term if acc is None else acc + term

            chunks.append(acc)
    return chunks


def decrypt_decode_ct(ct: Any) -> torch.Tensor:
    """Client side: decrypt and decode one Orion ciphertext to a torch tensor.

    Only the real part is kept. For real-only ciphertexts the imaginary
    part is exactly zero; for ciphertexts produced via the complex
    lane-packing trick in softmax_v_gqa_he, the imaginary part is
    discardable cross-term noise (the real part already holds the wanted
    sum -- see softmax_v_gqa_he's docstring).
    """
    v = ct.decrypt().decode()
    if torch.is_complex(v):
        v = v.real
    return torch.as_tensor(v, dtype=torch.float64)


def decrypt_decode_ct_complex(ct: Any) -> torch.Tensor:
    """Like decrypt_decode_ct but preserves the imaginary part.

    Used for att_cts: the imaginary part carries the score for a second,
    complex lane-packed token (see qkt_gqa_he / vmm_kv's pos >= t_p path).
    """
    v = ct.decrypt().decode()
    return torch.as_tensor(v, dtype=torch.complex128)


def decrypt_decode_nested(obj: Any) -> Any:
    """Decrypt/decode nested lists of ciphertexts; useful for verification."""
    if isinstance(obj, list):
        return [decrypt_decode_nested(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(decrypt_decode_nested(x) for x in obj)
    return decrypt_decode_ct(obj)


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
    """Compute encrypted QK^T scores against encrypted cached keys.

    `k_ciphertexts` may be complex lane-packed (2 tokens per ciphertext:
    real part = even token, imaginary part = odd token). Since `Q_rep` is
    always purely real (im=0; Q is never lane-packed), `Q_rep * k_ct` does
    NOT mix the two lanes -- the result's real/imaginary parts hold the
    independent scores for the paired tokens. Each entry of the returned
    `att_cts` rows therefore represents a *pair* of token scores.
    """
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
                folded = folded + ct_roll(folded, -step)
                step *= 2
            row.append(folded)
        att_cts.append(row)

    return att_cts, level-1


def softmax_v_gqa_he(
    att_cts: List[List[Any]],
    vcache: HEVCache,
    dims: GQADims,
    level: int,
    pack_complex: bool = True,
) -> Tuple[List[Any], int]:
    """Compute scores times encrypted V cache.

    TODO: As in your plaintext simulator, this does NOT evaluate Softmax yet.  It uses
    the structural attention scores directly.

    `pack_complex` selects between the complex lane-packed V cache (one
    ct-ct mult against `conjugate(V)` covers 2 tokens) and the original
    real-only behavior (one ct-ct mult against `V` directly covers 1
    token). Both are numerically equivalent when `vcache` was built with
    the matching `pack_complex` setting: for a real-only ciphertext (im=0),
    `conjugate(V) == V`, so the conjugate trick is only needed -- and only
    charged -- when packing is actually in use.
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
        # out = vcache.ciphertexts[0] * 0.0
        out = ct_zero(dims.n_he, out_level)

        for b, att in enumerate(att_cts[c]):
            if b >= len(vcache.ciphertexts):
                break
            counter.ct_ct_mult += 1
            if pack_complex:
                # vcache.ciphertexts[b] is complex lane-packed (2 tokens per
                # ciphertext). Multiplying by its conjugate instead of itself
                # gives, in the real part: re(att)*re(V) + im(att)*im(V), i.e.
                # exactly the sum of both paired tokens' score*V contributions
                # with no cross terms -- one ct-ct mult covers 2 tokens.
                counter.conjugations += 1
                prod = att * vcache.ciphertexts[b].conjugate()
            else:
                prod = att * vcache.ciphertexts[b]
            step = 1
            while step < dims.t_p:
                counter.rotations += 1
                prod = prod + ct_roll(prod, -step)
                step *= 2
            counter.ct_pt_mult += 1
            out = out + prod * out_mask_pt

        O_ct.append(out)

    return O_ct, out_level

def attention_gqa_he(
     X_enc_chunks_ct: List[Any],
    kcache: HEKCache,
    vcache: HEVCache,
    Wq_list: List[List[torch.Tensor]],
    Wk_list: List[torch.Tensor],
    Wv_list: List[torch.Tensor],
    dims: GQADims,
    level: int,
    pack_complex: bool = True,
    ) -> Tuple[List[Any], Any, Any, List[List[Any]], List[Any], int, dict]:
    """Execute one encrypted GQA decoding step.

    Returns the usual outputs plus `phase_stats`: a dict mapping phase name
    -> op-count deltas (rotations/ct_pt_mult/ct_ct_mult/conjugations) for
    just that phase, for performance breakdowns.
    """
    phase_stats: dict = {}
    snap = counter.snapshot()

    Q_cts, q_level = vmm_q_gqa_he(X_enc_chunks_ct, Wq_list, dims, level)
    phase_stats["q_projection"] = counter.delta_since(snap)
    snap = counter.snapshot()
 
    pos_k = len(kcache) % kcache.t
    K_new, k_level = vmm_kv(X_enc_chunks_ct, Wk_list, dims, level, pos=pos_k)
    kcache.append(K_new)
    phase_stats["k_projection"] = counter.delta_since(snap)
    snap = counter.snapshot()
 
    pos_v = len(vcache) % vcache.t
    V_new, v_level = vmm_kv(X_enc_chunks_ct, Wv_list, dims, level, pos=pos_v)
    vcache.append(V_new)
    phase_stats["v_projection"] = counter.delta_since(snap)
    snap = counter.snapshot()

    # Q, K, V all start from the same input `level` and each pass through
    # exactly one vmm_kv call, so they land at the same level.
    assert q_level == k_level == v_level
 
    att_cts, att_level = qkt_gqa_he(Q_cts, kcache.ciphertexts, dims, q_level)
    phase_stats["qkt"] = counter.delta_since(snap)
    snap = counter.snapshot()

    O, out_level = softmax_v_gqa_he(att_cts, vcache, dims, att_level, pack_complex=pack_complex)
    phase_stats["scores_v"] = counter.delta_since(snap)
 
    return Q_cts, K_new, V_new, att_cts, O, out_level, phase_stats


def run_attention_gqa_he(
    config_path: str,
    n_he:int,
    d: int,
    H: int,
    n_kv: int,
    n_prefill: int,
    level: int,
    seeds: Tuple[int, ...] = (1, 2, 3, 99),
    verify: bool = True,
    verbose: bool = True,
    pack_complex: bool = True,
    return_stats: bool = False,
):
    """End-to-end encrypted test driver.

    `pack_complex` toggles the complex lane-packing optimization for the
    K/V cache: True (default) uses the packed layout (2 tokens/ciphertext,
    conjugate trick in scores*V); False reproduces the original real-only
    layout (1 token/ciphertext, plain multiply in scores*V) for direct
    performance/correctness comparison.

    If `return_stats` is True, an extra dict is returned as a 3rd tuple
    element with latency, peak-RSS, ciphertext-count, and per-phase op-count
    breakdowns (see the body for exact keys), plus the decoded dense
    attention map / output (`att_dense`, `O_dense`) so two runs (complex vs
    non-complex) can be compared directly.
    """
    # scheme = orion.init_scheme(config_path)
    # n_he = 1 << (scheme.params.get_logn() - 1)
    if verbose:
        print(f"n_he (from scheme): {n_he}, pack_complex={pack_complex}")
    counter.reset()
    run_t0 = time.perf_counter()
 
    dims = make_gqa_dims(GQAConfig(n_he=n_he, d=d, H=H, n_kv=n_kv, n_prefill=n_prefill))
 
    Wq_raw, Wq_list = make_he_weights_q_gqa(dims, seed=seeds[0], n_he=n_he, level=level)
    Wk_raw, Wk_list = make_he_weights_kv(dims, seed=seeds[1], n_he=n_he, level=level)
    Wv_raw, Wv_list = make_he_weights_kv(dims, seed=seeds[2], n_he=n_he, level=level)
 
    kcache = HEKCache(n_he, dims.d_kv, pack_complex=pack_complex)
    vcache = HEVCache(n_he, dims.d_kv, H=dims.n_kv, pack_complex=pack_complex)
 
    def encrypt_sparse(x: torch.Tensor) -> List[Any]:
        sparse_plain = make_sparse_input_kv(x, dims)
        return [orion.encrypt(encode_pt(v, level)) for v in sparse_plain]
 
    toks = [init_input(dims.d, seed=10 + i) for i in range(n_prefill)]

    prefill_snap = counter.snapshot()
    prefill_t0 = time.perf_counter()
    for x in toks:
        x_sparse_ct = encrypt_sparse(x)
        xc_ct = expand_sparse_input_kv_he(x_sparse_ct, dims)
 
        k_new, _ = vmm_kv(xc_ct, Wk_list, dims, level, pos=len(kcache) % kcache.t)
        kcache.append(k_new)
 
        v_new, _ = vmm_kv(xc_ct, Wv_list, dims, level, pos=len(vcache) % vcache.t)
        vcache.append(v_new)
    prefill_s = time.perf_counter() - prefill_t0
    prefill_ops = counter.delta_since(prefill_snap)
 
    x_new = init_input(dims.d, seed=seeds[3])
    x_sparse_ct = encrypt_sparse(x_new)
    xc_new_ct = expand_sparse_input_kv_he(x_sparse_ct, dims)
 
    decode_t0 = time.perf_counter()
    Q_cts, K_new, V_new, att_cts, O_cts, out_level, phase_stats = attention_gqa_he(
        xc_new_ct, kcache, vcache, Wq_list, Wk_list, Wv_list, dims, level,
        pack_complex=pack_complex,
    )
    decode_s = time.perf_counter() - decode_t0
    total_s = time.perf_counter() - run_t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    n_att_cts = dims.ratio * len(att_cts[0]) if att_cts and att_cts[0] else 0
    ciphertext_counts = {
        "kcache": kcache.num_ciphertexts,
        "vcache": vcache.num_ciphertexts,
        "att_cts": n_att_cts,
        "o_cts": len(O_cts),
    }
    ops_by_phase = {"prefill": prefill_ops, **phase_stats}

    def _print_perf() -> None:
        print(
            f"[pack_complex={pack_complex}] latency(s): prefill={prefill_s:.4f} "
            f"decode_step={decode_s:.4f} total={total_s:.4f}   "
            f"peak_rss={peak_rss_mb:.1f} MB (process high-water mark)"
        )
        print(
            f"[pack_complex={pack_complex}] ciphertext counts: "
            f"kcache={ciphertext_counts['kcache']} vcache={ciphertext_counts['vcache']} "
            f"att_cts={ciphertext_counts['att_cts']} O_cts={ciphertext_counts['o_cts']}"
        )
        for phase, ops in ops_by_phase.items():
            print(
                f"[pack_complex={pack_complex}]   phase={phase:<12s} "
                f"ct-ct={ops['ct_ct_mult']:<4d} ct-pt={ops['ct_pt_mult']:<5d} "
                f"rotations={ops['rotations']:<5d} conjugations={ops['conjugations']}"
            )

    if not verify:
        if verbose:
            print(
                f"encrypted run done (level {level} -> {out_level}): "
                f"rotations={counter.rotations}, ct-pt={counter.ct_pt_mult}, "
                f"ct-ct={counter.ct_ct_mult}, conjugations={counter.conjugations}"
            )
            _print_perf()
        if return_stats:
            stats = {
                "pack_complex": pack_complex,
                "latency_s": {"prefill": prefill_s, "decode_step": decode_s, "total": total_s},
                "peak_rss_mb": peak_rss_mb,
                "ciphertext_counts": ciphertext_counts,
                "ops_by_phase": ops_by_phase,
                "ops_total": counter.snapshot(),
            }
            return float("nan"), float("nan"), stats
        return float("nan"), float("nan")
 
    # Decrypt only for local correctness checking. att_cts entries are
    # complex lane-packed (2 tokens per cache ciphertext), so decode them
    # preserving the imaginary part; decode_attention_map() knows how to
    # split re/im back into the two paired tokens.
    att_torch = torch.stack(
        [torch.stack([decrypt_decode_ct_complex(ct) for ct in row]) for row in att_cts]
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
        pack_complex=pack_complex,
    )
 
    if verbose:
        ok = max(err_qkt, err_v) < 1e-3 #5e-3
        print(
            f"n_he={dims.n_he} d={dims.d} H={dims.H} n_kv={dims.n_kv} "
            f"(d_h={dims.d_h}, ratio={dims.ratio}, d_kv={dims.d_kv}, "
            f"t_p={dims.t_p}, B={dims.B}, R={dims.R}) "
            f"n_prefill={n_prefill} (n={len(kcache)}): "
            f"qkt_err={err_qkt:.2e}, softmaxv_err={err_v:.2e} -> "
            f"{'PASS' if ok else 'CHECK'}"
        )
        print(
            f"level: {level} -> {out_level}   "
            f"ops: rotations={counter.rotations}, "
            f"ct-pt={counter.ct_pt_mult}, ct-ct={counter.ct_ct_mult}, "
            f"conjugations={counter.conjugations}"
        )
        _print_perf()

    if return_stats:
        n_tokens = len(toks) + 1
        att_dense = decode_attention_map(att_torch.numpy(), n_tokens, dims, pack_complex=pack_complex)
        O_dense = decode_output(O_torch.numpy(), dims)
        stats = {
            "pack_complex": pack_complex,
            "latency_s": {"prefill": prefill_s, "decode_step": decode_s, "total": total_s},
            "peak_rss_mb": peak_rss_mb,
            "ciphertext_counts": ciphertext_counts,
            "ops_by_phase": ops_by_phase,
            "ops_total": counter.snapshot(),
            "att_dense": att_dense,
            "O_dense": O_dense,
        }
        return err_qkt, err_v, stats

    return err_qkt, err_v
 