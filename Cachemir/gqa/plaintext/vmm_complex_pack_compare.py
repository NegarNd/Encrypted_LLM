"""Compare complex-lane-packing strategies for the compact GQA decode step.

This module builds THREE variants of one decode step (prefill `n_prefill`
tokens into the K/V cache, then process one new token) and tallies
rotations / ct-pt / ct-ct multiplications / conjugations via the shared
`counter` (the same op-counting convention used by verify_gqa.py). No FHE
library is involved -- this mirrors how attention.py/ops.py already
simulate cost with plain torch tensors; here `torch.complex128` tensors
stand in for a complex-lane-packed ciphertext (real+imaginary = 2 packed
real values per slot).

Strategies for one decode step:
  - "baseline":     no complex packing anywhere (same math as attention.py,
                    minus the exact softmax -- see note below).
  - "attn_complex": complex-pack 2 TOKENS per ciphertext in the K-cache and
                    V-cache (mirrors gqa/ciphertext/he_attention_orion.py's
                    `pack_complex` trick for QK^T and scores*V). The QKV
                    projection (VMM) stage stays real/unpacked.
  - "vmm_complex":  complex-pack the K/V *weight* projection, and pairs of
                    Q projections (when dims.ratio > 1), so a single ct-pt
                    multiply produces two independent real outputs at once
                    (X is real, so packing two plaintexts as W_a + i*W_b and
                    multiplying once is exact -- no cross-term risk, unlike
                    packing two ciphertexts). QK^T / scores*V stay real.

Note on softmax: none of the three strategies evaluates softmax. This repo's
own ciphertext pipeline (he_attention_orion.py) does not evaluate softmax
yet either (see its docstring TODO) -- all three strategies compute
structural QK^T scores times V. This keeps the comparison isolated to
"where is complex packing applied", independent of the (still unresolved)
question of how a nonlinear softmax interacts with complex-packed lanes.

This repo does not implement an FFN/SwiGLU block, so "vmm_complex" only
covers the QKV projection VMMs that actually exist here; FFN packing is
noted but not benchmarked.

Run: `cd Cachemir && python -m gqa.plaintext.vmm_complex_pack_compare`
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from .dims import GQAConfig, GQADims, make_gqa_dims
from .encoding import (
    init_input,
    make_sparse_input_kv,
    expand_sparse_input_kv_plain,
    make_weights_kv,
    make_weights_q_gqa,
)
from .cache import KCache, VCache
from .ops import vmm_kv, block_replicate
from .attention import vmm_q_gqa, qkt_gqa, softmax_v_gqa
from .reference import decode_output
from .counter import counter
from .verify_gqa import CASES

# Tracks conjugations separately since the shared `Counter` dataclass
# (counter.py) does not have that field -- adding it there would change
# behavior for every other module that imports the singleton.
EXTRA = {"conjugations": 0}

# Illustrative per-op latencies (microseconds), loosely representative of
# the GPU RNS-CKKS microbenchmarks reported in Cachemir's Table 9 (arXiv
# 2602.11470). These are NOT measured on this machine -- they only give a
# rough, consistent way to convert op counts into a comparable latency
# number. Conjugation is a Galois automorphism like rotation, so it is
# priced the same as a rotation.
LATENCY_US = {
    "ct_pt_mult": 140.0,
    "ct_ct_mult": 580.0,
    "rotations": 480.0,
    "conjugations": 480.0,
}


def estimated_latency_us(tally: Dict[str, int]) -> float:
    return sum(tally[k] * LATENCY_US[k] for k in LATENCY_US)


def _read_counter_and_reset() -> Dict[str, int]:
    tally = {
        "rotations": counter.rotations,
        "ct_pt_mult": counter.ct_pt_mult,
        "ct_ct_mult": counter.ct_ct_mult,
        "conjugations": EXTRA["conjugations"],
    }
    counter.reset()
    EXTRA["conjugations"] = 0
    return tally


# --------------------------------------------------------------------------
# Strategy A: baseline (no complex packing anywhere).
# --------------------------------------------------------------------------
def run_baseline(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new):
    kcache = KCache(dims.n_he, dims.d_kv)
    vcache = VCache(dims.n_he, dims.d_kv, H=dims.n_kv)

    for x in toks:
        xc = expand_sparse_input_kv_plain(make_sparse_input_kv(x, dims), dims)
        kcache.append(vmm_kv(xc, Wk_enc, dims, pos=len(kcache) % dims.t_p))
        vcache.append(vmm_kv(xc, Wv_enc, dims, pos=len(vcache) % dims.t_p))

    xc_new = expand_sparse_input_kv_plain(make_sparse_input_kv(x_new, dims), dims)
    Q_cts = vmm_q_gqa(xc_new, Wq_enc_list, dims)
    att_cts = qkt_gqa(Q_cts, kcache.ciphertexts, dims)
    O = softmax_v_gqa(att_cts, vcache, dims)
    return O


# --------------------------------------------------------------------------
# Strategy B: complex-pack 2 tokens/ciphertext in K-cache and V-cache
# (mirrors he_attention_orion.py's pack_complex trick for QK^T / scores*V).
# --------------------------------------------------------------------------
def _complex_cache_append(
    cache_list: List[torch.Tensor], xc, W_enc, dims: GQADims, tok_index: int
) -> None:
    t_p = dims.t_p
    pos = tok_index % t_p
    is_imag = (tok_index % (2 * t_p)) >= t_p

    real_vec = vmm_kv(xc, W_enc, dims, pos=pos)

    if tok_index % (2 * t_p) == 0:
        cache_list.append(torch.zeros(dims.n_he, dtype=torch.complex128))

    if is_imag:
        # Encoding a real contribution into the imaginary lane costs one
        # ct-pt scalar multiply (x * i) that the real-only baseline never pays.
        counter.ct_pt_mult += 1
        cache_list[-1] = cache_list[-1] + real_vec.to(torch.complex128) * 1j
    else:
        cache_list[-1] = cache_list[-1] + real_vec.to(torch.complex128)


def qkt_gqa_complex(Q_cts, k_ciphertexts: List[torch.Tensor], dims: GQADims):
    fold_nsteps = int(math.log2(dims.d_h))
    att_cts: List[List[torch.Tensor]] = []

    for c in range(dims.ratio):
        Q_rep = block_replicate(Q_cts[c], dims.t_p)  # real (Q is never lane-packed)
        row: List[torch.Tensor] = []
        for k_ct in k_ciphertexts:  # complex; HALF as many entries as the real cache
            counter.ct_ct_mult += 1
            folded = Q_rep.to(torch.complex128) * k_ct
            step = dims.B
            for _ in range(fold_nsteps):
                counter.rotations += 1
                folded = folded + torch.roll(folded, -step)
                step *= 2
            row.append(folded)
        att_cts.append(row)

    return att_cts


def scores_v_complex(att_cts, v_ciphertexts: List[torch.Tensor], dims: GQADims):
    out_mask = torch.zeros(dims.n_he, dtype=torch.float64)
    out_mask[:: dims.t_p] = 1.0
    O = torch.zeros((dims.ratio, dims.n_he), dtype=torch.complex128)

    for c in range(dims.ratio):
        for b, att in enumerate(att_cts[c]):
            if b >= len(v_ciphertexts):
                break
            counter.ct_ct_mult += 1
            EXTRA["conjugations"] += 1
            prod = att * v_ciphertexts[b].conj()
            step = 1
            while step < dims.t_p:
                counter.rotations += 1
                prod = prod + torch.roll(prod, -step)
                step *= 2
            O[c] = O[c] + prod * out_mask

    return O  # complex; .real holds the correct paired sum (see he_attention_orion.py)


def run_attn_complex(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new):
    k_cache_c: List[torch.Tensor] = []
    v_cache_c: List[torch.Tensor] = []

    for i, x in enumerate(toks):
        xc = expand_sparse_input_kv_plain(make_sparse_input_kv(x, dims), dims)
        _complex_cache_append(k_cache_c, xc, Wk_enc, dims, i)
        _complex_cache_append(v_cache_c, xc, Wv_enc, dims, i)

    xc_new = expand_sparse_input_kv_plain(make_sparse_input_kv(x_new, dims), dims)
    Q_cts = vmm_q_gqa(xc_new, Wq_enc_list, dims)  # real, unpaired (VMM stays unpacked)
    att_cts = qkt_gqa_complex(Q_cts, k_cache_c, dims)
    O = scores_v_complex(att_cts, v_cache_c, dims)
    return O.real


# --------------------------------------------------------------------------
# Strategy C: complex-pack the VMM (QKV projection) stage. K & V share the
# same real input X, so W_k + i*W_v can be multiplied once (X real => no
# cross-term risk). Q's `ratio` copies are paired the same way when
# dims.ratio > 1. QK^T / scores*V stay real (real cache, no token packing).
# --------------------------------------------------------------------------
def vmm_kv_pair_complex(Xc, Wa_enc, Wb_enc, dims: GQADims, pos: int = 0):
    out = torch.zeros(dims.n_he, dtype=torch.complex128)
    mask = torch.zeros(dims.n_he, dtype=torch.float64)
    mask[pos :: dims.t_p] = 1.0

    for xchunk, wa, wb in zip(Xc, Wa_enc, Wb_enc):
        w_pair = wa.to(torch.complex128) + 1j * wb.to(torch.complex128)
        acc = xchunk.to(torch.complex128) * w_pair
        counter.ct_pt_mult += 1  # ONE ct-pt mult instead of two

        step, i = 1, 0
        while step < dims.t_p:
            counter.rotations += 1  # ONE shared fold instead of two
            acc = acc + torch.roll(acc, +step if (pos >> i) & 1 else -step)
            step *= 2
            i += 1

        counter.ct_pt_mult += 1  # ONE mask mult instead of two
        out = out + acc * mask

    return out  # .real -> A's result, .imag -> B's result


def run_vmm_complex(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new):
    kcache = KCache(dims.n_he, dims.d_kv)
    vcache = VCache(dims.n_he, dims.d_kv, H=dims.n_kv)

    for x in toks:
        xc = expand_sparse_input_kv_plain(make_sparse_input_kv(x, dims), dims)
        pos = len(kcache) % dims.t_p
        kv_pair = vmm_kv_pair_complex(xc, Wk_enc, Wv_enc, dims, pos=pos)
        kcache.append(kv_pair.real)
        vcache.append(kv_pair.imag)

    xc_new = expand_sparse_input_kv_plain(make_sparse_input_kv(x_new, dims), dims)
    Q_cts: List[torch.Tensor] = [None] * dims.ratio  # type: ignore[list-item]
    c = 0
    while c < dims.ratio:
        if c + 1 < dims.ratio:
            q_pair = vmm_kv_pair_complex(xc_new, Wq_enc_list[c], Wq_enc_list[c + 1], dims, pos=0)
            Q_cts[c] = q_pair.real
            Q_cts[c + 1] = q_pair.imag
            c += 2
        else:
            Q_cts[c] = vmm_kv(xc_new, Wq_enc_list[c], dims, pos=0)  # unpaired leftover
            c += 1

    att_cts = qkt_gqa(Q_cts, kcache.ciphertexts, dims)
    O = softmax_v_gqa(att_cts, vcache, dims)
    return O


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _build_case(N, d, H, n_kv, n_prefill, seeds=(1, 2, 3, 99)):
    dims = make_gqa_dims(GQAConfig(n_he=N, d=d, H=H, n_kv=n_kv, n_prefill=n_prefill))
    _, Wq_enc_list = make_weights_q_gqa(dims, seed=seeds[0])
    _, Wk_enc = make_weights_kv(dims, seed=seeds[1])
    _, Wv_enc = make_weights_kv(dims, seed=seeds[2])
    toks = [init_input(dims.d, seed=10 + i) for i in range(n_prefill)]
    x_new = init_input(dims.d, seed=seeds[3])
    return dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new


def _fmt_tally(name: str, tally: Dict[str, int]) -> str:
    lat = estimated_latency_us(tally)
    return (
        f"{name:14s} rot={tally['rotations']:5d} ct-pt={tally['ct_pt_mult']:5d} "
        f"ct-ct={tally['ct_ct_mult']:5d} conj={tally['conjugations']:4d} "
        f"est_latency={lat/1000:8.2f} ms"
    )


def main() -> None:
    totals = {"baseline": [], "attn_complex": [], "vmm_complex": []}

    for N, d, H, n_kv, n_prefill in CASES:
        dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new = _build_case(N, d, H, n_kv, n_prefill)

        counter.reset()
        EXTRA["conjugations"] = 0
        O_base = run_baseline(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new)
        tally_base = _read_counter_and_reset()

        O_attn = run_attn_complex(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new)
        tally_attn = _read_counter_and_reset()

        O_vmm = run_vmm_complex(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new)
        tally_vmm = _read_counter_and_reset()

        # Correctness: all three must decode to the same dense output.
        dense_base = decode_output(O_base, dims)
        dense_attn = decode_output(O_attn, dims)
        dense_vmm = decode_output(O_vmm, dims)
        err_attn = float(torch.max(torch.abs(dense_base - dense_attn)))
        err_vmm = float(torch.max(torch.abs(dense_base - dense_vmm)))
        ok = err_attn < 1e-9 and err_vmm < 1e-9

        print(
            f"\nN={dims.n_he:4d} d={dims.d:3d} H={dims.H} n_kv={dims.n_kv} "
            f"ratio={dims.ratio} t_p={dims.t_p} n_prefill={n_prefill} "
            f"(err_attn={err_attn:.1e}, err_vmm={err_vmm:.1e}) -> "
            f"{'PASS' if ok else 'FAIL'}"
        )
        print(f"  {_fmt_tally('baseline', tally_base)}")
        print(f"  {_fmt_tally('attn_complex', tally_attn)}")
        print(f"  {_fmt_tally('vmm_complex', tally_vmm)}")

        assert ok, f"mismatch for case N={N} d={d} H={H} n_kv={n_kv} n_prefill={n_prefill}"

        totals["baseline"].append(tally_base)
        totals["attn_complex"].append(tally_attn)
        totals["vmm_complex"].append(tally_vmm)

    print("\n=== Summary across all cases (sum of op counts / estimated latency) ===")
    summed = {}
    for strat, tallies in totals.items():
        summed[strat] = {
            k: sum(t[k] for t in tallies) for k in ("rotations", "ct_pt_mult", "ct_ct_mult", "conjugations")
        }
    base_lat = estimated_latency_us(summed["baseline"])
    for strat in ("baseline", "attn_complex", "vmm_complex"):
        lat = estimated_latency_us(summed[strat])
        pct = 100.0 * (1.0 - lat / base_lat) if strat != "baseline" else 0.0
        print(f"  {_fmt_tally(strat, summed[strat])}  ({pct:+.1f}% vs baseline)")


if __name__ == "__main__":
    main()
