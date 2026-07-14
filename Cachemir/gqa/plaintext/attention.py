"""One-file GQA attention pipeline."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from .cache import KCache, VCache
from .dims import GQAConfig, GQADims, make_gqa_dims
from .encoding import (
    head_perm,
    init_input,
    make_weights_kv,
    make_weights_q_gqa,
    expand_sparse_input_kv_plain,
    make_sparse_input_kv,
)
from .ops import block_replicate, vmm_kv
from .reference import compare_attention_outputs
from .counter import counter


def vmm_q_gqa(
    X_enc_chunks: List[torch.Tensor], Wq_enc_list: List[List[torch.Tensor]], dims: GQADims
) -> torch.Tensor:
    """Compute Q ciphertexts in sparse canonical layout."""
    Q_cts = torch.zeros((dims.ratio, dims.n_he), dtype=torch.float64)
    for c in range(dims.ratio):
        Q_cts[c] = vmm_kv(X_enc_chunks, Wq_enc_list[c], dims, pos=0)
    return Q_cts


def qkt_gqa(Q_cts: torch.Tensor, k_ciphertexts: List[torch.Tensor], dims: GQADims) -> torch.Tensor:
    """Compute structural QK^T scores against cached keys."""
    fold_nsteps = int(math.log2(dims.d_h))
    att_cts = torch.zeros((dims.ratio, len(k_ciphertexts), dims.n_he), dtype=torch.float64)

    for c in range(dims.ratio):
        Q_rep = block_replicate(Q_cts[c], dims.t_p)
        for b, k_ct in enumerate(k_ciphertexts):
            counter.ct_ct_mult += 1
            folded = Q_rep * k_ct
            step = dims.B
            for _ in range(fold_nsteps):
                counter.rotations += 1
                folded = folded + torch.roll(folded, -step)
                step *= 2
            att_cts[c, b] = folded

    return att_cts


def softmax_v_gqa(att_cts: torch.Tensor, vcache: VCache, dims: GQADims) -> torch.Tensor:
    """Compute scores times V. Softmax approximation is intentionally omitted."""
    out_mask = torch.zeros(dims.n_he, dtype=torch.float64)
    out_mask[:: dims.t_p] = 1.0
    O_ct = torch.zeros((dims.ratio, dims.n_he), dtype=torch.float64)

    for c in range(dims.ratio):
        for b in range(att_cts.shape[1]):
            if b >= len(vcache.ciphertexts):
                break
            counter.ct_ct_mult += 1
            prod = att_cts[c, b] * vcache.ciphertexts[b]
            step = 1
            while step < dims.t_p:
                counter.rotations += 1
                prod = prod + torch.roll(prod, -step)
                step *= 2
            O_ct[c] += prod * out_mask

    return O_ct



def softmax_gqa(att_cts: torch.Tensor, dims: GQADims, n_tokens: int) -> torch.Tensor:
    """Row-softmax over the token axis, independently per (query-group c, kv-head)."""
    ratio, n_b, n_he = att_cts.shape
    t_p, B, n_kv = dims.t_p, dims.B, dims.n_kv
    out = torch.zeros_like(att_cts)

    # valid tokens in the last ciphertext
    rem = n_tokens - (n_b - 1) * t_p

    # valid mask for the last ciphertext for a single block - needs to be replicated in the ciphertext (done by tile)
    base_valid = torch.zeros(B, dtype=torch.float64)
    for kv in range(n_kv):
        base_valid[kv * t_p: kv * t_p + rem] = 1.0
    valid_last = base_valid.tile(n_he // B)

    # mask keeping exactly one base slot per lane (for isolating summation fold results)
    base_reduce = torch.zeros(B, dtype=torch.float64)
    base_reduce[::t_p] = 1.0
    reduce_mask = base_reduce.tile(n_he // B)

    for c in range(ratio):
        # per-lane max, computed for ALL kv at once via reshape (no kv loop) - this is not supported in FHE (needes to be changed)
        last_block = att_cts[c, n_b - 1][:B].reshape(n_kv, t_p)
        last_valid = base_valid.reshape(n_kv, t_p) > 0
        lane_max = torch.stack(
            [att_cts[c, b][:B].reshape(n_kv, t_p).max(dim=1).values for b in range(n_b - 1)]
            + [torch.where(
                last_valid,
                last_block,
                torch.full_like(last_block, -float("inf")),
              ).max(dim=1).values]
        ).max(dim=0).values  # shape (n_kv,)

        # broadcast each lane's max to its own t_p slots, tiled over d_h blocks
        row_max = lane_max.repeat_interleave(t_p).tile(n_he // B)

        # mask the padding slots of the last block to 0 BEFORE exp: raw
        # attention scores are unscaled dot products and can be large in
        # magnitude, so exp(padding_value - row_max) can overflow to inf if
        # left unmasked; forcing the diff to exactly 0 keeps exp bounded
        # (exp(0) == 1). Masking again AFTER exp zeroes those 1s back out so
        # they don't pollute the denominator sum below.
        exp_cts = []
        for b in range(n_b):
            diff = att_cts[c, b] - row_max
            if b == n_b - 1:
                diff = diff * valid_last
            e = torch.exp(diff)
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
        inv_denom = torch.where(denom > 0, 1.0 / denom, torch.zeros_like(denom))

        for b in range(n_b):
            out[c, b] = exp_cts[b] * inv_denom

    return out


def attention_gqa(
    X_enc_chunks: List[torch.Tensor],
    kcache: KCache,
    vcache: VCache,
    Wq_enc_list: List[List[torch.Tensor]],
    Wk_enc: List[torch.Tensor],
    Wv_enc: List[torch.Tensor],
    dims: GQADims,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute one GQA decoding step."""
    Q_cts = vmm_q_gqa(X_enc_chunks, Wq_enc_list, dims)

    pos_k = len(kcache) % dims.t_p
    K_new = vmm_kv(X_enc_chunks, Wk_enc, dims, pos=pos_k)
    kcache.append(K_new)

    pos_v = len(vcache) % dims.t_p
    V_new = vmm_kv(X_enc_chunks, Wv_enc, dims, pos=pos_v)
    vcache.append(V_new)

    att_cts = qkt_gqa(Q_cts, kcache.ciphertexts, dims)
    smax_cts = softmax_gqa(att_cts, dims, n_tokens=len(kcache))
    O = softmax_v_gqa(smax_cts, vcache, dims)
    return Q_cts, K_new, V_new, att_cts, smax_cts, O


def run_attention_gqa(
    n_he: int,
    d: int,
    H: int,
    n_kv: int,
    n_prefill: int,
    seeds: Tuple[int, ...] = (1, 2, 3, 99),
    verbose: bool = True,
) -> tuple[float, float]:
    """Run the compact GQA simulation and compare to dense reference math."""
    dims = make_gqa_dims(GQAConfig(n_he=n_he, d=d, H=H, n_kv=n_kv, n_prefill=n_prefill))

    Wq_raw, Wq_enc_list = make_weights_q_gqa(dims, seed=seeds[0])
    Wk_raw, Wk_enc = make_weights_kv(dims, seed=seeds[1])
    Wv_raw, Wv_enc = make_weights_kv(dims, seed=seeds[2])

    kcache = KCache(dims.n_he, dims.d_kv)
    vcache = VCache(dims.n_he, dims.d_kv, H=dims.n_kv)

    toks = [init_input(dims.d, seed=10 + i) for i in range(n_prefill)]
    for x in toks:
        x_sparse = make_sparse_input_kv(x, dims)
        xc = expand_sparse_input_kv_plain(x_sparse, dims)
        kcache.append(vmm_kv(xc, Wk_enc, dims, pos=len(kcache) % dims.t_p))
        vcache.append(vmm_kv(xc, Wv_enc, dims, pos=len(vcache) % dims.t_p))

    x_new = init_input(dims.d, seed=seeds[3])
    # print("x_new" , x_new)
    x_sparse = make_sparse_input_kv(x_new, dims)
    # print("len sparse:" , x_sparse)
    xc_new = expand_sparse_input_kv_plain(x_sparse, dims)
    # print("len expanded:" , xc_new)
    # print("input without encodong:" , x_sparse)

    Q_cts, K_new, V_new, att_cts, smax_cts, O = attention_gqa(
        xc_new, kcache, vcache, Wq_enc_list, Wk_enc, Wv_enc, dims
    )
    # print("output :" , len(O))

    err_qkt, err_v = compare_attention_outputs(
        toks=toks,
        x_new=x_new,
        Wq_raw=Wq_raw,
        Wk_raw=Wk_raw,
        Wv_raw=Wv_raw,
        att_cts=att_cts,
        O=O,
        dims=dims,
        head_perm=head_perm,
    )

    if verbose:
        ok = max(err_qkt, err_v) < 1e-9
        print(
            f"N={dims.n_he:3d} d={dims.d:2d} H={dims.H} n_kv={dims.n_kv} "
            f"(d_h={dims.d_h}, ratio={dims.ratio}, d_kv={dims.d_kv}, "
            f"t_p={dims.t_p}, B={dims.B}, R={dims.R}) "
            f"n_prefill={n_prefill} (n={len(kcache)}): "
            f"qkt_err={err_qkt:.2e}, softmaxv_err={err_v:.2e} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )

    return err_qkt, err_v