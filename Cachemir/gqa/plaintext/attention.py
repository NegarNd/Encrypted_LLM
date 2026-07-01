"""One-file GQA attention pipeline."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

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
    X_enc_chunks: List[np.ndarray], Wq_enc_list: List[List[np.ndarray]], dims: GQADims
) -> np.ndarray:
    """Compute Q ciphertexts in sparse canonical layout."""
    Q_cts = np.zeros((dims.ratio, dims.n_he), dtype=np.float64)
    for c in range(dims.ratio):
        Q_cts[c] = vmm_kv(X_enc_chunks, Wq_enc_list[c], dims, pos=0)
    return Q_cts


def qkt_gqa(Q_cts: np.ndarray, k_ciphertexts: List[np.ndarray], dims: GQADims) -> np.ndarray:
    """Compute structural QK^T scores against cached keys."""
    fold_nsteps = int(np.log2(dims.d_h))
    att_cts = np.zeros((dims.ratio, len(k_ciphertexts), dims.n_he), dtype=np.float64)

    for c in range(dims.ratio):
        Q_rep = block_replicate(Q_cts[c], dims.t_p)
        for b, k_ct in enumerate(k_ciphertexts):
            counter.ct_ct_mult += 1
            folded = Q_rep * k_ct
            step = dims.B
            for _ in range(fold_nsteps):
                counter.rotations += 1
                folded = folded + np.roll(folded, -step)
                step *= 2
            att_cts[c, b] = folded

    return att_cts


def softmax_v_gqa(att_cts: np.ndarray, vcache: VCache, dims: GQADims) -> np.ndarray:
    """Compute scores times V. Softmax approximation is intentionally omitted."""
    out_mask = np.zeros(dims.n_he, dtype=np.float64)
    out_mask[:: dims.t_p] = 1.0
    O_ct = np.zeros((dims.ratio, dims.n_he), dtype=np.float64)

    for c in range(dims.ratio):
        for b in range(att_cts.shape[1]):
            if b >= len(vcache.ciphertexts):
                break
            counter.ct_ct_mult += 1
            prod = att_cts[c, b] * vcache.ciphertexts[b]
            step = 1
            while step < dims.t_p:
                counter.rotations += 1
                prod = prod + np.roll(prod, -step)
                step *= 2
            O_ct[c] += prod * out_mask

    return O_ct


def attention_gqa(
    X_enc_chunks: List[np.ndarray],
    kcache: KCache,
    vcache: VCache,
    Wq_enc_list: List[List[np.ndarray]],
    Wk_enc: List[np.ndarray],
    Wv_enc: List[np.ndarray],
    dims: GQADims,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute one GQA decoding step."""
    Q_cts = vmm_q_gqa(X_enc_chunks, Wq_enc_list, dims)

    pos_k = len(kcache) % dims.t_p
    K_new = vmm_kv(X_enc_chunks, Wk_enc, dims, pos=pos_k)
    kcache.append(K_new)

    pos_v = len(vcache) % dims.t_p
    V_new = vmm_kv(X_enc_chunks, Wv_enc, dims, pos=pos_v)
    vcache.append(V_new)

    att_cts = qkt_gqa(Q_cts, kcache.ciphertexts, dims)
    O = softmax_v_gqa(att_cts, vcache, dims)
    return Q_cts, K_new, V_new, att_cts, O


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
    x_sparse = make_sparse_input_kv(x_new, dims)
    xc_new = expand_sparse_input_kv_plain(x_sparse, dims)

    Q_cts, K_new, V_new, att_cts, O = attention_gqa(
        xc_new, kcache, vcache, Wq_enc_list, Wk_enc, Wv_enc, dims
    )

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