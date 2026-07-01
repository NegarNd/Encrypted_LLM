"""Ciphertext-domain GQA attention using Orion.

This mirrors attention.py.  The client-side input encoding and all weight
encodings remain plaintext torch tensors.  Only X chunks, K/V cache entries,
Q/K/V projections, attention scores, and output are Orion ciphertexts.
"""

from __future__ import annotations
import math

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

from ..plaintext.reference import compare_attention_outputs
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
    """Client side: decrypt and decode one Orion ciphertext to a torch tensor."""
    return torch.as_tensor(ct.decrypt().decode(), dtype=torch.float64)


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
) -> Tuple[List[Any], int]:
    """Compute scores times encrypted V cache.

    TODO: As in your plaintext simulator, this does NOT evaluate Softmax yet.  It uses
    the structural attention scores directly.
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
    ) -> Tuple[List[Any], Any, Any, List[List[Any]], List[Any], int]:
    """Execute one encrypted GQA decoding step."""
    Q_cts, q_level = vmm_q_gqa_he(X_enc_chunks_ct, Wq_list, dims, level)
 
    pos_k = len(kcache) % dims.t_p
    K_new, k_level = vmm_kv(X_enc_chunks_ct, Wk_list, dims, level, pos=pos_k)
    kcache.append(K_new)
 
    pos_v = len(vcache) % dims.t_p
    V_new, v_level = vmm_kv(X_enc_chunks_ct, Wv_list, dims, level, pos=pos_v)
    vcache.append(V_new)

    # Q, K, V all start from the same input `level` and each pass through
    # exactly one vmm_kv call, so they land at the same level.
    assert q_level == k_level == v_level
 
    att_cts, att_level = qkt_gqa_he(Q_cts, kcache.ciphertexts, dims, q_level)
    O, out_level = softmax_v_gqa_he(att_cts, vcache, dims, att_level)
 
    return Q_cts, K_new, V_new, att_cts, O, out_level


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
) -> Tuple[float, float]:
    """End-to-end encrypted test driver."""
    # scheme = orion.init_scheme(config_path)
    # n_he = 1 << (scheme.params.get_logn() - 1)
    if verbose:
        print(f"n_he (from scheme): {n_he}")
    counter.reset()
 
    dims = make_gqa_dims(GQAConfig(n_he=n_he, d=d, H=H, n_kv=n_kv, n_prefill=n_prefill))
 
    Wq_raw, Wq_list = make_he_weights_q_gqa(dims, seed=seeds[0], n_he=n_he, level=level)
    Wk_raw, Wk_list = make_he_weights_kv(dims, seed=seeds[1], n_he=n_he, level=level)
    Wv_raw, Wv_list = make_he_weights_kv(dims, seed=seeds[2], n_he=n_he, level=level)
 
    kcache = HEKCache(n_he, dims.d_kv)
    vcache = HEVCache(n_he, dims.d_kv, H=dims.n_kv)
 
    def encrypt_sparse(x: torch.Tensor) -> List[Any]:
        sparse_plain = make_sparse_input_kv(x, dims)
        return [orion.encrypt(encode_pt(v, level)) for v in sparse_plain]
 
    toks = [init_input(dims.d, seed=10 + i) for i in range(n_prefill)]
    for x in toks:
        x_sparse_ct = encrypt_sparse(x)
        xc_ct = expand_sparse_input_kv_he(x_sparse_ct, dims)
 
        k_new, _ = vmm_kv(xc_ct, Wk_list, dims, level, pos=len(kcache) % dims.t_p)
        kcache.append(k_new)
 
        v_new, _ = vmm_kv(xc_ct, Wv_list, dims, level, pos=len(vcache) % dims.t_p)
        vcache.append(v_new)
 
    x_new = init_input(dims.d, seed=seeds[3])
    x_sparse_ct = encrypt_sparse(x_new)
    xc_new_ct = expand_sparse_input_kv_he(x_sparse_ct, dims)
 
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
            f"ct-pt={counter.ct_pt_mult}, ct-ct={counter.ct_ct_mult}"
        )
 
    return err_qkt, err_v
 