"""
cachemir_attention_gqa.py
=========================
GQA Attention mechanism logic and numerical precision verification suites.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from cachemir_attention import (
    check_dims, head_perm, init_input, init_weights, make_weights_kv,
    normalize_heads, preprocess_input_kv, vmm_kv
)
from k_cache import KCache
from v_cache import VCache


def gqa_dims(N: int, d: int, H: int, n_kv: int) -> Dict[str, int]:
    """Calculate and validate dimensions for GQA computation blocks."""
    check_dims(N, d)
    H = normalize_heads(H)
    if H % n_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by n_kv ({n_kv}).")
    
    d_h = d // H
    ratio = H // n_kv
    d_kv = n_kv * d_h
    t_p = (N // d) * ratio
    B = n_kv * t_p
    
    if N != d_h * B:
        raise ValueError(f"N ({N}) must equal d_h*B = {d_h * B}.")
    
    return dict(H=H, d_h=d_h, ratio=ratio, d_kv=d_kv, t_p=t_p, B=B)

def gqa_q_perm(d_h: int, n_kv: int, ratio: int, c: int) -> List[int]:
    """Generate Query block permutations for index group selection."""
    return [((g % n_kv) * ratio + c) * d_h + (g // n_kv) for g in range(n_kv * d_h)]


def make_weights_q_gqa(N: int, d: int, H: int, n_kv: int, seed: int = 1) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """Generate Query matrix along with structured block-replicated layout weights."""
    dims = gqa_dims(N, d, H, n_kv)
    H, d_h, ratio, d_kv = dims["H"], dims["d_h"], dims["ratio"], dims["d_kv"]
    t_p = N // d_kv
    R = max(d // t_p, 1)
    Wq = init_weights(d, seed)
    
    encs = []
    for c in range(ratio):
        perm_c = gqa_q_perm(d_h, n_kv, ratio, c)
        Wc = Wq[:, perm_c]
        chunks_enc = []
        for r in range(R):
            block = np.zeros((t_p, d_kv), dtype=np.float64)
            rows = Wc[r * t_p:(r + 1) * t_p, :]
            block[:len(rows), :] = rows
            chunks_enc.append(block.T.flatten())
        encs.append(chunks_enc)
    
    return Wq, encs


def block_replicate(v: np.ndarray, t_p: int) -> np.ndarray:
    """Replicate structural array blocks using power-of-two roll steps."""
    out = v.copy()
    step = 1
    while step < t_p:
        out = out + np.roll(out, step)
        step *= 2
    return out


def vmm_q_gqa(X_enc_chunks: List[np.ndarray], Wq_enc_list: List[List[np.ndarray]], N: int, d: int, H: int, n_kv: int) -> np.ndarray:
    """Perform parallelized VMM calculations for GQA Query vectors."""
    dims = gqa_dims(N, d, H, n_kv)
    d_kv, t_p, ratio = dims["d_kv"], dims["t_p"], dims["ratio"]
    Q_cts = np.zeros((ratio, N), dtype=np.float64)

    for c in range(ratio):
        sparse = vmm_kv(X_enc_chunks, Wq_enc_list[c], N, d_kv, pos=0)
        Q_cts[c] = block_replicate(sparse, t_p)
    return Q_cts


def qkt_gqa(Q_cts: np.ndarray, k_ciphertexts: List[np.ndarray], N: int, d: int, H: int, n_kv: int) -> np.ndarray:
    """Compute structural cross-attention scores matrix."""
    dims = gqa_dims(N, d, H, n_kv)
    d_h, B, ratio = dims["d_h"], dims["B"], dims["ratio"]
    fold_nsteps = int(np.log2(d_h))

    n_k_ct = len(k_ciphertexts)
    n_att_ct = int(np.ceil(n_k_ct / d_h)) if n_k_ct > 0 else 0
    att_cts = np.zeros((ratio, n_att_ct, N), dtype=np.float64)

    # Precompute mask once: keep first B slots, zero the rest.
    mask_B = np.zeros(N, dtype=np.float64)
    mask_B[:B] = 1.0

    for c in range(ratio):
        for g in range(n_att_ct):
            acc = np.zeros(N, dtype=np.float64)
            for k_local in range(d_h):
                k = g * d_h + k_local
                if k >= n_k_ct:
                    break
                prod = Q_cts[c] * k_ciphertexts[k]
                folded = prod.copy()
                step = B
                for _ in range(fold_nsteps):
                    folded = folded + np.roll(folded, -step)
                    step *= 2

                tmp = folded * mask_B
                acc += np.roll(tmp, k_local * B)
            att_cts[c, g] = acc
    return att_cts


def softmax_v_gqa(att_cts: np.ndarray, vcache: VCache, N: int, d: int, H: int, n_kv: int) -> np.ndarray:
    dims = gqa_dims(N, d, H, n_kv)
    H, d_h, B, t_p, ratio = dims["H"], dims["d_h"], dims["B"], dims["t_p"], dims["ratio"]

    # precompute slot masks once: O[h,dim] lands at slot dim*B + kv*t_p in O_ct[c]
    slot_masks = np.zeros((d_h, n_kv, N), dtype=np.float64)
    for dim in range(d_h):
        for kv in range(n_kv):
            slot_masks[dim, kv, dim * B + kv * t_p] = 1.0

    O_ct = np.zeros((ratio, N), dtype=np.float64)

    for c in range(ratio):
        for g in range(att_cts.shape[1]):
            if g >= len(vcache.blocks):
                break
            for dim in range(d_h):
                prod = att_cts[c, g] * vcache.blocks[g][dim]

                # sum across k_local layers within the block
                step = B
                while step < N:
                    prod = prod + np.roll(prod, -step)
                    step *= 2

                # fold within each t_p token group
                step = 1
                while step < t_p:
                    prod = prod + np.roll(prod, -step)
                    step *= 2

                for kv in range(n_kv):
                    O_ct[c] += prod * slot_masks[dim, kv]
    # print("output id:" , O_ct)
    return O_ct


def attention_gqa(X_enc_chunks: List[np.ndarray], kcache: KCache, vcache: VCache, Wq_enc_list: List[List[np.ndarray]], Wk_enc: List[np.ndarray], Wv_enc: List[np.ndarray], N: int, d: int, H: int, n_kv: int):
    """Execute GQA multi-head pipeline sequence across a decoding step."""
    dims = gqa_dims(N, d, H, n_kv)
    d_kv, t_p = dims["d_kv"], dims["t_p"]

    Q_cts = vmm_q_gqa(X_enc_chunks, Wq_enc_list, N, d, H, n_kv)

    pos_k = len(kcache) % t_p
    K_new = vmm_kv(X_enc_chunks, Wk_enc, N, d_kv, pos=pos_k)
    kcache.append(K_new)

    V_new = vmm_kv(X_enc_chunks, Wv_enc, N, d_kv, pos=0)
    vcache.append(V_new)

    att_cts = qkt_gqa(Q_cts, kcache.ciphertexts, N, d, H, n_kv)
    O = softmax_v_gqa(att_cts, vcache, N, d, H, n_kv)
    return Q_cts, K_new, V_new, att_cts, O

def run_attention_gqa(N: int, d: int, H: int, n_kv: int, n_prefill: int, seeds: Tuple[int, ...] = (1, 2, 3, 99)) -> Tuple[float, float]:
    """Run and evaluate GQA precision verification suite."""
    dims = gqa_dims(N, d, H, n_kv)
    H, d_h, ratio, d_kv, t_p, B = dims["H"], dims["d_h"], dims["ratio"], dims["d_kv"], dims["t_p"], dims["B"]

    Wq_raw, Wq_enc_list = make_weights_q_gqa(N, d, H, n_kv, seed=seeds[0])
    Wk_raw, Wk_enc = make_weights_kv(N, d, H, n_kv, seed=seeds[1])
    Wv_raw, Wv_enc = make_weights_kv(N, d, H, n_kv, seed=seeds[2])

    kcache = KCache(N, d_kv)
    vcache = VCache(N, d_kv, H=n_kv)

    toks = [init_input(d, seed=10 + i) for i in range(n_prefill)]
    for x in toks:
        xc = preprocess_input_kv(x, N, d, H, n_kv)
        kcache.append(vmm_kv(xc, Wk_enc, N, d_kv, pos=len(kcache) % t_p))
        vcache.append(vmm_kv(xc, Wv_enc, N, d_kv, pos=0))

    x_new = init_input(d, seed=seeds[3])
    xc_new = preprocess_input_kv(x_new, N, d, H, n_kv)

    Q_cts, K_new, V_new, att_cts, O = attention_gqa(
        xc_new, kcache, vcache, Wq_enc_list, Wk_enc, Wv_enc, N, d, H, n_kv
    )

    # Computing the reference GQA to check the correctness of encoding
    n = len(kcache)
    all_toks = toks + [x_new]
    perm_kv = head_perm(d_kv, n_kv)
    K_matrix = np.array([x @ Wk_raw[:, perm_kv] for x in all_toks]).T
    V_matrix = np.array([x @ Wv_raw[:, perm_kv] for x in all_toks])
    Qf = x_new @ Wq_raw

    n_att_ct = att_cts.shape[1]
    decoded_map = np.zeros((H, n))
    for c in range(ratio):
        for g in range(n_att_ct):
            for s in range(N):
                k_local = s // B
                within = s % B
                kv = within // t_p
                tok_in_ct = within % t_p
                k = g * d_h + k_local
                tok = k * t_p + tok_in_ct
                if tok >= n: 
                    continue
                decoded_map[kv * ratio + c, tok] = att_cts[c, g, s]

    ref_map = np.zeros((H, n))
    for h in range(H):
        kv = h // ratio
        for tok in range(n):
            for i in range(d_h):
                ref_map[h, tok] += Qf[h * d_h + i] * K_matrix[i * n_kv + kv, tok]

    def score(h: int, tok: int) -> float:
        kv = h // ratio
        return sum(Qf[h * d_h + i] * K_matrix[i * n_kv + kv, tok] for i in range(d_h))

    ref_O = np.zeros((H, d_h))
    for h in range(H):
        kv = h // ratio
        for dim in range(d_h):
            g = dim * n_kv + kv
            ref_O[h, dim] = sum(V_matrix[tok, g] * score(h, tok) for tok in range(n))
    
    # extract dense O from O for comparison
    O_dense = np.zeros((H, d_h), dtype=np.float64)
    for c in range(ratio):
        for kv in range(n_kv):
            for dim in range(d_h):
                h = kv * ratio + c
                O_dense[h, dim] = O[c][dim * B + kv * t_p]
    err_qkt = np.max(np.abs(decoded_map - ref_map))
    err_v = np.max(np.abs(O_dense - ref_O))
    ok = max(err_qkt, err_v) < 1e-9
    
    print(
        f"N={N:3d} d={d:2d} H={H} n_kv={n_kv} (d_h={d_h},ratio={ratio},d_kv={d_kv},"
        f"t_p={t_p},B={B}) n_prefill={n_prefill} (n={n}): "
        f"qkt_err={err_qkt:.2e}, softmaxv_err={err_v:.2e} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return err_qkt, err_v


if __name__ == "__main__":

    # Test suites pulled outside to module level space
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=6)
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=3)
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=4)
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=10)
    run_attention_gqa(N=32, d=16, H=8, n_kv=4, n_prefill=5)
    run_attention_gqa(N=16, d=8, H=4, n_kv=4, n_prefill=3)   # MHA
    run_attention_gqa(N=16, d=8, H=8, n_kv=1, n_prefill=3)   # MQA
    run_attention_gqa(N=16, d=8, H=8, n_kv=2, n_prefill=4)
    run_attention_gqa(N=32, d=16, H=8, n_kv=1, n_prefill=5)  # MQA Large
    run_attention_gqa(N=32, d=16, H=16, n_kv=2, n_prefill=4)
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=1) 
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=2) 
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=5)
    run_attention_gqa(N=16, d=8, H=4, n_kv=2, n_prefill=8)
    run_attention_gqa(N=64, d=16, H=8, n_kv=1, n_prefill=6) 
    run_attention_gqa(N=64, d=32, H=8, n_kv=4, n_prefill=7)
    run_attention_gqa(N=64, d=32, H=16, n_kv=2, n_prefill=5) 
    run_attention_gqa(N=16, d=8, H=1, n_kv=1, n_prefill=4)
    run_attention_gqa(N=32, d=8, H=2, n_kv=2, n_prefill=6)