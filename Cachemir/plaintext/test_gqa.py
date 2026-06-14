import numpy as np

# ---------------------------------------------------------------------------
# GQA verification: QK^T  and  Softmax(QK^T) @ V
#
# Set parameters below, then run. Both steps share the same Q/K encoding and
# attention-map ciphertexts (att_cts).
#
# d_h   = d // H
# ratio = H // n_kv          (query heads per KV head)
# t_p   = (N // d) * ratio   (tokens per K ciphertext)
# B     = n_kv * t_p         (fold step; N == d_h * B)
#
# n_k_ct   = ceil(n_p / t_p)         (number of K/V ciphertexts)
# n_att_ct = ceil(n_k_ct / d_h)      (number of attention-map ciphertexts
#                                      per Q ciphertext -- needed because a
#                                      single ciphertext can only address
#                                      k_local in [0, d_h) before the
#                                      accumulation shift wraps around)
# ---------------------------------------------------------------------------

# ----- parameters: edit these -----
N     = 8
d     = 8
H     = 4
n_kv  = 2
n_p   = 17
seed  = 42
# -----------------------------------

d_h         = d // H
ratio       = H // n_kv
t_p         = (N // d) * ratio
B           = n_kv * t_p
fold_nsteps = int(np.log2(d_h))   # QKT fold steps (sum over d_h)
fold_t_p    = int(np.log2(t_p))   # Softmax.V final fold steps (combine t_p stride-classes)
n_q_ct      = ratio
n_k_ct      = int(np.ceil(n_p / t_p))
n_att_ct    = int(np.ceil(n_k_ct / d_h))
n_tok_pad   = n_att_ct * d_h * t_p

assert N == d_h * B

print(f"d_h={d_h}, ratio={ratio}, t_p={t_p}, B={B}, "
      f"n_k_ct={n_k_ct}, n_att_ct={n_att_ct}")

rng = np.random.default_rng(seed)
Q = rng.standard_normal(d).astype(np.float64)
K = rng.standard_normal((n_kv * d_h, n_p)).astype(np.float64)
V = rng.standard_normal((n_p, n_kv * d_h)).astype(np.float64)
V_pad = np.zeros((n_tok_pad, n_kv * d_h))
V_pad[:n_p] = V


# ---------------------------------------------------------------------------
# Encode Q -- ratio ciphertexts
#   slot = dim*B + kv*t_p + rep,   rep in [0..t_p)
#   value = Q[(kv*ratio + c)*d_h + dim]
# ---------------------------------------------------------------------------
Q_cts = np.zeros((n_q_ct, N))
for c in range(ratio):
    for dim in range(d_h):
        for kv in range(n_kv):
            for rep in range(t_p):
                slot = dim * B + kv * t_p + rep
                Q_cts[c, slot] = Q[(kv * ratio + c) * d_h + dim]

# ---------------------------------------------------------------------------
# Encode K -- n_k_ct ciphertexts
#   slot = dim*B + kv*t_p + tok_in_ct
#   value = K[kv*d_h+dim, k*t_p+tok_in_ct]   (0 if token >= n_p)
# ---------------------------------------------------------------------------
K_cts = np.zeros((n_k_ct, N))
for k in range(n_k_ct):
    for dim in range(d_h):
        for kv in range(n_kv):
            for tok_in_ct in range(t_p):
                tok = k * t_p + tok_in_ct
                if tok >= n_p:
                    continue
                slot = dim * B + kv * t_p + tok_in_ct
                K_cts[k, slot] = K[kv * d_h + dim, tok]


# ---------------------------------------------------------------------------
# QK^T  -- att_cts[c][g], g in [0, n_att_ct), each covering tokens
#          [g*d_h*t_p, (g+1)*d_h*t_p)
# ---------------------------------------------------------------------------
att_cts = np.zeros((n_q_ct, n_att_ct, N))
for c in range(n_q_ct):
    for g in range(n_att_ct):
        acc = np.zeros(N)
        for k_local in range(d_h):
            k = g * d_h + k_local
            if k >= n_k_ct:
                break
            prod   = Q_cts[c] * K_cts[k]
            folded = prod.copy()
            step   = B
            for _ in range(fold_nsteps):
                folded = folded + np.roll(folded, -step)
                step  *= 2
            tmp = np.zeros(N)
            tmp[:B] = folded[:B]
            acc += np.roll(tmp, k_local * B)
        att_cts[c, g] = acc


# ---------------------------------------------------------------------------
# Verification: QK^T
# ---------------------------------------------------------------------------
def verify_qkt():
    ref_map = np.zeros((H, n_p))
    for h in range(H):
        kv = h // ratio
        for tok in range(n_p):
            for i in range(d_h):
                ref_map[h, tok] += Q[h * d_h + i] * K[kv * d_h + i, tok]

    decoded_map = np.zeros((H, n_p))
    for c in range(ratio):
        for g in range(n_att_ct):
            for s in range(N):
                k_local   = s // B
                within    = s %  B
                kv        = within // t_p
                tok_in_ct = within %  t_p
                k         = g * d_h + k_local
                tok       = k * t_p + tok_in_ct
                if tok >= n_p:
                    continue
                q_head = kv * ratio + c
                decoded_map[q_head, tok] = att_cts[c, g, s]

    err = np.max(np.abs(decoded_map - ref_map))
    print("\n=== QK^T verification ===")
    print("Reference map (H x n_p):\n", np.round(ref_map, 4))
    print("Decoded map (H x n_p):\n",   np.round(decoded_map, 4))
    print(f"Max absolute error: {err:.2e}", " PASS ✓" if err < 1e-10 else " FAIL ✗")
    return err


# ---------------------------------------------------------------------------
# Verification: Softmax(QK^T) @ V
#   (att_cts treated as raw scores -- softmax to be inserted later)
# ---------------------------------------------------------------------------
def verify_softmaxv():
    def score(h, tok):
        kv = h // ratio
        return sum(Q[h * d_h + i] * K[kv * d_h + i, tok] for i in range(d_h))

    ref_O = np.zeros((H, d_h))
    for h in range(H):
        kv = h // ratio
        for dim in range(d_h):
            col = kv * d_h + dim
            ref_O[h, dim] = sum(V[t, col] * score(h, t) for t in range(n_p))

    decoded_O = np.zeros((H, d_h))
    for c in range(n_q_ct):
        for g in range(n_att_ct):
            # sum over m = 0..d_h-1: Row_m * roll(att, -m*B)
            # Row_m[s] = V[ g*d_h*t_p + tok_in_ct + ((dim_block+m) % d_h)*t_p, col ]
            result = np.zeros(N)
            for m in range(d_h):
                Row_m = np.zeros(N)
                for s in range(N):
                    dim_block = s // B
                    within    = s %  B
                    kv        = within // t_p
                    tok_in_ct = within %  t_p
                    col       = kv * d_h + dim_block
                    j         = (dim_block + m) % d_h
                    token     = g * d_h * t_p + tok_in_ct + j * t_p
                    Row_m[s]  = V_pad[token, col]
                result += Row_m * np.roll(att_cts[c, g], -m * B)

            # combine the t_p stride-classes (binary-tree fold, strides 1,2,4,...)
            final = result.copy()
            step = 1
            for _ in range(fold_t_p):
                final = final + np.roll(final, -step)
                step *= 2

            for s in range(N):
                if s % t_p == 0:
                    dim_block = s // B
                    kv        = (s % B) // t_p
                    head      = kv * ratio + c
                    decoded_O[head, dim_block] += final[s]

    err = np.max(np.abs(decoded_O - ref_O))
    print("\n=== Softmax(QK^T) @ V verification ===")
    print("Reference O (H x d_h):\n", np.round(ref_O, 4))
    print("Decoded O (H x d_h):\n",   np.round(decoded_O, 4))
    print(f"Max absolute error: {err:.2e}", " PASS ✓" if err < 1e-10 else " FAIL ✗")
    return err


if __name__ == "__main__":
    verify_qkt()
    verify_softmaxv()