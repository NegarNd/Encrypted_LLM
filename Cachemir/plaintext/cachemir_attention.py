"""
cachemir_attention.py
==================
Plaintext simulation of Cachemir's Interleaved Replicated Packing.
Conventions
-----------
  N        polynomial degree (power of two)
  n_he     slot count   (power of two)
  d        feature dimension
  t = N/d  interleaving / replication factor
  n_iters = d/t   number of multiply-accumulate steps

  C = X @ B   means   C[c] = sum_r X[r] * B[r][c].

Slot layout
-----------
Everything stays in the *interleaved* format end to end: a length-d logical
vector V is stored with V[g] in slot g*t, the other slots being scratch.
The output of one VMM can feed straight into the next
layer without re-encoding (needed for chaining Q/K/V projections, MLP, etc.).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Step 1a - preprocess the input vector X  (client-side, before encryption)
# ---------------------------------------------------------------------------
def preprocess_input(X, N, d):
    """
    Encode X into the interleaved replicated layout.
 
    The client supplies the sparse encoding [x0, 0, x1, 0, ...] (X at every t-th
    slot).  Filling it into the replicated form takes log2(t) = log2(N/d)
    rotate-and-add steps with stride 2^i*(t-1):
 
        enc[s] = X[(s//t + s%t) % d]
 
    Example (d=4, N=8, t=2):  [x0, x1, x1, x2, x2, x3, x3, x0]
    """
    t = N // d
    enc = np.zeros(N, dtype=np.float64)
    enc[::t] = X                         # client-side sparse encoding
    i = 0
    while (1 << i) < t:                  # log2(t) rotate-add steps
        enc = enc + np.roll(enc, -((1 << i) * (t - 1)))
        i += 1
    return enc

 
# ---------------------------------------------------------------------------
# Step 1b - preprocess the weight matrix B into interleaved diagonals
# ---------------------------------------------------------------------------
def preprocess_weights(B, N, d):
    """
    Encode B into n_iters = d/t interleaved generalized-diagonal plaintexts.

        W_iter[s] = B[(s//t + s%t + iter*t) % d][(s//t) % d]

    - column index  (s//t) % d   selects which output column (diagonal group)
    - row index     (s//t + s%t + iter*t) % d   walks the diagonal
    """
    t = N // d
    n_iters = d // t
    return [
        np.array(
            [B[(s // t + s % t + it * t) % d][(s // t) % d] for s in range(N)],
            dtype=np.float64,
        )
        for it in range(n_iters)
    ]


# ---------------------------------------------------------------------------
# Steps 2-4 - multiply-accumulate, reduce, mask 
# Vector-Matrix Multiply, generating Q, K, and V
# ---------------------------------------------------------------------------
def vmm(X_enc, W_list, N, d, pos=0):
    """
    Apply the interleaved-replicated VMM to the encoded input.

      Step 2 (multiply-accumulate): for each iteration `it`, rotate the input
              left by it*t^2, multiply by the interleaved diagonal W_it, and
              accumulate.  Standard rotations only - no inner rotations.
      Step 3 (reduce):              fold the t partial sums sitting in each
              group of t adjacent slots with log2(t) rotations.  The rotation
              directions are flipped per the bits of `pos` so the group sum
              lands at slot j*t + pos (default 0).
      Step 4 (mask):                keep the interleaved slots {pos, t+pos, ...}.

    The `pos` argument fuses the K-cache placement rotation into the projection
    (pos=0 reproduces the plain interleaved output).

    Returns the full N-slot result still in interleaved format.  Use
    `extract(out, N, d)` to pull the dense length-d vector (pos=0 only).
    """
    t = N // d

    # Step 2: multiply-accumulate
    acc = np.zeros(N, dtype=np.float64)
    for it, W_it in enumerate(W_list):
        rotated = np.roll(X_enc, -(it * t * t))     # rotate left by it*t^2
        acc += rotated * W_it

    # Step 3: reduce - fold each group, landing the sum at offset `pos`
    stride, i = 1, 0
    while stride < t:
        acc += np.roll(acc, +stride if (pos >> i) & 1 else -stride)
        stride *= 2
        i += 1

    # Step 4: mask - keep slots {pos, t+pos, 2t+pos, ...}
    mask = np.zeros(N, dtype=np.float64)
    mask[pos::t] = 1.0
    return acc * mask


def extract(out_interleaved, N, d):
    """
    Pull the dense length-d output vector C from the interleaved slots:
    C[g] lives in slot g*t.
    """
    t = N // d
    return out_interleaved[::t][:d].copy()


# ---------------------------------------------------------------------------
# Initialization - dimension check + separate init for input and weights
# ---------------------------------------------------------------------------
def check_dims(N, d):
    """
    Check that the dimensions are valid: N must be a power of two and
    divisible by d.
    """
    if N & (N - 1) != 0:
        raise ValueError(f"N ({N}) must be a power of two.")
    if N % d != 0:
        raise ValueError(f"N ({N}) must be divisible by d ({d}).")


def init_input(d, seed=0):
    """Build a random input vector X of length d."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(d)


def init_weights(d, seed=1):
    """Build a random weight matrix B of shape (d, d)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, d))


# ---------------------------------------------------------------------------
# Multi-head helpers
# ---------------------------------------------------------------------------
def normalize_heads(H):
    """H=0 or H=1 -> single head (returns 1); otherwise returns H."""
    return 1 if H in (0, 1) else H


def head_perm(d, H):
    """
    Column permutation that head-interleaves the d output dims at group
    granularity.  Group g carries global dim perm[g] = (g % H)*(d/H) + g//H.
    For H=1 this is the identity, so the single-head path is unchanged.

    Example d=4, H=2:  [0, 2, 1, 3]  ->  K packs as
        [k00, k10 | k02, k12 | k01, k11 | k03, k13]   (heads interleaved).
    """
    H = normalize_heads(H)
    dh = d // H
    return [(g % H) * dh + (g // H) for g in range(d)]


# ---------------------------------------------------------------------------
# "make" helpers - generate + encode in one step (raw value never floats free)
# ---------------------------------------------------------------------------
def make_input(N, d, seed=0):
    """
    Generate a random input vector and its interleaved encoding.

    Returns
    -------
    X      : np.ndarray (d,)      the raw input vector
    X_enc  : np.ndarray (N,)      its interleaved replicated encoding
    """
    X = init_input(d, seed)
    return X, preprocess_input(X, N, d)


def make_weights(N, d, seed=1, H=1):
    """
    Generate a random weight matrix and its interleaved-diagonal encoding.

    For multi-head (H>1) the weight columns are reordered by head_perm before
    encoding, so the projection output comes out head-interleaved.  H=0/1 leaves
    the matrix untouched (single head).

    Returns
    -------
    W       : np.ndarray (d, d)            the raw weight matrix
    W_enc   : list[np.ndarray (N,)]        its per-iteration diagonal plaintexts
    """
    W = init_weights(d, seed)
    perm = head_perm(d, H)
    return W, preprocess_weights(W[:, perm], N, d)


# ---------------------------------------------------------------------------
# QK^T over cached keys (Figure 5c)
# ---------------------------------------------------------------------------
def qkt(Q, k_ciphertexts, n_tokens, N, d, H=1):
    """
    Attention scores Q . K^T over the cached keys.

    Q              : interleaved query (vmm output, head-interleaved if H>1)
    k_ciphertexts  : list of packed K-cache ciphertexts (KCache.ciphertexts)
    n_tokens       : number of cached tokens (len(cache))
    H              : number of heads (0/1 = single head)

    Single head: returns [m0, m1, ..., 0, ...] with m_i = sum_j Q[j]*K_i[j].
    Multi head:  the fold sums only within each head (stride H*t, log2(d/H)
    steps) and each K ciphertext's H*t results are packed at offset c*H*t, so
    the map is head-interleaved:
        slot (c*H*t + h*t + i%t) = m_{c*t+i}^{head h}.
    """
    H = normalize_heads(H)
    t = N // d

    # Step 1: preprocess - replicate Q within each group: [q0,q0,q1,q1,...]
    Qr = Q.copy()
    s = 1
    while s < t:
        Qr = Qr + np.roll(Qr, s)            # rotate right, add (log2(t) steps)
        s *= 2

    attn = np.zeros(N, dtype=np.float64)
    for c, ct in enumerate(k_ciphertexts):
        # Step 2: multiply, then fold groups H apart (log2(d/H) rotations).
        # H=1 -> stride t, log2(d) steps == single head.
        acc = Qr * ct
        s = H * t
        while s < N:
            acc = acc + np.roll(acc, -s)    # rotate left, add
            s *= 2
        # Step 3: mask the H*t valid (head, token) results of this ciphertext
        valid = min(t, n_tokens - c * t)            # tokens in this ciphertext
        mask = np.zeros(N, dtype=np.float64)
        for h in range(H):
            mask[h * t: h * t + valid] = 1.0        # head h block, valid tokens
        # Step 4: position the H*t-slot block at offset c*H*t and accumulate
        attn += np.roll(acc * mask, c * H * t)

    return attn


# ---------------------------------------------------------------------------
# Softmax * V over the cached values (Figure 6c)
# ---------------------------------------------------------------------------
def softmax_v(scores, vcache, N, d, H=1):
    """
    Attention output Att = scores . V over the cached values.

    scores : the QK^T / softmax output (head-interleaved map if H>1; softmax is
             skipped here since it preserves the format)
    vcache : VCache holding the tokens' V vectors (diagonal-interleaved)
    H      : number of heads (0/1 = single head)

    Returns Att in interleaved format. Single head: Att[j] at slot j*t.
    Multi head: head-interleaved, Att for (head h, local l) at slot (l*H+h)*t.

    Per block there are d/H V-ciphertexts holding up to N/H tokens; the scores
    are folded against them by rotating left by c*H*t.  H=1 -> d ciphertexts,
    roll c*t == single head.
    """
    H = normalize_heads(H)
    t = N // d
    n = len(vcache)
    cap = N // H                 # tokens per block
    dh = d // H                  # V-ciphertexts per block
    att = np.zeros(N, dtype=np.float64)

    for b, block in enumerate(vcache.blocks):
        # scores for the tokens in this block
        lo = b * cap
        cnt = min(cap, n - lo)
        S = np.zeros(N, dtype=np.float64)
        S[:len(scores)] = scores if b == 0 else 0.0   # single-block map (typical)

        # multiply-accumulate over the block's d/H ciphertexts
        P = np.zeros(N, dtype=np.float64)
        for c in range(dh):
            P += np.roll(S, -(c * H * t)) * block[c]   # rotate S left by c*H*t

        # fold each group of t slots (log2(t) rotations)
        s = 1
        while s < t:
            P += np.roll(P, -s)
            s *= 2

        att += P

    # mask to interleaved output (head-interleaved when H>1)
    mask = np.zeros(N, dtype=np.float64)
    mask[::t] = 1.0
    return att * mask


# ---------------------------------------------------------------------------
# Attention - one decoding step (projections -> cache update -> QK^T -> *V)
# ---------------------------------------------------------------------------
def attention(x_enc, kcache, vcache, Wq_enc, Wk_enc, Wv_enc, N, d, H=1):
    """
    One decoding step of attention (softmax omitted - it preserves the format).

    x_enc                  : interleaved-encoded input of the new token
    kcache, vcache         : KCache / VCache holding the previous tokens
    Wq_enc, Wk_enc, Wv_enc : encoded projection weights (make_weights output;
                             pass H to make_weights so they are head-reordered)
    H                      : number of heads (0/1 = single head)

    Projects the new token to Q, K, V; appends K and V to their caches;
    computes QK^T over all cached keys; and multiplies by the cached values.

    Returns
    -------
    Q, K, V : interleaved projections of the new token (head-interleaved if H>1)
    scores  : attention map (QK^T, pre-softmax; head-interleaved if H>1)
    Att     : attention output in interleaved format
    """
    H = normalize_heads(H)
    Q = vmm(x_enc, Wq_enc, N, d)
    K = vmm(x_enc, Wk_enc, N, d, pos=len(kcache) % (N // d))   # fused K placement
    V = vmm(x_enc, Wv_enc, N, d)

    kcache.append(K)                                  # update K cache
    vcache.append(V)                                  # update V cache

    scores = qkt(Q, kcache.ciphertexts, len(kcache), N, d, H)   # QK^T
    Att = softmax_v(scores, vcache, N, d, H)          # (softmax) * V
    return Q, K, V, scores, Att


# ---------------------------------------------------------------------------
# Main kernel - tie everything together
# ---------------------------------------------------------------------------
def kernel(N, d, seed_x=0, seed_w=1):
    """
    Run the full Cachemir VMM for given dimensions and return (X, B, C).
    C is the dense length-d output, equal to X @ B.
    """
    check_dims(N, d)

    X = init_input(d, seed_x)                   # initialize input
    B = init_weights(d, seed_w)                 # initialize weights

    X_enc = preprocess_input(X, N, d)           # Step 1a (input)
    W_list = preprocess_weights(B, N, d)        # Step 1b (weights)
    out = vmm(X_enc, W_list, N, d)              # Steps 2-4 (interleaved output)
    C = extract(out, N, d)                      # dense length-d result

    return X, B, C


if __name__ == "__main__":
    from k_cache import KCache
    from v_cache import VCache

    def run_attention(N, d, H, n_prefill):
        """Prefill caches with n_prefill tokens, decode one, verify vs numpy."""
        H = normalize_heads(H)
        t = N // d
        dh = d // H
        assert n_prefill + 1 <= N // H, (
            f"n={n_prefill + 1} exceeds single-ciphertext capacity N/H={N // H}"
        )

        # head-reordered projection weights
        Wq, Wq_enc = make_weights(N, d, seed=1, H=H)
        Wk, Wk_enc = make_weights(N, d, seed=2, H=H)
        Wv, Wv_enc = make_weights(N, d, seed=3, H=H)

        kcache = KCache(N, d)
        vcache = VCache(N, d, H=H)
        toks = [init_input(d, seed=10 + i) for i in range(n_prefill)]
        for x in toks:
            x_enc = preprocess_input(x, N, d)
            kcache.append(vmm(x_enc, Wk_enc, N, d, pos=len(kcache) % t))
            vcache.append(vmm(x_enc, Wv_enc, N, d))

        x_new = init_input(d, seed=99)
        Q, K, V, scores, Att = attention(
            preprocess_input(x_new, N, d), kcache, vcache,
            Wq_enc, Wk_enc, Wv_enc, N, d, H=H,
        )

        # reference: per-head scores and Att (softmax skipped), head-interleaved
        n = len(kcache)
        all_toks = toks + [x_new]
        Qf = x_new @ Wq
        ref = np.zeros(d)
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            Sh = np.array([Qf[sl] @ (all_toks[i] @ Wk)[sl] for i in range(n)])
            Vh = np.array([(all_toks[i] @ Wv)[sl] for i in range(n)])
            Ah = Sh @ Vh
            for l in range(dh):
                ref[l * H + h] = Ah[l]          # head-interleaved group

        got = np.array([Att[k * t] for k in range(d)])
        err = np.max(np.abs(got - ref))
        print(f"N={N}, d={d}, H={H}, tokens={n}: Att max abs error = {err:.2e}")

    # single head and multi head, same code path
    run_attention(N=8, d=4, H=1, n_prefill=5)
    run_attention(N=8, d=4, H=2, n_prefill=3)
    run_attention(N=16, d=8, H=2, n_prefill=6)
    run_attention(N=16, d=8, H=4, n_prefill=3)