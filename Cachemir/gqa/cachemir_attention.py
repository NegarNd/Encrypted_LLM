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
"""

from typing import List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Initialization - dimension check + separate init for input and weights
# ---------------------------------------------------------------------------
def check_dims(N, d):
    """Check that the dimensions are valid powers of two and divisible."""
    if N & (N - 1) != 0:
        raise ValueError(f"N ({N}) must be a power of two.")
    if N % d != 0:
        raise ValueError(f"N ({N}) must be divisible by d ({d}).")

def normalize_heads(H):
    """Normalize head count: 0 or 1 is treated as a single head (returns 1)."""
    return 1 if H in (0, 1) else H

def init_input(d, seed=0):
    """Build a random input vector X of length d."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(d)


def init_weights(d, seed=1):
    """Build a random weight matrix B of shape (d, d)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, d))

def head_perm(d, H):
    """
    Column permutation that head-interleaves the d output dims at group
    granularity.
    """
    H = normalize_heads(H)
    dh = d // H
    return [(g % H) * dh + (g // H) for g in range(d)]

# ---------------------------------------------------------------------------
# Step 1a - preprocess the input vector X  (client-side, before encryption)
# ---------------------------------------------------------------------------
def preprocess_input(X, N, d):
    """Encode X into the interleaved replicated layout via log2(t) rotate-add steps."""
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
    """Encode B into n_iters = d/t interleaved generalized-diagonal plaintexts."""
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
    Apply interleaved-replicated VMM to the encoded input.
    Fuses the K-cache placement rotation via the `pos` argument.
    """
    t = N // d
    acc = np.zeros(N, dtype=np.float64)

    # Multiply-accumulate
    for it, W_it in enumerate(W_list):
        rotated = np.roll(X_enc, -(it * t * t))     # rotate left by it*t^2
        acc += rotated * W_it

    # Reduce - folding
    stride, i = 1, 0
    while stride < t:
        acc += np.roll(acc, +stride if (pos >> i) & 1 else -stride)
        stride *= 2
        i += 1

    # Masking
    mask = np.zeros(N, dtype=np.float64)
    mask[pos::t] = 1.0
    return acc * mask


def extract(out_interleaved, N, d):
    """Extract dense length-d output vector from interleaved slots."""
    t = N // d
    return out_interleaved[::t][:d].copy()


# ---------------------------------------------------------------------------
# "make" helpers - generate + encode in one step (raw value never floats free)
# ---------------------------------------------------------------------------
def make_input(N: int, d:int, seed:int = 0):
    """Generate a random input vector and its interleaved encoding."""
    X = init_input(d, seed)
    return X, preprocess_input(X, N, d)


def make_weights(N:int, d:int, seed:int = 1, H:int = 1):
    """Generate a random weight matrix and its interleaved-diagonal encoding."""
    W = init_weights(d, seed)
    perm = head_perm(d, H)
    return W, preprocess_weights(W[:, perm], N, d)


# ---------------------------------------------------------------------------
# GQA helpers
# ---------------------------------------------------------------------------
def gqa_head_perm(d: int, H: int, n_kv: int):
    """Column permutation (length d) for K/V projections under Grouped-Query Attention (GQA)."""
    H = normalize_heads(H)
    if H % n_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by n_kv ({n_kv}).")
    dh = d // H
    ratio = H // n_kv
    return [((g % H) // ratio) * dh + (g // H) for g in range(d)]


def make_weights_gqa(N: int, d: int, n_kv: int, H: int = 1, seed: int = 1):
    """Generate a GQA K/V projection weight matrix and its encoding."""
    H = normalize_heads(H)
    dh = d // H
    d_kv = n_kv * dh
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, d_kv))
    perm = gqa_head_perm(d, H, n_kv)
    return W, preprocess_weights(W[:, perm], N, d)


# ---------------------------------------------------------------------------
# GQA K/V: compact encoding (d -> d_kv, interleaved with t_p = N/d_kv = ratio*t)
# ---------------------------------------------------------------------------
def make_weights_kv(N: int, d: int, H: int, n_kv: int, seed: int = 1):
    """Generate a GQA K/V projection weight matrix and its compact encoding."""
    H = normalize_heads(H)
    dh = d // H
    d_kv = n_kv * dh
    t_p = N // d_kv
    R = max(d // t_p, 1)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, d_kv))
    perm = head_perm(d_kv, n_kv)
    Wp = W[:, perm]
    chunks_enc = []
    for r in range(R):
        block = np.zeros((t_p, d_kv), dtype=np.float64)
        rows = Wp[r*t_p : (r+1)*t_p, :]   # truncates naturally if t_p > d
        block[:len(rows), :] = rows
        chunks_enc.append(block.T.flatten())
    return W, chunks_enc


def preprocess_input_kv(X: np.ndarray, N: int, d: int, H: int, n_kv: int):
    """Encode input X into chunk encodings for vmm_kv."""
    H = normalize_heads(H)
    dh = d // H
    d_kv = n_kv * dh
    t_p = N // d_kv
    R = max(d // t_p, 1)

    chunks = []
    for r in range(R):
        chunk = np.zeros(t_p, dtype=np.float64)
        chunk[:min(t_p, d - r*t_p)] = X[r*t_p : (r+1)*t_p]
        chunks.append(np.tile(chunk, d_kv))
    return chunks


def vmm_kv(X_enc_chunks: List[np.ndarray], W_enc_chunks: List[np.ndarray], N: int, d_kv: int, pos: int = 0):
    """Compact GQA K/V projection summing R chunk outputs."""
    t_p = N // d_kv
    out = np.zeros(N, dtype=np.float64)

    mask = np.zeros(N, dtype=np.float64)
    mask[pos::t_p] = 1.0

    for Xc, Wc in zip(X_enc_chunks, W_enc_chunks):
        acc = Xc * Wc
        step, i = 1, 0
        while step < t_p:
            acc += np.roll(acc, +step if (pos >> i) & 1 else -step)
            step *= 2
            i += 1

        out += acc * mask
    return out


# ---------------------------------------------------------------------------
# QK^T over cached keys (Figure 5c)
# ---------------------------------------------------------------------------
def qkt(Q: np.ndarray, k_ciphertexts: List[np.ndarray], n_tokens: int, N: int, d: int, H: int = 1):
    """Compute Attention scores Q . K^T over the cached keys."""
    H = normalize_heads(H)
    t = N // d

    # Preprocess - replicate Q
    s = 1
    while s < t:
        Qr = Qr + np.roll(Qr, s)          
        s *= 2

    attn = np.zeros(N, dtype=np.float64)
    for c, ct in enumerate(k_ciphertexts):
        # Multiply & Fold
        acc = Qr * ct
        s = H * t
        while s < N:
            acc = acc + np.roll(acc, -s)    
            s *= 2

        # Mask
        valid = min(t, n_tokens - c * t)           
        mask = np.zeros(N, dtype=np.float64)
        for h in range(H):
            mask[h * t: h * t + valid] = 1.0        #
        # Rotate and accumulate
        attn += np.roll(acc * mask, c * H * t)

    return attn

# ---------------------------------------------------------------------------
# Softmax * V over the cached values (Figure 6c)
# ---------------------------------------------------------------------------
def softmax_v(scores, vcache, N, d, H=1):
    """Compute Attention output Att = scores . V over the cached values."""
    H = normalize_heads(H)
    t = N // d
    n = len(vcache)
    cap = N // H                 # tokens per block
    dh = d // H                  # V-ciphertexts per block
    att = np.zeros(N, dtype=np.float64)

    for b, block in enumerate(vcache.blocks):
        S = np.zeros(N, dtype=np.float64)
        S[:len(scores)] = scores if b == 0 else 0.0   

    
        P = np.zeros(N, dtype=np.float64)
        for c in range(dh):
            P += np.roll(S, -(c * H * t)) * block[c]  

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
    """Execute one decoding step of attention (softmax omitted)."""
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
    """Run the full Cachemir VMM for given dimensions and return (X, B, C)."""

    check_dims(N, d)

    X = init_input(d, seed_x)                   # initialize input
    B = init_weights(d, seed_w)                 # initialize weights

    X_enc = preprocess_input(X, N, d)           # Step 1a (input)
    W_list = preprocess_weights(B, N, d)        # Step 1b (weights)
    out = vmm(X_enc, W_list, N, d)              # Steps 2-4 (interleaved output)
    C = extract(out, N, d)                      # dense length-d result

    return X, B, C
