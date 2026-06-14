"""
v_cache.py
==========
Interleaved V cache for Cachemir (Figure 6 of arXiv:2602.11470).

The V cache stores value vectors in a layout *different* from the K cache.
Whereas the K cache packs t tokens per ciphertext (one ciphertext = t whole
tokens), the V cache packs V in interleaved *diagonal* form so that the
Softmax x V step is exactly the same interleaved VMM used elsewhere:

    Att[j] = sum_i  S[i] * V[i][j]          (S length n,  V is n x d)

A "block" of d ciphertexts holds up to N tokens.  Within a block, the layout is

    Vcache[c][s] = V[token][dim],   token = ((s//t + c)*t + s%t) % N,  dim = s//t

so a single token's d values are spread across all d ciphertexts (one value per
ciphertext), each in a different slot.  This is what lets the Softmax x V VMM
run with standard rotations only.

V cache update (Figure 6d): a new token i is placed with
    q, r = divmod(i mod N, t)
    for each dim j:  ciphertext c = (q - j) % d,  slot s = j*t + r
i.e. the "customizable interleaved packing of V[i]" plus d masked adds.  As in
the paper, the masking can be fused with the X*Wv projection, so the update
costs only cheap pt-ct operations.

When the token count exceeds N, a new block of d ciphertexts is started; the
cache therefore holds ceil(n/N) blocks of d ciphertexts each.

Stateful, so it lives in a class - separate from the stateless kernel, mirroring
KCache.
"""

import numpy as np

from cachemir_attention import check_dims, normalize_heads, preprocess_input, preprocess_weights, vmm


class VCache:
    """
    Interleaved-diagonal V cache.  A block of d/H ciphertexts holds up to N/H
    tokens.  H=0/1 -> single head (d ciphertexts, N tokens per block).

    Parameters
    ----------
    N : int   polynomial degree / slot count (power of two)
    d : int   feature dimension
    H : int   number of heads (0/1 = single head)
    """

    def __init__(self, N, d, H=1):
        check_dims(N, d)
        self.N = N
        self.d = d
        self.H = normalize_heads(H)
        self.t = N // d
        self.dh = d // self.H        # V-ciphertexts per blockی
        self.cap = N // self.H       # tokens per block
        self.blocks = []     # each block: list of d/H length-N arrays
        self.length = 0      # number of tokens cached

    # ------------------------------------------------------------------
    @property
    def num_blocks(self):
        """Number of blocks = ceil(n / (N/H))."""
        return len(self.blocks)

    @property
    def num_ciphertexts(self):
        """Total ciphertexts across all blocks = (d/H) * num_blocks."""
        return self.dh * len(self.blocks)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def append(self, v_interleaved):
        """
        Append one token's V vector (Figure 6d), head-interleaved when H>1.

        v_interleaved : the token's V vector from vmm (head-interleaved if H>1;
                        group k carries value at slot k*t).

        Token i_in = (n mod cap) within block b = n // cap.  For each group k,
        the value at slot k*t is placed into ciphertext ((i_in//t) - (k//H)) %
        (d/H) at slot k*t + (i_in % t).  This is the rotation-free customizable
        packing (rotate once by r, then d/H masked adds in the FHE version).
        """
        v = np.asarray(v_interleaved, dtype=np.float64)
        if v.shape != (self.N,):
            raise ValueError(f"v must have shape ({self.N},), got {v.shape}.")

        i = self.length
        b, i_in = divmod(i, self.cap)
        if b == len(self.blocks):
            self.blocks.append([np.zeros(self.N, dtype=np.float64) for _ in range(self.dh)])

        r = i_in % self.t
        Vr = np.roll(v, r)                       # rotate right by r (FHE-legal)
        for c in range(self.dh):
            # ciphertext c receives the group k with ((i_in//t) - (k//H)) % dh == c
            for k in range(self.d):
                if ((i_in // self.t) - (k // self.H)) % self.dh == c:
                    mask = np.zeros(self.N, dtype=np.float64)
                    mask[k * self.t + r] = 1.0
                    self.blocks[b][c] += Vr * mask
        self.length += 1

    def append_from_input(self, X, Wv_enc_or_raw, raw=True):
        """Project input X through Wv and append. Pass the raw Wv (raw=True)."""
        Wv = Wv_enc_or_raw
        # caller passes raw Wv; head reorder must match make_weights
        from cachemir_attention import head_perm
        perm = head_perm(self.d, self.H)
        v = vmm(
            preprocess_input(X, self.N, self.d),
            preprocess_weights(Wv[:, perm], self.N, self.d),
            self.N, self.d,
        )
        self.append(v)

    # ------------------------------------------------------------------
    # Read-back (verification / inspection)
    # ------------------------------------------------------------------
    def get_token(self, i):
        """
        Reconstruct the dense length-d V vector of cached token i, in the same
        head-interleaved order the projection used (group k = global dim
        perm[k]).
        """
        if not 0 <= i < self.length:
            raise IndexError(f"token {i} out of range [0, {self.length}).")
        b, i_in = divmod(i, self.cap)
        r = i_in % self.t
        block = self.blocks[b]
        out = np.zeros(self.d, dtype=np.float64)
        for k in range(self.d):
            c = ((i_in // self.t) - (k // self.H)) % self.dh
            out[k] = block[c][k * self.t + r]
        return out

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            f"VCache(N={self.N}, d={self.d}, H={self.H}, t={self.t}, "
            f"tokens={self.length}, blocks={self.num_blocks}, "
            f"ciphertexts={self.num_ciphertexts})"
        )


if __name__ == "__main__":
    from cachemir_attention import init_weights, init_input

    N, d = 8, 4
    Wv = init_weights(d, 7)

    cache = VCache(N, d)
    toks = [init_input(d, 20 + i) for i in range(5)]   # n = 5
    for x in toks:
        cache.append_from_input(x, Wv)

    print(cache)
    max_err = 0.0
    for i, x in enumerate(toks):
        rec = cache.get_token(i)
        ref = x @ Wv
        max_err = max(max_err, np.max(np.abs(rec - ref)))
    print("max abs error over all cached tokens:", max_err)