"""
kv_cache.py
===========
Interleaved K cache for Cachemir (Figure 5 of arXiv:2602.11470).

The QKV projection (cachemir_kernel.vmm) emits each token's K vector in the
interleaved format: K[j] sits in slot j*t, the other slots being scratch.
The K cache packs t = N/d such tokens into a single ciphertext, interleaved:

    [k00, k10, k01, k11, k02, k12, k03, k13]      (N=8, d=4, t=2)

where k_ij is dimension j of token i, stored at slot  j*t + i.

Because token i's K is already interleaved (K[j] at slot j*t), appending it as
the i-th token in a group is just a right-rotation by i followed by an add -
the "customizable interleaved packing without extra rotations" of Figure 5(d).
A new ciphertext is started whenever the current group of t tokens fills up;
the cache therefore holds ceil(n'/t) ciphertexts for n' tokens.

This module is stateful by design (the cache grows across decoding steps), so
it lives in a class, separate from the stateless kernel primitives.
"""

import numpy as np

from cachemir_attention import (
    check_dims,
    preprocess_input,
    preprocess_weights,
    vmm,
)


class KCache:
    """
    Interleaved K cache that packs t = N/d tokens per ciphertext.

    Parameters
    ----------
    N : int   polynomial degree / slot count (power of two)
    d : int   feature dimension
    """

    def __init__(self, N, d):
        check_dims(N, d)
        self.N = N
        self.d = d
        self.t = N // d
        self.ciphertexts = []   # list of length-N arrays, each packing <= t tokens
        self.length = 0         # number of tokens currently cached (n')

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_ciphertexts(self):
        """Number of ciphertexts holding the cache = ceil(n'/t)."""
        return len(self.ciphertexts)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def append(self, k_positioned):
        """
        Append one token's K vector to the cache (Figure 5d).

        Parameters
        ----------
        k_positioned : np.ndarray, shape (N,)
            The token's K vector already produced at the correct group offset
            pos = (n' % t) - i.e. via vmm(..., pos=n'%t).  No rotation is needed
            here; the placement rotation is fused into the projection's reduce
            step.  When the offset is 0 a new ciphertext is started.
        """
        k = np.asarray(k_positioned, dtype=np.float64)
        if k.shape != (self.N,):
            raise ValueError(f"k must have shape ({self.N},), got {k.shape}.")

        pos = self.length % self.t          # slot offset within the group
        if pos == 0:
            self.ciphertexts.append(np.zeros(self.N, dtype=np.float64))
        self.ciphertexts[-1] += k           # already at offset pos - no roll
        self.length += 1

    def append_from_input(self, X, Wk):
        """
        Convenience: project input X through weight Wk at the correct group
        offset and append the result (placement rotation fused into vmm).
        """
        pos = self.length % self.t
        k = vmm(
            preprocess_input(X, self.N, self.d),
            preprocess_weights(Wk, self.N, self.d),
            self.N,
            self.d,
            pos=pos,
        )
        self.append(k)

    # ------------------------------------------------------------------
    # Read-back (for verification / inspection)
    # ------------------------------------------------------------------

    def get_token(self, i):
        """
        Reconstruct the dense length-d K vector of cached token i.

        token i lives in ciphertext i//t at slots j*t + (i % t) for j in 0..d-1.
        """
        if not 0 <= i < self.length:
            raise IndexError(f"token {i} out of range [0, {self.length}).")
        ct = self.ciphertexts[i // self.t]
        off = i % self.t
        return np.array([ct[j * self.t + off] for j in range(self.d)], dtype=np.float64)

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            f"KCache(N={self.N}, d={self.d}, t={self.t}, "
            f"tokens={self.length}, ciphertexts={self.num_ciphertexts})"
        )


if __name__ == "__main__":
    from cachemir_attention import init_weights

    N, d = 8, 4
    rng = np.random.default_rng(0)
    Wk = init_weights(d, 5)

    cache = KCache(N, d)
    tokens = [rng.standard_normal(d) for _ in range(5)]   # n' = 5
    for x in tokens:
        cache.append_from_input(x, Wk)

    print(cache)
    print(f"expected ciphertexts = ceil(5/{cache.t}) = {-(-5 // cache.t)}")

    max_err = 0.0
    for i, x in enumerate(tokens):
        rec = cache.get_token(i)
        ref = x @ Wk
        max_err = max(max_err, np.max(np.abs(rec - ref)))
    print("max abs error over all cached tokens:", max_err)