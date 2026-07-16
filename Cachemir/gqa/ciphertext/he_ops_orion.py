"""Orion ciphertext primitives for the GQA attention code."""

from __future__ import annotations
 
from typing import Any, Iterable, List, Tuple
 
import torch
import orion
 
from ..plaintext.dims import GQADims, check_dims
from ..plaintext.counter import counter

# Orion's ct.roll(k) rotates opposite to torch.roll(x, k), so every rotation
# below calls .roll(-shift) to stay consistent with the plaintext simulator's
# torch.roll(x, -shift) calls.


def encode_pt(vec: torch.Tensor, level: int) -> Any:
    """Encode a torch slot vector as an Orion plaintext at `level`."""
    vec = torch.as_tensor(vec, dtype=torch.float64)
    return orion.encode(vec, level)


def ct_zero(n_he: int, level: int) -> Any:
    """Create an encrypted zero vector at `level`."""
    zero_vec = torch.zeros(n_he, dtype=torch.float64)
    return orion.encrypt(orion.encode(zero_vec, level))

def vmm_kv(
    X_enc_chunks_ct: List[Any],
    W_enc_chunks: List[torch.Tensor],
    dims: GQADims,
    level: int,
    pos: int = 0,
    ) -> Tuple[Any, int]:
    """Compact GQA K/V or Q projection over encrypted sparse chunks."""
    if len(X_enc_chunks_ct) != len(W_enc_chunks):
        raise ValueError(
            f"X chunks ({len(X_enc_chunks_ct)}) and W chunks ({len(W_enc_chunks)}) differ."
        )
    if not X_enc_chunks_ct:
        raise ValueError("X_enc_chunks_ct cannot be empty.")


    mask = torch.zeros(dims.n_he, dtype= torch.float64)
    mask[pos::dims.t_p] = 1.0
    acc_level = level - 1
    mask_pt = orion.encode(mask, acc_level)

    out = None
    for Xc, Wc in zip(X_enc_chunks_ct, W_enc_chunks):
        counter.ct_pt_mult += 1
        acc = Xc * Wc #_maybe_encode_plain(Wc, pt_level, encode_plaintexts)

        step, i = 1, 0
        while step < dims.t_p:
            counter.rotations += 1
            acc = acc + acc.roll(-step if (pos >> i) & 1 else step)
            step *= 2
            i += 1

        counter.ct_pt_mult += 1
        masked = acc * mask_pt
 
        out = masked if out is None else out + masked

    return out, level-2


def block_replicate(v_ct: Any, t_p: int) -> Any:
    """Replicate structural blocks with power-of-two rotate-add steps."""
    out = v_ct
    step = 1
    while step < t_p:
        counter.rotations += 1
        out = out + out.roll(-step)
        step *= 2
    return out



def decrypt_cipher_list(ct_list: List[Any], length: int) -> List[torch.Tensor]:
    """Decrypt and decode a list of Orion ciphertexts.
    Returns a list of numpy vectors.
    """
    decoded_list: List[torch.Tensor] = []
 
    for ct in ct_list:
        v = ct.decrypt().decode()
 
        # CKKS may have tiny imaginary noise.
        if hasattr(v, "real"):
            v = v.real
 
        v = torch.as_tensor(v, dtype=torch.float64)
        decoded_list.append(v[:length])
 
    return decoded_list