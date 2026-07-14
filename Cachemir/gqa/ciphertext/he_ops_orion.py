"""Orion ciphertext primitives for the GQA attention code."""

from __future__ import annotations
 
from typing import Any, Iterable, List, Tuple
 
import torch
import orion
 
from ..plaintext.dims import GQADims, check_dims
from ..plaintext.counter import counter
 
# Set this to False if Orion's rotate/rot API has torch.roll's sign convention.
ROTATION_IS_LEFT = True


def encode_pt(vec: torch.Tensor, level: int) -> Any:
    """Encode a torch slot vector as an Orion plaintext at `level`."""
    vec = torch.as_tensor(vec, dtype=torch.float64)
    return orion.encode(vec, level)

# def _rotate_raw(ct: Any, amount: int) -> Any:
#     """Call the rotation method exposed by the installed Orion ciphertext."""
#     # Bound methods on the ciphertext object.
#     for name in ("rotate", "rot", "roll"):
#         meth = getattr(ct, name, None)
#         if callable(meth):
#             return meth(amount)

#     # Module-level helpers, depending on Orion version.
#     for name in ("rotate", "rot", "roll"):
#         fn = getattr(orion, name, None)
#         if callable(fn):
#             return fn(ct, amount)

#     raise AttributeError(
#         "Could not find an Orion ciphertext rotation API. Tried ct.rotate/ct.rot/ct.roll "
#         "and orion.rotate/orion.rot/orion.roll. Adjust _rotate_raw() for your Orion build."
#     )


def ct_roll(ct: Any, shift: int) -> Any:
    """Ciphertext equivalent of torch.roll(ct, shift)."""
    if shift == 0:
        return ct
    amount = -shift if ROTATION_IS_LEFT else shift
    return ct.roll(amount)

def ct_zero(n_he: int, level: int) -> Any:
    """Create an encrypted zero vector at `level`."""
    zero_vec = torch.zeros(n_he, dtype=torch.float64)
    return orion.encrypt(orion.encode(zero_vec, level))


def ct_add_many(values: Iterable[Any]) -> Any:
    """Sum a list of ciphertexts. Pure ct-ct addition -- level-neutral, no
    plaintexts involved, so no `level` argument is needed."""
    vals = list(values)
    if not vals:
        raise ValueError("ct_add_many() needs at least one ciphertext.")
    acc = vals[0] * 0.0
    for v in vals[1:]:
        acc = acc + v
    return acc


def vmm_kv(
    X_enc_chunks_ct: List[Any],
    W_enc_chunks: List[torch.Tensor],
    dims: GQADims,
    level: int,
    pos: int = 0,
    ) -> Tuple[Any, int]:
    """Compact GQA K/V or Q projection over encrypted sparse chunks.

    `pos` ranges over `[0, 2*dims.t_p)`. Positions `< t_p` land in the real
    part of physical lane `pos` (as before). Positions `>= t_p` land in the
    *imaginary* part of physical lane `pos - t_p`, packing a second,
    independent token into the same ciphertext slots via CKKS's unused
    imaginary half. This lets one ciphertext hold `2*t_p` tokens instead of
    `t_p`, halving cache ciphertext count for the same sequence length.
    """
    if len(X_enc_chunks_ct) != len(W_enc_chunks):
        raise ValueError(
            f"X chunks ({len(X_enc_chunks_ct)}) and W chunks ({len(W_enc_chunks)}) differ."
        )
    if not X_enc_chunks_ct:
        raise ValueError("X_enc_chunks_ct cannot be empty.")

    is_imag = pos >= dims.t_p
    phys_pos = pos - dims.t_p if is_imag else pos

    mask = torch.zeros(dims.n_he, dtype=torch.complex128 if is_imag else torch.float64)
    mask[phys_pos::dims.t_p] = 1j if is_imag else 1.0
    acc_level = level - 1
    mask_pt = orion.encode(mask, acc_level)

    out = None
    for Xc, Wc in zip(X_enc_chunks_ct, W_enc_chunks):
        counter.ct_pt_mult += 1
        acc = Xc * Wc #_maybe_encode_plain(Wc, pt_level, encode_plaintexts)

        step, i = 1, 0
        while step < dims.t_p:
            counter.rotations += 1
            acc = acc + ct_roll(acc, +step if (phys_pos >> i) & 1 else -step)
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
        out = out + ct_roll(out, step)
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