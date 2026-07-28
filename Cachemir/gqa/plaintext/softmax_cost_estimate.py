"""Estimate the cost of integrating softmax_jh.py's AppExp/AppInv softmax
into the decode pipeline, and recompute the pipeline-level impact of the two
complex-packing strategies from vmm_complex_pack_compare.py once that cost
is included.

softmax_jh.py (Cachemir/softmax_jh.py) is currently an orphaned draft: its
relative imports (`from .counter import counter`, etc.) assume it lives
inside the gqa.plaintext package, so `import softmax_jh` fails as-is. This
module loads it via importlib with `__package__` forced to "gqa.plaintext",
so its relative imports resolve against the REAL counter/dims/ops already
used by the rest of this simulator -- no code is copied or duplicated, and
if softmax_jh.py's algorithm changes this script picks up the change too.

softmax_jh.py itself only tallies `counter.rotations` (for its fold steps);
it never counts the AppExp/AppInv multiplications. To get an accurate,
non-manual count of those, this module temporarily monkey-patches
`torch.Tensor.__mul__`/`__pow__` for the duration of one `app_softmax_gqa`
call: an operand with numel()==1 (or a plain Python int/float) is a
plaintext constant -> ct-pt; two full-size tensors multiplied together is
ct-ct. `e ** delta1` is counted as log2(delta1) ct-ct multiplications
(repeated squaring), matching softmax_jh.py's own validation that delta1
must be a power of two.

Run: `cd Cachemir && python -m gqa.plaintext.softmax_cost_estimate`
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Dict

import torch

from .cache import KCache
from .attention import vmm_q_gqa, qkt_gqa
from .encoding import expand_sparse_input_kv_plain, make_sparse_input_kv
from .ops import vmm_kv
from .counter import counter
from .verify_gqa import CASES
from .vmm_complex_pack_compare import (
    _build_case,
    run_baseline,
    run_attn_complex,
    run_vmm_complex,
    LATENCY_US,
    estimated_latency_us,
    EXTRA,
)


def _load_softmax_jh():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "..", "softmax_jh.py"))
    spec = importlib.util.spec_from_file_location("gqa.plaintext._softmax_jh_ext", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "gqa.plaintext"  # makes its `.counter`/`.dims`/`.ops` imports resolve
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOFTMAX_JH = _load_softmax_jh()


def _is_scalar_like(v) -> bool:
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, torch.Tensor):
        return v.numel() == 1
    return False


class _MulPowTally:
    def __init__(self):
        self.ct_ct = 0
        self.ct_pt = 0


class count_mults:
    """Context manager: classify every torch.Tensor multiply as ct-ct/ct-pt."""

    def __init__(self, tally: _MulPowTally):
        self.tally = tally

    def __enter__(self) -> _MulPowTally:
        tally = self.tally
        self._orig_mul = torch.Tensor.__mul__
        self._orig_pow = torch.Tensor.__pow__

        def new_mul(a, b):
            if _is_scalar_like(a) or _is_scalar_like(b):
                tally.ct_pt += 1
            else:
                tally.ct_ct += 1
            return self._orig_mul(a, b)

        def new_pow(a, exponent):
            if isinstance(exponent, int) and exponent > 1:
                tally.ct_ct += int(round(math.log2(exponent)))
            return self._orig_pow(a, exponent)

        torch.Tensor.__mul__ = new_mul
        torch.Tensor.__pow__ = new_pow
        return tally

    def __exit__(self, *exc):
        torch.Tensor.__mul__ = self._orig_mul
        torch.Tensor.__pow__ = self._orig_pow


def softmax_cost(dims, att_cts: torch.Tensor, n_tokens: int) -> Dict[str, int]:
    """Real op count for one app_softmax_gqa call (rotations authoritative
    from the real code; ct-ct/ct-pt derived by monkey-patched instrumentation)."""
    counter.reset()
    tally = _MulPowTally()
    with count_mults(tally):
        SOFTMAX_JH.app_softmax_gqa(att_cts, dims, n_tokens)
    rot = counter.rotations
    counter.reset()
    return {"rotations": rot, "ct_pt_mult": tally.ct_pt, "ct_ct_mult": tally.ct_ct, "conjugations": 0}


def _read_and_reset() -> Dict[str, int]:
    tally = {
        "rotations": counter.rotations,
        "ct_pt_mult": counter.ct_pt_mult,
        "ct_ct_mult": counter.ct_ct_mult,
        "conjugations": EXTRA["conjugations"],
    }
    counter.reset()
    EXTRA["conjugations"] = 0
    return tally


def _add(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: a[k] + b.get(k, 0) for k in a}


def main() -> None:
    sums = {
        "baseline": {"rotations": 0, "ct_pt_mult": 0, "ct_ct_mult": 0, "conjugations": 0},
        "attn_complex": {"rotations": 0, "ct_pt_mult": 0, "ct_ct_mult": 0, "conjugations": 0},
        "vmm_complex": {"rotations": 0, "ct_pt_mult": 0, "ct_ct_mult": 0, "conjugations": 0},
        "softmax_only": {"rotations": 0, "ct_pt_mult": 0, "ct_ct_mult": 0, "conjugations": 0},
        "unpack_overhead": {"rotations": 0, "ct_pt_mult": 0, "ct_ct_mult": 0, "conjugations": 0},
    }

    for N, d, H, n_kv, n_prefill in CASES:
        dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new = _build_case(N, d, H, n_kv, n_prefill)

        counter.reset()
        run_baseline(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new)
        sums["baseline"] = _add(sums["baseline"], _read_and_reset())

        run_attn_complex(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new)
        sums["attn_complex"] = _add(sums["attn_complex"], _read_and_reset())

        run_vmm_complex(dims, Wq_enc_list, Wk_enc, Wv_enc, toks, x_new)
        sums["vmm_complex"] = _add(sums["vmm_complex"], _read_and_reset())

        # Real attention scores (real-domain, as softmax_jh.py expects) to
        # drive an accurate op count for one softmax call on this case.
        kcache = KCache(dims.n_he, dims.d_kv)
        for x in toks:
            xc = expand_sparse_input_kv_plain(make_sparse_input_kv(x, dims), dims)
            kcache.append(vmm_kv(xc, Wk_enc, dims, pos=len(kcache) % dims.t_p))
        xc_new = expand_sparse_input_kv_plain(make_sparse_input_kv(x_new, dims), dims)
        Q_cts = vmm_q_gqa(xc_new, Wq_enc_list, dims)
        att_cts = qkt_gqa(Q_cts, kcache.ciphertexts, dims)
        n_tokens = len(kcache) + 1

        sums["softmax_only"] = _add(sums["softmax_only"], softmax_cost(dims, att_cts, n_tokens))

        # attn_complex packs scores as 2 tokens/ciphertext; softmax_jh.py
        # only understands real-domain scores, so attn_complex must unpack
        # (1 shared conjugation + 2 ct-pt scalar mults per complex block)
        # before it can feed softmax -- vmm_complex needs no such unpack
        # since its scores were never complex-packed.
        n_b_complex = max(1, -(-n_tokens // (2 * dims.t_p)))
        sums["unpack_overhead"]["conjugations"] += n_b_complex
        sums["unpack_overhead"]["ct_pt_mult"] += 2 * n_b_complex

    print("=== Summed op counts across all 21 CASES ===")
    for name in ("baseline", "attn_complex", "vmm_complex", "softmax_only", "unpack_overhead"):
        t = sums[name]
        print(
            f"  {name:16s} rot={t['rotations']:6d} ct-pt={t['ct_pt_mult']:6d} "
            f"ct-ct={t['ct_ct_mult']:6d} conj={t['conjugations']:4d} "
            f"est_latency={estimated_latency_us(t)/1000:9.2f} ms"
        )

    print("\n=== Pipeline totals WITHOUT softmax (from vmm_complex_pack_compare.py) ===")
    lat_base_no_smax = estimated_latency_us(sums["baseline"])
    lat_attn_no_smax = estimated_latency_us(sums["attn_complex"])
    lat_vmm_no_smax = estimated_latency_us(sums["vmm_complex"])
    print(f"  baseline     : {lat_base_no_smax/1000:9.2f} ms")
    print(f"  attn_complex : {lat_attn_no_smax/1000:9.2f} ms  ({100*(1-lat_attn_no_smax/lat_base_no_smax):+.1f}%)")
    print(f"  vmm_complex  : {lat_vmm_no_smax/1000:9.2f} ms  ({100*(1-lat_vmm_no_smax/lat_base_no_smax):+.1f}%)")

    print("\n=== Pipeline totals WITH softmax integrated ===")
    lat_smax = estimated_latency_us(sums["softmax_only"])
    lat_unpack = estimated_latency_us(sums["unpack_overhead"])

    total_base = lat_base_no_smax + lat_smax
    total_attn = lat_attn_no_smax + lat_unpack + lat_smax  # attn_complex must unpack before softmax
    total_vmm = lat_vmm_no_smax + lat_smax  # scores were never packed -> no unpack needed

    print(f"  softmax cost added to all three: {lat_smax/1000:9.2f} ms")
    print(f"  attn_complex unpack overhead (extra, only for attn_complex): {lat_unpack/1000:9.2f} ms")
    print(f"  baseline + softmax     : {total_base/1000:9.2f} ms")
    print(
        f"  attn_complex + unpack + softmax : {total_attn/1000:9.2f} ms  "
        f"({100*(1-total_attn/total_base):+.1f}% vs baseline+softmax)"
    )
    print(
        f"  vmm_complex + softmax            : {total_vmm/1000:9.2f} ms  "
        f"({100*(1-total_vmm/total_base):+.1f}% vs baseline+softmax)"
    )

    print(
        f"\n  softmax is {100*lat_smax/total_base:.1f}% of the baseline+softmax total "
        f"(was {0.0:.1f}% before integration)."
    )


if __name__ == "__main__":
    main()
