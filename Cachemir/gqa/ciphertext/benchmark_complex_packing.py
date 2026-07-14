"""Benchmark: complex lane packing vs. the original real-only K/V cache layout.

Spawns one fresh subprocess per packing mode (so `peak_rss_mb`, an OS-level
high-water mark, reflects only that run rather than accumulating across
modes), then prints a side-by-side comparison of latency, peak RAM,
ct-ct multiplications (overall and per pipeline phase), and ciphertext
counts (K/V cache, attention-score, and output ciphertexts).

Usage:
    cd Cachemir && python -m gqa.ciphertext.benchmark_complex_packing
    cd Cachemir && python -m gqa.ciphertext.benchmark_complex_packing --worker --pack-complex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Cases to benchmark: (d, H, n_kv, n_prefill, level)
CASES = [
    dict(d=8, H=4, n_kv=2, n_prefill=5, level=7),
    dict(d=128, H=8, n_kv=4, n_prefill=9, level=7),
    dict(d=128, H=8, n_kv=4, n_prefill=20, level=7),
]


def _config_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here / "configs" / "gqa.yml")


def _run_worker(case: Dict[str, Any], pack_complex: bool) -> Dict[str, Any]:
    """Runs inside a fresh subprocess: executes one config in one packing
    mode and prints a single JSON line with latency/memory/op-count stats."""
    import orion
    from ..plaintext.counter import counter
    from .he_attention_orion import run_attention_gqa_he

    config_path = _config_path()
    scheme = orion.init_scheme(config_path)
    n_he = 1 << (scheme.params.get_logn() - 1)

    counter.reset()
    err_qkt, err_v, stats = run_attention_gqa_he(
        config_path=config_path, n_he=n_he, verbose=False,
        return_stats=True, pack_complex=pack_complex, **case,
    )
    result = {
        "err_qkt": err_qkt,
        "err_v": err_v,
        "pack_complex": stats["pack_complex"],
        "latency_s": stats["latency_s"],
        "peak_rss_mb": stats["peak_rss_mb"],
        "ciphertext_counts": stats["ciphertext_counts"],
        "ops_by_phase": stats["ops_by_phase"],
        "ops_total": stats["ops_total"],
    }
    print("BENCHMARK_RESULT_JSON " + json.dumps(result))


def _run_in_subprocess(case: Dict[str, Any], pack_complex: bool) -> Dict[str, Any]:
    args = [
        sys.executable, "-m", "gqa.ciphertext.benchmark_complex_packing",
        "--worker",
        "--pack-complex" if pack_complex else "--no-pack-complex",
        "--d", str(case["d"]), "--H", str(case["H"]), "--n_kv", str(case["n_kv"]),
        "--n_prefill", str(case["n_prefill"]), "--level", str(case["level"]),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if line.startswith("BENCHMARK_RESULT_JSON "):
            return json.loads(line[len("BENCHMARK_RESULT_JSON "):])
    raise RuntimeError(f"No benchmark result found in subprocess output:\n{proc.stdout}\n{proc.stderr}")


def _print_comparison(case: Dict[str, Any], complex_r: Dict[str, Any], real_r: Dict[str, Any]) -> None:
    print(
        f"\n=== d={case['d']} H={case['H']} n_kv={case['n_kv']} "
        f"n_prefill={case['n_prefill']} ==="
    )
    print(f"{'metric':<28s} {'complex':>14s} {'real-only':>14s} {'ratio':>8s}")

    def row(label: str, c_val: float, r_val: float, fmt: str = "{:.3f}") -> None:
        ratio = f"{(c_val / r_val):.2f}x" if r_val else "n/a"
        print(f"{label:<28s} {fmt.format(c_val):>14s} {fmt.format(r_val):>14s} {ratio:>8s}")

    row("latency: prefill (s)", complex_r["latency_s"]["prefill"], real_r["latency_s"]["prefill"])
    row("latency: decode step (s)", complex_r["latency_s"]["decode_step"], real_r["latency_s"]["decode_step"])
    row("latency: total (s)", complex_r["latency_s"]["total"], real_r["latency_s"]["total"])
    row("peak RSS (MB)", complex_r["peak_rss_mb"], real_r["peak_rss_mb"])
    row("kcache ciphertexts", complex_r["ciphertext_counts"]["kcache"], real_r["ciphertext_counts"]["kcache"], "{:.0f}")
    row("vcache ciphertexts", complex_r["ciphertext_counts"]["vcache"], real_r["ciphertext_counts"]["vcache"], "{:.0f}")
    row("att_cts ciphertexts", complex_r["ciphertext_counts"]["att_cts"], real_r["ciphertext_counts"]["att_cts"], "{:.0f}")
    row("O ciphertexts", complex_r["ciphertext_counts"]["o_cts"], real_r["ciphertext_counts"]["o_cts"], "{:.0f}")
    row("ct-ct mult (total)", complex_r["ops_total"]["ct_ct_mult"], real_r["ops_total"]["ct_ct_mult"], "{:.0f}")
    row("ct-pt mult (total)", complex_r["ops_total"]["ct_pt_mult"], real_r["ops_total"]["ct_pt_mult"], "{:.0f}")
    row("rotations (total)", complex_r["ops_total"]["rotations"], real_r["ops_total"]["rotations"], "{:.0f}")
    row("conjugations (total)", complex_r["ops_total"]["conjugations"], real_r["ops_total"]["conjugations"], "{:.0f}")

    print("  ct-ct mult by phase:")
    for phase in complex_r["ops_by_phase"]:
        c_ct = complex_r["ops_by_phase"][phase]["ct_ct_mult"]
        r_ct = real_r["ops_by_phase"][phase]["ct_ct_mult"]
        print(f"    {phase:<16s} complex={c_ct:<4d} real-only={r_ct:<4d}")


def main() -> None:
    for case in CASES:
        complex_r = _run_in_subprocess(case, pack_complex=True)
        real_r = _run_in_subprocess(case, pack_complex=False)
        _print_comparison(case, complex_r, real_r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pack-complex", dest="pack_complex", action="store_true")
    parser.add_argument("--no-pack-complex", dest="pack_complex", action="store_false")
    parser.set_defaults(pack_complex=True)
    parser.add_argument("--d", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--n_kv", type=int)
    parser.add_argument("--n_prefill", type=int)
    parser.add_argument("--level", type=int)
    args = parser.parse_args()

    if args.worker:
        _run_worker(
            dict(d=args.d, H=args.H, n_kv=args.n_kv, n_prefill=args.n_prefill, level=args.level),
            pack_complex=args.pack_complex,
        )
    else:
        main()
