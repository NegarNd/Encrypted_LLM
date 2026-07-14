# Encrypted LLM

This repository contains experiments for encrypted LLM decoder inference, with a current focus on GQA attention and KV-cache computation.

The code is organized into two versions:

```text
gqa/
  plaintext/    # Plaintext simulator / reference implementation
  ciphertext/   # Orion-based ciphertext implementation
```

The plaintext version is used to check the packing, encoding, and attention computation without FHE. The ciphertext version uses Orion and requires Orion to be installed first.

## Requirements

- Python 3.10
- NumPy
- PyTorch
- Orion FHE library

Orion is required for the ciphertext implementation.
Orion repository's installation: <https://github.com/baahl-nyu/orion>

## Setup

### 1. Create or Activate a Python Environment

Using `mamba`:

```bash
mamba create -n orion python=3.10
mamba activate orion
```

or using `conda`:

```bash
conda create -n orion python=3.10
conda activate orion
```

### 2. Check Orion

After installing Orion using the instructions from the Orion repository, check that it is available in your current environment:

```bash
python -c "import orion; print('Orion import works')"
```

### 3. Clone This Repository

```bash
git clone https://github.com/NegarNd/Encrypted_LLM.git
cd Encrypted_LLM
```

## Running the Code

Run all commands from the repository root:

```bash
cd Encrypted_LLM/Cachemir
```

### 4. Run the Plaintext GQA Verification

```bash
python -m gqa.plaintext.verify_gqa
```

This runs the plaintext GQA implementation and checks the packed attention computation against the reference computation.

### 6. Run the Orion Ciphertext GQA Verification

Make sure the Orion environment is activated:

```bash
mamba activate orion
```

Then run:

```bash
python -m gqa.ciphertext.verify_orion
```

This runs the ciphertext implementation using Orion and compares the decrypted output against the plaintext reference.

## Verification Output

The verification scripts compare the packed implementation against a reference attention computation.

A successful run may print errors like:

```text
qkt_err=1.72e-04, softmaxv_err=1.44e-02
```

`qkt_err` is the error in the `QK^T` attention scores.

`softmaxv_err` is the error in the final attention output after applying the attention weights to `V`.

For CKKS/FHE runs, small numerical differences are expected because CKKS is approximate.

## Complex Lane Packing: Validation Tests and Benchmarks

The ciphertext implementation supports two K/V cache layouts, selected via
the `pack_complex` flag threaded through `run_attention_gqa_he` (and the
`HEKCache`/`HEVCache`/`softmax_v_gqa_he` helpers it calls):

- `pack_complex=True` (default): packs 2 tokens per ciphertext using the
  real and imaginary parts of each CKKS slot, roughly halving K/V cache
  ciphertext count and ciphertext--ciphertext (ct-ct) multiplications as
  sequence length grows.
- `pack_complex=False`: the original real-only layout (1 token per
  ciphertext slot), kept for direct comparison.

See `Cachemir/gqa/ciphertext/docs/complex_packing.pdf` for the full
mathematical write-up and benchmark results.

### 7. Run the Complex-Packing Validation Tests

Make sure the Orion environment is activated, then from `Cachemir/`:

```bash
python -m pytest gqa/ciphertext/test_complex_packing.py -v
```

This runs the same encrypted decoding step with `pack_complex=True` and
`pack_complex=False` (identical seeds/config) and asserts that:

- both modes agree with the plaintext reference within CKKS tolerance,
- the two modes' decrypted attention scores and outputs match each other
  (packing must not change the computed values),
- the complex-packed run uses strictly fewer K/V cache ciphertexts and
  ct-ct multiplications, and only the complex-packed run performs any
  homomorphic conjugations.

You can also run it as a plain script instead of via pytest:

```bash
python -m gqa.ciphertext.test_complex_packing
```

### 8. Run the Complex vs. Real-Only Performance Benchmark

From `Cachemir/`:

```bash
python -m gqa.ciphertext.benchmark_complex_packing
```

This spawns one fresh subprocess per configuration per packing mode (so
that peak-RAM measurements aren't contaminated across modes) and prints a
side-by-side comparison table for each configuration, including:

- latency (prefill / decode step / total, seconds),
- peak resident set size (RSS, MB),
- ciphertext counts affected by packing (`kcache`, `vcache`, `att_cts`,
  `O` ciphertexts),
- op counts (ct-ct multiplications, ct-pt multiplications, rotations,
  conjugations), both in total and broken down per pipeline phase
  (`q_projection`, `k_projection`, `v_projection`, `qkt`, `scores_v`).

To benchmark a single configuration/mode directly (e.g. for scripting or
profiling), invoke the worker mode directly:

```bash
python -m gqa.ciphertext.benchmark_complex_packing --worker --pack-complex \
    --d 128 --H 8 --n_kv 4 --n_prefill 20 --level 7

python -m gqa.ciphertext.benchmark_complex_packing --worker --no-pack-complex \
    --d 128 --H 8 --n_kv 4 --n_prefill 20 --level 7
```

Each invocation prints one `BENCHMARK_RESULT_JSON {...}` line with the
same metrics in machine-readable form.

