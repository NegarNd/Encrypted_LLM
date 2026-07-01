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
