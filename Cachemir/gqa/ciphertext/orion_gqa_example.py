"""Example driver for the Orion ciphertext GQA code.

Copy this file next to your package files, adjust CONFIG, and run it from the
parent directory as a module or script.
"""

from pathlib import Path

from gqa.he_attention_orion import run_attention_gqa_he


def get_config_path(yml_name: str) -> str:
    # Change this to your local Orion checkout/config location.
    orion_path = Path.home() / "Git" / "orion"
    return str(orion_path / "configs" / yml_name)


if __name__ == "__main__":
    # Choose N to match the number of CKKS slots expected by your encoding.
    # For the small plaintext tests you used N=16/32/64. For real CKKS config,
    # use a slot count supported by the scheme, e.g. LogN=13 with conjugate-
    # invariant CKKS may expose 4096 or 8192 slots depending on Orion settings.
    run_attention_gqa_he(
        config_path=get_config_path("mlp.yml"),
        N=64,
        d=16,
        H=8,
        n_kv=1,
        n_prefill=5,
        input_level=None,
        encode_plaintexts=False,
        verify=True,
    )
