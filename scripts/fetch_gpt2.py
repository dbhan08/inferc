"""Download pre-exported GPT-2-small ONNX from HuggingFace Hub.

We use Xenova/gpt2 — a community ONNX export widely used by Transformers.js,
which already provides multiple flavors of the GPT-2 graph:

  decoder_model.onnx              — full forward pass over the sequence,
                                    NO past_key_values as graph I/O.
                                    inferc uses this; we manage KV cache
                                    ourselves at the executor level.
  decoder_with_past_model.onnx    — same model but with past_key_values
                                    exposed as graph I/O. Used in v2's
                                    Session 17 ORT comparison so ORT can
                                    do KV-cached decode too (apples-to-apples).
  decoder_model_merged.onnx       — both modes in one graph (we don't use this).

GPT-2-small specs: 124M params, 12 transformer blocks, 768 hidden dim,
12 attention heads, 50257 vocab, max position 1024.

Run via: poetry run python scripts/fetch_gpt2.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Xenova/gpt2"
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# (HF filename, local filename)
FILES_TO_FETCH = [
    ("onnx/decoder_model.onnx",           "gpt2.onnx"),
    ("onnx/decoder_with_past_model.onnx", "gpt2_with_past.onnx"),
]


def fetch_one(hf_filename: str, out_filename: str) -> int:
    out_path = MODELS / out_filename
    print(f"Fetching {REPO_ID}/{hf_filename} ...", flush=True)
    cached = hf_hub_download(repo_id=REPO_ID, filename=hf_filename)
    shutil.copy(cached, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  saved {out_path}  ({size_mb:.1f} MB)")
    return 0


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    for hf_fn, out_fn in FILES_TO_FETCH:
        if (MODELS / out_fn).exists():
            print(f"Already present: {MODELS / out_fn} — skipping download.")
            continue
        rc = fetch_one(hf_fn, out_fn)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
