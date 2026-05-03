"""Download a pre-exported DistilBERT ONNX from HuggingFace Hub.

We use the SST-2 fine-tuned variant: identical encoder body to vanilla
DistilBERT-base, plus a small 2-class classification head. Makes the
demo concrete (positive vs negative sentiment).

Run via: poetry run python scripts/fetch_distilbert.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "optimum/distilbert-base-uncased-finetuned-sst-2-english"
FILENAME = "model.onnx"
OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "distilbert.onnx"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {REPO_ID}/{FILENAME} ...", flush=True)
    cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    shutil.copy(cached, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
