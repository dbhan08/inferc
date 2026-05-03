"""Tokenize a fixed prompt for DistilBERT-SST2 and generate golden outputs.

Writes:
  models/input_ids.bin       — int64, shape [1, 128]
  models/attention_mask.bin  — int64, shape [1, 128]
  models/golden_logits.bin   — float32, shape [1, 2]  (from ORT CPU EP)
  models/inputs_meta.txt     — human-readable summary of what was generated

These binaries are the canonical fixtures that inferc's runtime (Session 5)
will be tested against. The numerical-correctness gate in v1 requires that
inferc's own logits match `golden_logits.bin` to within 1e-3 max-abs-diff.

Run via: poetry run python scripts/make_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

REPO_ID = "distilbert-base-uncased-finetuned-sst-2-english"
PROMPT = "The food at this restaurant was incredible."
MAX_LEN = 128
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ONNX_PATH = MODELS / "distilbert.onnx"


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"error: {ONNX_PATH} not found. Run scripts/fetch_distilbert.py first.",
              file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    encoded = tokenizer(
        PROMPT,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)            # [1, 128]
    attention_mask = encoded["attention_mask"].astype(np.int64)  # [1, 128]

    print(f"Prompt:        {PROMPT!r}")
    print(f"input_ids:     shape {input_ids.shape}, dtype {input_ids.dtype}")
    print(f"attention_mask shape {attention_mask.shape}, dtype {attention_mask.dtype}")

    input_ids.tofile(MODELS / "input_ids.bin")
    attention_mask.tofile(MODELS / "attention_mask.bin")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print("\nORT session loaded.")
    print("  inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("  outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

    outs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    logits = outs[0].astype(np.float32)
    print(f"\nLogits: shape {logits.shape}")
    print(f"  raw:        {logits.ravel().tolist()}")
    pred = int(np.argmax(logits))
    label = ["NEGATIVE", "POSITIVE"][pred]
    softmax = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    print(f"  prediction: {label}  (softmax: {softmax.ravel().tolist()})")

    logits.tofile(MODELS / "golden_logits.bin")

    meta = MODELS / "inputs_meta.txt"
    meta.write_text(
        f"prompt: {PROMPT!r}\n"
        f"max_len: {MAX_LEN}\n"
        f"input_ids: int64 [1, {MAX_LEN}]\n"
        f"attention_mask: int64 [1, {MAX_LEN}]\n"
        f"golden_logits: float32 [1, 2]\n"
        f"prediction: {label}\n"
        f"raw_logits: {logits.ravel().tolist()}\n"
    )
    print(f"\nSaved fixtures to {MODELS}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
