"""Generate a broad end-to-end correctness suite with ORT golden outputs.

The original fixtures (make_inputs.py, make_gpt2_inputs.py) use ONE fixed
sentence per model. This script produces a suite that is not hand-picked:

DistilBERT-SST2 (models/e2e_suite/distilbert/):
  - 6 curated sentences spanning length, sentiment, negation, numbers,
    punctuation, and truncation at max length;
  - 6 random-token sequences (seeded RNG) of random length in [4, 127],
    wrapped in [CLS] ... [SEP];
  - padded lengths of 128, 64 and 32 so dynamic sequence shapes are covered.

GPT-2 (models/e2e_suite/gpt2/):
  - 5 prompts from 1 to ~60 tokens plus one seeded random-token prompt;
  - ORT logits at every position and a 16-token ORT greedy continuation
    (full-recompute, no cache) as golden.

Every case is written as raw little-endian binaries plus one manifest.json
per model. tests/e2e_suite_test.cc walks the manifest.

Run: <venv>/bin/python scripts/make_e2e_suite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT = MODELS / "e2e_suite"
SEED = 20260923

DISTILBERT_REPO = "distilbert-base-uncased-finetuned-sst-2-english"
GPT2_REPO = "gpt2"

DISTILBERT_SENTENCES = [
    "Great.",
    "I did not like this movie at all, and I would not recommend it.",
    "The plot was fine but the pacing dragged; still, 3 out of 5 stars.",
    "What a waste of 2 hours and $15!!! Never again...",
    "An understated, quietly devastating film that rewards patience.",
    ("Although the first act meanders and the supporting cast is uneven, the film "
     "eventually finds its footing in a second half that is by turns funny, tense, "
     "and unexpectedly moving; by the time the credits roll it has earned every one "
     "of its emotional beats, even if the ending is a touch too neat and the score "
     "occasionally overwhelms the dialogue, which is a shame because the writing is "
     "sharp, the performances are committed, and the cinematography is gorgeous "
     "throughout, especially in the long night sequences that close the picture."),
]

GPT2_PROMPTS = [
    "Hello",
    "The capital of France is",
    "In 1969, astronauts landed on the Moon and",
    ("Once upon a time, in a small village nestled between two mountains, there "
     "lived an old clockmaker who had never once been late for anything in his "
     "life. Every morning at exactly"),
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
]


def write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def make_distilbert() -> int:
    onnx_path = MODELS / "distilbert.onnx"
    if not onnx_path.exists():
        print(f"error: {onnx_path} missing; run scripts/fetch_distilbert.py", file=sys.stderr)
        return 1
    tok = AutoTokenizer.from_pretrained(DISTILBERT_REPO)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(SEED)
    out_dir = OUT / "distilbert"
    cases = []

    def add_case(name: str, ids: np.ndarray, mask: np.ndarray, note: str) -> None:
        logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0].astype(np.float32)
        write(out_dir / f"{name}_ids.bin", ids.astype(np.int64))
        write(out_dir / f"{name}_mask.bin", mask.astype(np.int64))
        write(out_dir / f"{name}_logits.bin", logits)
        cases.append({
            "name": name, "seq_len": int(ids.shape[1]),
            "n_real_tokens": int(mask.sum()), "note": note,
            "golden_logits": logits.ravel().tolist(),
        })

    pad_lens = [128, 64, 32]
    for i, text in enumerate(DISTILBERT_SENTENCES):
        L = pad_lens[i % len(pad_lens)]
        enc = tok(text, padding="max_length", truncation=True, max_length=L, return_tensors="np")
        add_case(f"text{i}", enc["input_ids"], enc["attention_mask"], f"curated, pad={L}: {text[:40]!r}")

    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    n_special = 999  # ids below this are [unused]/special in bert-base-uncased vocab
    for i in range(6):
        L = pad_lens[i % len(pad_lens)]
        n_real = int(rng.integers(4, L))  # includes CLS/SEP
        body = rng.integers(n_special, tok.vocab_size, size=n_real - 2)
        ids = np.full((1, L), pad, dtype=np.int64)
        ids[0, 0] = cls
        ids[0, 1:n_real - 1] = body
        ids[0, n_real - 1] = sep
        mask = np.zeros((1, L), dtype=np.int64)
        mask[0, :n_real] = 1
        add_case(f"rand{i}", ids, mask, f"random tokens, seed={SEED}, n_real={n_real}, pad={L}")

    (out_dir / "manifest.json").write_text(json.dumps({"seed": SEED, "cases": cases}, indent=1))
    print(f"DistilBERT: {len(cases)} cases -> {out_dir}")
    return 0


def make_gpt2() -> int:
    onnx_path = MODELS / "gpt2.onnx"
    if not onnx_path.exists():
        print(f"error: {onnx_path} missing; run scripts/fetch_gpt2.py", file=sys.stderr)
        return 1
    tok = AutoTokenizer.from_pretrained(GPT2_REPO)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    rng = np.random.default_rng(SEED)
    out_dir = OUT / "gpt2"
    cases = []
    n_greedy = 16

    def run(ids: np.ndarray) -> np.ndarray:
        feed = {"input_ids": ids}
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.ones_like(ids, dtype=np.int64)
        if "position_ids" in input_names:
            feed["position_ids"] = np.arange(ids.shape[1], dtype=np.int64).reshape(1, -1)
        return sess.run(None, feed)[0].astype(np.float32)

    def add_case(name: str, ids: np.ndarray, note: str) -> None:
        logits = run(ids)
        cur = ids.copy()
        gen = []
        for _ in range(n_greedy):
            nxt = int(np.argmax(run(cur)[0, -1]))
            gen.append(nxt)
            cur = np.concatenate([cur, np.array([[nxt]], dtype=np.int64)], axis=1)
        write(out_dir / f"{name}_ids.bin", ids.astype(np.int64))
        write(out_dir / f"{name}_logits.bin", logits)
        write(out_dir / f"{name}_greedy.bin", np.array(gen, dtype=np.int64))
        cases.append({
            "name": name, "seq_len": int(ids.shape[1]), "vocab": int(logits.shape[2]),
            "n_greedy": n_greedy, "note": note,
            "greedy_tokens": gen, "greedy_text": tok.decode(gen),
        })

    for i, text in enumerate(GPT2_PROMPTS):
        ids = tok(text, return_tensors="np")["input_ids"].astype(np.int64)
        add_case(f"text{i}", ids, f"prompt: {text[:40]!r}")
    n = int(rng.integers(8, 48))
    ids = rng.integers(0, tok.vocab_size, size=(1, n)).astype(np.int64)
    add_case("rand0", ids, f"random tokens, seed={SEED}, n={n}")

    (out_dir / "manifest.json").write_text(json.dumps({"seed": SEED, "cases": cases}, indent=1))
    print(f"GPT-2: {len(cases)} cases -> {out_dir}")
    return 0


if __name__ == "__main__":
    rc = make_distilbert()
    rc = make_gpt2() or rc
    sys.exit(rc)
