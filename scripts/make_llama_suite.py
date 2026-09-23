"""ORT golden outputs for the Llama-family end-to-end gate.

Two models:
  models/llama15M.onnx            Xenova/llama2.c-stories15M (older export: If nodes,
                                  Less, Sigmoid; inputs input_ids + attention_mask)
  models/tinyllama/onnx/model.onnx Xenova/TinyLlama-1.1B-Chat-v1.0 fp32 (external
                                  weights; inputs input_ids, attention_mask,
                                  position_ids, past_key_values.*)

For each prompt: ORT prefill logits at every position, plus an n-token greedy
continuation computed by full recompute (no cache). Written under
models/e2e_suite/<model>/ with a manifest.json; tests/llama_test.cc reads it.

Run: <venv>/bin/python scripts/make_llama_suite.py [--skip-tinyllama]
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

PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "Lily and Tom went to the park. They saw a big",
]


def write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def run_suite(tag: str, onnx_path: Path, tok_repo: str, n_greedy: int, with_past_inputs: bool) -> int:
    if not onnx_path.exists():
        print(f"skip {tag}: {onnx_path} missing", file=sys.stderr)
        return 0
    tok = AutoTokenizer.from_pretrained(tok_repo)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}
    out_dir = OUT / tag
    rng = np.random.default_rng(SEED)

    def feed_for(ids: np.ndarray) -> dict:
        n = ids.shape[1]
        f = {"input_ids": ids, "attention_mask": np.ones_like(ids, dtype=np.int64)}
        if "position_ids" in inputs:
            f["position_ids"] = np.arange(n, dtype=np.int64).reshape(1, n)
        if with_past_inputs:
            for name, meta in inputs.items():
                if name.startswith("past_key_values."):
                    shp = [1 if isinstance(d, str) else d for d in meta.shape]
                    shp[2] = 0  # past_sequence_length = 0 for prefill
                    f[name] = np.zeros(shp, dtype=np.float32)
        return f

    def logits_for(ids: np.ndarray) -> np.ndarray:
        return sess.run(["logits"], feed_for(ids))[0].astype(np.float32)

    cases = []
    prompts = list(PROMPTS)
    n_rand = int(rng.integers(6, 20))
    rand_ids = rng.integers(3, tok.vocab_size, size=(1, n_rand)).astype(np.int64)
    for i, text in enumerate(prompts):
        ids = tok(text, return_tensors="np")["input_ids"].astype(np.int64)
        cases.append((f"text{i}", ids, f"prompt: {text!r}"))
    cases.append(("rand0", rand_ids, f"random tokens, seed={SEED}, n={n_rand}"))

    manifest = []
    for name, ids, note in cases:
        logits = logits_for(ids)
        cur = ids.copy()
        gen = []
        for _ in range(n_greedy):
            nxt = int(np.argmax(logits_for(cur)[0, -1]))
            gen.append(nxt)
            cur = np.concatenate([cur, np.array([[nxt]], dtype=np.int64)], axis=1)
        write(out_dir / f"{name}_ids.bin", ids)
        write(out_dir / f"{name}_logits.bin", logits)
        write(out_dir / f"{name}_greedy.bin", np.array(gen, dtype=np.int64))
        manifest.append({
            "name": name, "seq_len": int(ids.shape[1]), "vocab": int(logits.shape[2]),
            "n_greedy": n_greedy, "note": note, "greedy_tokens": gen,
            "greedy_text": tok.decode(gen),
        })
        print(f"{tag} {name}: N={ids.shape[1]} greedy={tok.decode(gen)!r}")
    (out_dir / "manifest.json").write_text(json.dumps({"seed": SEED, "cases": manifest}, indent=1))
    print(f"{tag}: {len(manifest)} cases -> {out_dir}")
    return 0


if __name__ == "__main__":
    rc = run_suite("llama15m", MODELS / "llama15M.onnx", "Xenova/llama2.c-stories15M", 12, False)
    if "--skip-tinyllama" not in sys.argv:
        rc = run_suite("tinyllama", MODELS / "tinyllama" / "onnx" / "model.onnx",
                       "Xenova/TinyLlama-1.1B-Chat-v1.0", 6, True) or rc
    sys.exit(rc)
