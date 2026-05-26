"""Per-token GPT-2 decode latency for ONNX Runtime — the inferc-vs-ORT baseline.

Mirrors `inferc decode`'s measurement: prefill over the prompt via gpt2.onnx,
then a KV-cached decode loop via gpt2_with_past.onnx, timing each single-token
step (the `sess.run` call only, excluding argmax/cache bookkeeping). Reports
mean / p50 / min ms per token over the decode steps.

ORT thread count matters a lot for this comparison: inferc's only parallelism
is inside Accelerate's BLAS calls, so we report single-threaded ORT by default
(matching the v1 DistilBERT bench methodology) and also print ORT's default
multi-threaded number for context.

Run via:
  poetry run python bench/bench_ort_gpt2_decode.py --max-tokens 32
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent.parent / "models"
PREFILL = MODELS / "gpt2.onnx"
WITH_PAST = MODELS / "gpt2_with_past.onnx"
IDS = MODELS / "gpt2_input_ids.bin"


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(s) - 1)
    return s[lo] + (rank - lo) * (s[hi] - s[lo])


def run(threads: int, max_tokens: int) -> dict:
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(PREFILL), so, providers=["CPUExecutionProvider"])
    sess_past = ort.InferenceSession(str(WITH_PAST), so, providers=["CPUExecutionProvider"])

    prompt = np.fromfile(IDS, dtype=np.int64).reshape(1, -1)
    n = prompt.shape[1]
    in_names = {i.name for i in sess.get_inputs()}
    feed = {"input_ids": prompt}
    if "attention_mask" in in_names:
        feed["attention_mask"] = np.ones((1, n), dtype=np.int64)
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, feed)

    logits = outs[0]
    cache = {
        name.replace("present.", "past_key_values."): arr
        for name, arr in zip(out_names, outs)
        if name.startswith("present.")
    }
    out_names_past = [o.name for o in sess_past.get_outputs()]
    next_token = int(np.argmax(logits[0, -1]))
    tokens = [next_token]
    cur = n
    step_ms: list[float] = []

    for _ in range(max_tokens - 1):
        cur += 1
        feed_past = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((1, cur), dtype=np.int64),
            **cache,
        }
        t0 = time.perf_counter()
        out_past = sess_past.run(None, feed_past)
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        next_token = int(np.argmax(out_past[0][0, 0]))
        tokens.append(next_token)
        cache = {
            name.replace("present.", "past_key_values."): arr
            for name, arr in zip(out_names_past, out_past)
            if name.startswith("present.")
        }

    return {
        "threads": threads,
        "steps": len(step_ms),
        "mean": statistics.mean(step_ms),
        "p50": percentile(step_ms, 50),
        "min": min(step_ms),
        "tokens": tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    print(f"ORT GPT-2 decode, prompt={np.fromfile(IDS, dtype=np.int64).size} tokens, "
          f"{args.max_tokens} generated\n")
    print(f"{'config':<22}{'steps':>7}{'mean(ms)':>11}{'p50(ms)':>11}{'min(ms)':>11}")
    print("-" * 62)
    for threads in (1, 0):  # single-threaded, then ORT default
        r = run(threads, args.max_tokens)
        label = "ort 1-thread" if threads == 1 else "ort default-threads"
        print(f"{label:<22}{r['steps']:>7}{r['mean']:>11.1f}{r['p50']:>11.1f}{r['min']:>11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
