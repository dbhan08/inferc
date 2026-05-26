"""Bench ONNX Runtime CPU EP on the same DistilBERT fixtures inferc uses.

Two-pass measurement:
  Pass 1 (no profiling) — measures total wall time per iter. These are the
                          numbers we report; ORT's own profiler adds overhead
                          so we don't trust its timings for headline latency.
  Pass 2 (profiling on, fewer iters) — extracts per-op-type breakdown from
                                       the chrome trace. Informative, not
                                       the source of total_ms stats.

Output JSON schema matches inferc::prof::Profiler::ToJson so `inferc compare`
can diff them directly:

  {
    "backend": "ort-cpu",   # or "ort-cpu-<N>t" / "ort-cpu-allt" for >1 thread
    "model": "...",
    "iterations": N,
    "total": {"mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"},
    "per_op_type": {OP: {"calls_per_iter", "total_ms": {...}}},
    "op_counts": {OP: count},
    "peak_rss_bytes": int,
    "activation_bytes_peak": int   # not measurable from ORT — left at 0
  }

Usage:
  poetry run python bench/bench_ort.py \
      --model models/distilbert.onnx \
      --input-ids models/input_ids.bin \
      --attention-mask models/attention_mask.bin \
      -n 100 --warmup 10 --out bench_out/baseline_ort.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import onnxruntime as ort

PROFILE_ITERS_DEFAULT = 5  # pass-2 iters with profiling on


def stats(values: list[float]) -> dict:
    if not values:
        return dict.fromkeys(("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"), 0.0)
    arr = np.array(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def make_session(model_path: str, profile: bool, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.enable_profiling = profile
    # threads=1: single-thread baseline (like-for-like vs inferc, which is
    # single-threaded). threads=0: ORT default = all cores (the full-machine
    # comparison — ORT scales across cores, inferc currently does not).
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def read_int64_bin(path: str, shape: tuple[int, ...]) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int64)
    return a.reshape(shape)


def aggregate_chrome_trace(trace_path: Path, n_profile_iters: int) -> tuple[dict, dict]:
    """Returns (per_op_type_stats, op_counts_per_iter).

    The chrome trace has events with cat='Node' and dur (microseconds). We
    aggregate ms summed by op_type per iter, then compute stats across iters.
    """
    with open(trace_path) as f:
        events = json.load(f)

    # Group sequential Node events into iterations. ORT emits them in run order
    # so we chunk by node count == ops per iter.
    node_events = [e for e in events if e.get("cat") == "Node" and "dur" in e]
    if not node_events:
        return {}, {}

    # Identify op_type per event. ORT records it under args.op_name.
    def op_of(e):
        return (e.get("args") or {}).get("op_name") or e.get("name", "Unknown")

    # Per-iter sums by op_type.
    ops_per_iter = len(node_events) // max(n_profile_iters, 1)
    per_iter_sums: list[dict[str, float]] = []
    counts: dict[str, int] = defaultdict(int)
    for i in range(n_profile_iters):
        chunk = node_events[i * ops_per_iter : (i + 1) * ops_per_iter]
        sums: dict[str, float] = defaultdict(float)
        for e in chunk:
            op = op_of(e)
            sums[op] += e["dur"] / 1000.0  # us -> ms
        per_iter_sums.append(dict(sums))

    # Use first iter for op_counts (graph is static across iters).
    for e in node_events[:ops_per_iter]:
        counts[op_of(e)] += 1

    # Stats per op_type.
    op_types = set()
    for s in per_iter_sums:
        op_types.update(s.keys())
    per_op = {}
    for op in op_types:
        vals = [s.get(op, 0.0) for s in per_iter_sums]
        per_op[op] = {
            "calls_per_iter": counts.get(op, 0),
            "total_ms": stats(vals),
        }
    return per_op, dict(counts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input-ids", required=True)
    p.add_argument("--attention-mask", required=True)
    p.add_argument("--shape", default="1,128", help="comma-separated B,S")
    p.add_argument("-n", "--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=1,
                   help="ORT intra-op threads; 1=single (like-for-like vs "
                        "single-threaded inferc), 0=all cores (full machine)")
    p.add_argument("--profile-iters", type=int, default=PROFILE_ITERS_DEFAULT,
                   help="iters for pass-2 per-op profile (default 5)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    input_ids = read_int64_bin(args.input_ids, shape)
    attention_mask = read_int64_bin(args.attention_mask, shape)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Pass 1: total times (no profiling).
    sess = make_session(args.model, profile=False, threads=args.threads)
    for _ in range(args.warmup):
        sess.run(None, feed)
    iter_ms: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        iter_ms.append((time.perf_counter() - t0) * 1000.0)
    total = stats(iter_ms)
    print(f"  pass-1 (totals): mean={total['mean_ms']:.2f}ms "
          f"p50={total['p50_ms']:.2f}ms p95={total['p95_ms']:.2f}ms "
          f"(n={args.iters})", flush=True)

    # Pass 2: per-op breakdown (profiling on, fewer iters).
    per_op: dict = {}
    op_counts: dict = {}
    if args.profile_iters > 0:
        sess_p = make_session(args.model, profile=True, threads=args.threads)
        for _ in range(args.warmup):
            sess_p.run(None, feed)
        for _ in range(args.profile_iters):
            sess_p.run(None, feed)
        prof_path = Path(sess_p.end_profiling())
        try:
            per_op, op_counts = aggregate_chrome_trace(prof_path, args.profile_iters)
        finally:
            prof_path.unlink(missing_ok=True)
        if per_op:
            top = sorted(per_op.items(),
                         key=lambda kv: -kv[1]["total_ms"]["mean_ms"])[:5]
            print("  pass-2 (top ops): "
                  + ", ".join(f"{op}={v['total_ms']['mean_ms']:.2f}ms"
                              for op, v in top), flush=True)

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes; on Linux it's in KB. Detect via platform.
    peak_rss_bytes = rusage.ru_maxrss if sys.platform == "darwin" \
        else rusage.ru_maxrss * 1024

    out = {
        "backend": "ort-cpu" if args.threads == 1 else f"ort-cpu-{args.threads or 'all'}t",
        "model": args.model,
        "iterations": args.iters,
        "total": total,
        "per_op_type": per_op,
        "op_counts": op_counts,
        "peak_rss_bytes": int(peak_rss_bytes),
        "activation_bytes_peak": 0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
