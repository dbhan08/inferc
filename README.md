# inferc

A C++17 ONNX inference optimizer and CPU runtime for Apple Silicon. Loads ONNX models, applies operator-fusion graph passes, and executes them through a custom CPU runtime backed by Apple Accelerate's AMX-backed sgemm.

## Bench

DistilBERT-SST2, n=30, M1, single-threaded ORT:

| backend          | mean(ms) | vs ORT |
|------------------|---------:|-------:|
| inferc-baseline  |  4935.11 | 39.5x slower |
| inferc-optimized |   875.20 | 7.0x slower |
| ort-cpu          |   124.82 | — |

**Fusion passes (MatMul+Add+GELU, LayerNorm, constant-folding) take inferc from 39.5x → 7.0x slower than ORT** — a 5.6x speedup over the unoptimized baseline. Output still matches ORT within the 1e-3 gate. On raw MatMul, **inferc beats ORT 10x** (23.4 vs 233.8 ms / iter) — Accelerate AMX wins decisively. The residual gap is unvectorized pointwise ops (vDSP is the next lever).

GPT-2-small autoregressive decode (M1, batch=1): **58.5 ms/token** with KV cache + constant-folded LM head + fused LayerNorm + shared-buffer weights (down from ~14.3s/token for the naive interpreter — a ~245x engineering win), vs ORT-CPU ~11 ms/token (5.2x). See [`CHALLENGES.md`](CHALLENGES.md) for the bugs found along the way.

## Run it

Requires macOS on Apple Silicon (M1/M2/M3).

```bash
brew install cmake protobuf poetry
poetry env use /opt/homebrew/bin/python3.13
poetry install --extras dev

cmake -B build && cmake --build build
cd build && ctest          # 70 tests, ~45 s
```

Reproduce the bench above (see [`DEMO.md`](DEMO.md) for full walkthrough):

```bash
poetry run python scripts/fetch_distilbert.py    # ~268 MB download
poetry run python scripts/make_inputs.py         # tokens + ORT golden logits
./build/inferc optimize models/distilbert.onnx --out models/distilbert.opt.onnx
./build/inferc bench --model models/distilbert.opt.onnx \
                     --ort-model models/distilbert.onnx \
                     -n 30 --warmup 5
```

## Commands

- `inferc inspect <model.onnx> [--ir]` — model summary, or IR dump with inferred shapes
- `inferc run <model> --input-ids <bin> --attention-mask <bin> --output <bin>` — execute end-to-end, write logits
- `inferc run ... --profile <out.json> -n <iters> --warmup <n>` — write a per-op JSON profile
- `inferc optimize <model> --out <plan>` — apply constant-folding (Transpose-of-constant) + LayerNorm fusion + RecognizeGELU + MatMul+Add+GELU fusion, write as ONNX
- `inferc compare <a.json> <b.json>` — side-by-side latency table (totals + top ops)
- `inferc bench [--model <plan>] [--ort-model <orig>]` — runs inferc + ORT on the canonical inputs and prints the table
- `inferc decode --model <gpt2.onnx> --past-model <gpt2_with_past.onnx> --prompt-ids <bin> --max-tokens N --output <bin> [--no-gemv] [--no-fold] [--profile <json>]` — autoregressive greedy decode with KV cache (GPT-2); reports per-token latency. Constant-folds the LM-head transpose + gated AMX-aware sgemv dispatch (98x faster per-token decode vs the naive interpreter; `--no-fold`/`--no-gemv` ablate)
- `inferc amx-probe [--out-csv <csv>] [--out-json <json>]` — sweep M/N/K through `cblas_sgemm`/`cblas_sgemv`, measure GFLOPs vs shape (AMX engagement curve). Plot with `poetry run python scripts/plot_amx.py` (needs `poetry install --extras plot`)

## How it works

- **Loader** parses ONNX protobuf, builds an internal IR (`Graph` of `Node`s + named `Tensor` table). Forward shape inference handles all 24 op types DistilBERT uses.
- **Passes** match patterns on the IR. `RecognizeGelu` folds the 7-node Erf-decomposed GELU from PyTorch ONNX export into one `Gelu` op. `FuseMatMulAddGelu` then collapses `MatMul → Add(bias) → Gelu` triplets into a `FusedMatMulAddGELU` op in a custom `inferc` domain. Single-use safety checks on every intermediate tensor.
- **Runtime** dispatches each IR node to a hand-written kernel. `MatMul`/`Gemm` route through `cblas_sgemm` (Apple Accelerate / AMX). The fused kernel does one sgemm into the output buffer, then a single per-element sweep adds bias and applies exact GELU (vs 5+ buffer passes pre-fusion).
- **Profiler** records per-op wall-clock + peak RSS + activation bytes; writes a JSON schema that the Python ORT bench script also produces, so `inferc compare` diffs them apples-to-apples.

## Stack

C++17 · CMake 3.20+ · Apple Accelerate (sgemm / AMX) · Protobuf 34.1 · GoogleTest 1.15 · nlohmann/json 3.11.3 · Python 3.13 (Poetry-managed) for ORT baseline + golden-output generation.

See [`PROJECT.md`](PROJECT.md) for the v1 spec, [`SESSIONS.md`](SESSIONS.md) for the session-by-session build log, and [`DEMO.md`](DEMO.md) for the reproducible demo.
