# inferc

A C++ ONNX inference optimizer and CPU runtime for Apple Silicon. Loads ONNX models, applies compiler-style graph optimizations (operator fusion, constant folding, shape inference), and executes them through a custom CPU runtime backed by Apple's Accelerate framework.

> **Status:** Work in progress. Built incrementally, session by session — see [`SESSIONS.md`](SESSIONS.md) for what's done and what's next. v1 ships a fused-attention DistilBERT inference path benchmarked against ONNX Runtime CPU EP, with numerical correctness validated within 1e-3.

## Run it

Requires macOS on Apple Silicon (M1/M2/M3).

```bash
brew install cmake protobuf poetry
cd inferc
poetry env use /opt/homebrew/bin/python3.13
poetry install --extras dev

cmake -B build && cmake --build build
./build/inferc --version
cd build && ctest
```

## Commands

Working today:

- `inferc inspect <model.onnx>` — model summary (op counts, IO shapes, opset, weight bytes)
- `inferc inspect <model.onnx> --ir` — internal IR with per-node inferred shapes
- `inferc run <model.onnx> --input-ids <bin> --attention-mask <bin> --output <bin>` — execute the model end-to-end and write logits to disk
- `inferc run ... --profile <out.json> -n <iters> --warmup <n>` — write a per-op JSON profile (mean/p50/p95 across iters, peak RSS, activation bytes)
- `inferc compare <a.json> <b.json>` — side-by-side comparison table (total latency + top ops)
- `inferc bench` — runs inferc + ORT-CPU on the canonical DistilBERT-SST2 inputs and prints the comparison table

End-to-end correctness gate (Session 5): inferc's logits on DistilBERT-SST2 match ONNX Runtime's CPU EP **within 4.76e-07 (max-abs-diff)** — 4 orders of magnitude tighter than the 1e-3 v1 gate.

Under the hood, in `inferc::rt`:

- `Tensor` — shape + dtype + shared storage, contiguous + strided layouts
- `MatMul`, `Gemm` — Accelerate `cblas_sgemm` (AMX path on Apple Silicon)
- `Add`, `Sub`, `Mul`, `Div`, `Pow` — numpy-broadcasting binary ops
- `Sqrt`, `Erf`, `Relu`, `Tanh`, `Gelu`, `Softmax`, `LayerNorm`, `ReduceMean`
- `Gather` (embedding lookup), `Reshape`, `Transpose`, `Concat`, `Slice`, `Unsqueeze`, `Squeeze`, `Cast`

Planned:

- `inferc optimize <model.onnx> --out <plan>` — apply MatMul+Add+GELU fusion and other passes (Session 7)

## Try it

```bash
poetry run python scripts/fetch_distilbert.py    # downloads ~268 MB
poetry run python scripts/make_inputs.py         # tokens + ORT golden logits
poetry run python scripts/dump_ort_shapes.py     # ORT golden shapes for tests

cmake -B build && cmake --build build
./build/inferc inspect models/distilbert.onnx
./build/inferc inspect models/distilbert.onnx --ir | head -30
./build/inferc bench -n 30 --warmup 5            # vs ORT CPU EP, ~2 min
cd build && ctest
```

## Bench numbers (Session 6 baseline, pre-fusion)

n=30, single-threaded ORT, Apple M1:

| backend          | mean(ms) | p50(ms) | p95(ms) | RSS(MB) |
|------------------|---------:|--------:|--------:|--------:|
| inferc-baseline  |  4935.11 | 4932.73 | 4964.62 |  1091.4 |
| ort-cpu          |   122.60 |  122.59 |  122.88 |   792.3 |

inferc is 40x slower on total latency pre-fusion, **but 4.6x faster on raw MatMul** (49.9ms inferc vs 231.3ms ORT per iter) — the AMX-backed sgemm path is doing its job. The gap is in everything-around-the-matmul: ReduceMean/LayerNorm (1.35s/iter) and other pointwise ops that ORT either has fused or vectorized.

Session 7 closes the gap by fusing MatMul + Add + GELU into a single kernel pass.

See [`PROJECT.md`](PROJECT.md) for the v1 spec and target resume bullet.

## Stack

C++17 · CMake 3.20+ · Apple Accelerate (sgemm/AMX) · Protobuf 34.1 · GoogleTest 1.15 · nlohmann/json 3.11.3 · Python 3.13 (Poetry-managed) for ORT baseline + golden-output generation.
