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

Under the hood, in `inferc::rt`:

- `Tensor` — shape + dtype + shared storage, contiguous + strided layouts
- `MatMul`, `Gemm` — Accelerate `cblas_sgemm` (AMX path on Apple Silicon)
- `Add`, `Sub`, `Mul`, `Div`, `Pow` — numpy-broadcasting binary ops
- `Sqrt`, `Erf`, `Relu`, `Tanh`, `Gelu`, `Softmax`, `LayerNorm`, `ReduceMean`
- `Gather` (embedding lookup), `Reshape`, `Transpose`, `Concat`, `Slice`, `Unsqueeze`, `Squeeze`, `Cast`

Planned:

- `inferc optimize <model.onnx> --out <plan>` — apply fusion + other passes (Session 7)
- `inferc run <model|plan> --input <bin>` — execute and produce outputs (Session 5)
- `inferc compare <a.json> <b.json>` — diff two profiles (Session 6)
- `inferc bench` — reproducible comparison vs ONNX Runtime CPU EP (Session 6)

## Try it

```bash
poetry run python scripts/fetch_distilbert.py    # downloads ~268 MB
poetry run python scripts/make_inputs.py         # tokens + ORT golden logits
poetry run python scripts/dump_ort_shapes.py     # ORT golden shapes for tests

cmake -B build && cmake --build build
./build/inferc inspect models/distilbert.onnx
./build/inferc inspect models/distilbert.onnx --ir | head -30
cd build && ctest
```

See [`PROJECT.md`](PROJECT.md) for the v1 spec and target resume bullet.

## Stack

C++17 · CMake 3.20+ · Apple Accelerate (sgemm/AMX) · Protobuf 34.1 · GoogleTest 1.15 · nlohmann/json 3.11.3 · Python 3.13 (Poetry-managed) for ORT baseline + golden-output generation.
