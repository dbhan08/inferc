# InferC — v1 Spec

## Pitch

A C++ ONNX inference optimizer and CPU runtime for Apple Silicon. Loads ONNX models, applies compiler-style graph optimizations (operator fusion, constant folding, shape inference), and executes them through a custom CPU runtime backed by Apple's Accelerate framework. v1 ships a fused-attention DistilBERT inference path benchmarked against ONNX Runtime CPU EP, with numerical correctness validated.

## Goals (v1)

- Load DistilBERT-base from ONNX (opset ~14)
- Convert ONNX → internal IR with forward shape inference
- One flagship optimization: **MatMul + Add + GELU fusion**
- CPU runtime executing the ops DistilBERT uses (no others)
- MatMul routed through Apple Accelerate `cblas_sgemm` (AMX-backed)
- Profiler producing per-op timing, peak RSS, activation memory
- Numerical correctness gate: max-abs-diff vs ONNX Runtime ≤ 1e-3
- CLI: `inferc inspect | optimize | run | compare | bench`
- Reproducible bench against ORT-CPU on the same M1 — anyone can `inferc bench` and get the same shape of result

## Non-goals (explicit, gated for later)

- v2: GPT-2 small + KV cache + `inferc chat` REPL
- v2: CLIP-ViT-B/32
- v3: INT8 weight quantization
- v3: Energy reporting via `powermetrics`
- Out of scope entirely: Conv kernels, BatchNorm folding, im2col packing, depthwise conv (CNN territory — different project)
- Out of scope entirely: GPU / Metal, training, fp16, dynamic batch
- Out of scope for v1: dead code elimination, constant folding, full memory planner (these are easy wins to add in v2; v1 prioritizes the fusion demo)

## System dependencies

Installed on the dev machine (Apple Silicon M1, macOS):

| Tool | Version | Source | Purpose |
|---|---|---|---|
| Apple Accelerate | system | macOS framework | `cblas_sgemm`, `vDSP` for kernels |
| `cmake` | 4.3.2 | `brew install cmake` | Build system |
| `protobuf` | 34.1 (libprotoc 34.1) | `brew install protobuf` | Parse ONNX models, codegen `onnx.pb.{h,cc}` at build time |
| `poetry` | 2.3.4 | `brew install poetry` | Python dep mgmt for baseline + golden generation |
| Python | 3.13 | `brew` (`python@3.13`) | Poetry-managed venv runs `onnxruntime`, `transformers`, `numpy`, `huggingface-hub` |

**PATH note:** the user's shell has anaconda's `protoc` 3.19.1 ahead of brew's 34.1. CMake explicitly sets `CMAKE_PREFIX_PATH=/opt/homebrew` so the brew protobuf is always found first; no shell-config changes required to build this project.

To reproduce on a fresh M1 Mac:
```bash
brew install cmake protobuf poetry
cd inferc
poetry env use /opt/homebrew/bin/python3.13
poetry install --extras dev
```

## Stack

- C++17, CMake 3.20+
- Apple Accelerate (sgemm, vDSP)
- ONNX protobuf (vendored under `third_party/onnx_pb/`)
- GoogleTest for unit tests
- nlohmann/json single-header for profile JSON
- **Poetry-managed Python 3.13** (`pyproject.toml` + `poetry.lock`) for baseline + golden-output generation: `onnxruntime`, `transformers` (tokenizer-only — no torch), `numpy`, `huggingface-hub`. Dev extras: `pytest`, `ruff`. Run via `poetry run python …`.

## Scope

Multi-week. ~7-8 focused sessions to v1 ship. Each session 2-3h.

## Risks / unknowns

- **ORT-CPU is heavily tuned.** Realistic v1 result: 0.7-1.0× ORT (parity to 30% slower) is respectable; beating ORT is stretch. Frame the resume bullet around the actual measured number, not a speculative one.
- **LayerNorm without a tuned kernel** could become an unexpected hot spot — DistilBERT has 2 LNs per block × 6 blocks = 12 LN calls per inference. Watch the profile.
- **ONNX opset variants.** DistilBERT exports vary by `transformers` version. Pin the export script in the repo.
- **Numerical drift** in fused kernels — fp32 accumulator order matters when reductions are reordered. Tolerance gate must be explicit and test-enforced.
- **Tokenization is upstream.** Don't reinvent it; use HuggingFace tokenizer in Python to produce `input.bin`, runtime just consumes int64 token IDs.

## Target resume bullet

> Reduced DistilBERT inference latency by **X%** (median over 1000 runs) vs ONNX Runtime CPU EP on Apple M1 by fusing MatMul+Add+GELU through Accelerate's AMX-backed sgemm path; validated numerical equivalence to ORT within 1e-3. Built loader, IR with shape inference, fusion pass, and CPU runtime in C++17.

X gets filled in by the bench harness, not made up. Range to defend in interviews: parity (1.0×) is a respectable result given ORT's tuning; anything beating ORT on the fused-op hot path is flagship.
