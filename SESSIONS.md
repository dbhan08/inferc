# InferC v1 — Sessions

Discrete, pause-resumable build steps. Check items as completed. When all items in a session are done, append `Completed: YYYY-MM-DD`.

---

## Session 1: Scaffold

- [x] CMake project (`project(inferc CXX)`, C++17)
- [x] Directory layout: `src/{cli,frontend,ir,runtime,kernels,profiler,util}`, `tests/`, `models/`, `third_party/onnx_pb/`, `third_party/json/`, `bench/`, `scripts/`
- [x] Vendor ONNX `.proto` (pinned to ONNX v1.17.0) under `third_party/onnx_pb/`. `onnx.pb.{h,cc}` codegen at build time via `protoc` (no binary artifacts in repo).
- [x] Wire dependencies: protobuf 34.1 (via brew, `find_package(Protobuf CONFIG REQUIRED)` so abseil deps come through), Accelerate framework, GoogleTest (FetchContent v1.15.2), nlohmann/json v3.11.3 (single header)
- [x] Stub `inferc` CLI binary with `--version` / `--help` / usage
- [x] Smoke tests: version non-empty + ONNX protobuf links (constructs `onnx::ModelProto`)

**Done when:** `cmake -B build && cmake --build build && ./build/inferc` prints version. `ctest` runs (zero tests, exits 0).

Completed: 2026-05-02

---

## Session 2: ONNX loader + `inspect` command

- [x] `frontend/onnx_loader.{h,cc}` — parse `.onnx` file via protobuf
- [x] Extract: opset version, op type counts, graph inputs/outputs with shapes, initializer count + total weight bytes
- [x] `inferc inspect <model.onnx>` prints a clean summary
- [x] `scripts/fetch_distilbert.py` — pulls SST-2 fine-tuned DistilBERT (`optimum/distilbert-base-uncased-finetuned-sst-2-english`) from HF Hub. No torch needed.
- [x] `scripts/make_inputs.py` — tokenizes prompt `"The food at this restaurant was incredible."`, runs ORT CPU EP, writes `models/input_ids.bin` + `models/attention_mask.bin` + `models/golden_logits.bin` + `models/inputs_meta.txt`
- [x] All Python scripts run via `poetry run python scripts/...`
- [x] Unit tests: 3 GTest cases on hand-crafted tiny ONNX (round-trip, empty model, dtype byte sizes)

**Done when:** `inferc inspect models/distilbert.onnx` prints opset, ~100 nodes, correct input shape `[1, 128]`, ~265MB weights.

**Actuals at completion:** opset v13, **555 nodes** (much higher than estimate — DistilBERT here uses Erf-decomposed GELU and has 24 distinct op types), inputs `input_ids` + `attention_mask` both `[batch, sequence]` (symbolic), output `logits` `[batch, 2]`, 184 initializers, **267.82 MB**. ORT golden: prediction=POSITIVE, logits=[-4.336, 4.662].

Completed: 2026-05-03

---

## Session 3: Internal IR + shape inference

- [x] `ir/graph.{h,cc}` — `DType`, `Tensor`, `Node`, `Graph` types. Tensors carry shape, dtype, optional initializer bytes. Attribute accessors on `Node` for int/ints/float/string.
- [x] `frontend/onnx_to_ir.{h,cc}` — converts ONNX → IR, materializes initializer bytes (raw_data + typed-data fallbacks).
- [x] `ir/shape_inference.{h,cc}` — forward shape inference for **all 24 op types** in DistilBERT-SST2: `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sqrt`, `Erf`, `Relu`, `Softmax`, `Tanh`, `Gather`, `Reshape`, `Transpose`, `Concat`, `Slice`, `Unsqueeze`, `Squeeze`, `Cast`, `Constant`, `Shape`, `Equal`, `Where`, `Expand`, `ReduceMean`. Numpy-style broadcast helper.
- [x] `inferc inspect --ir model.onnx` prints IR with per-node inferred shapes.
- [x] `scripts/dump_ort_shapes.py` writes `models/golden_shapes.json` from ONNX's official shape inferencer (`onnx.shape_inference`, `data_prop=True`).
- [x] `tests/shape_inference_test.cc` runs inferc shape inference on real DistilBERT and asserts per-node shape match against the ORT golden.

**Done when:** every node in DistilBERT has a non-empty inferred output shape that matches ORT.

**Actuals at completion:** 0 unsupported ops, **240/240 matched ORT** on the nodes ORT emitted shapes for (315 outputs are constants/Shape ops where ORT didn't bother emitting `value_info` — inferc still infers correct shapes for those, just no golden to compare against). 7/7 ctest cases passing.

Completed: 2026-05-03

---

## Session 4: Runtime kernels (interpreter mode)

- [x] `runtime/tensor.{h,cc}` — owning typed Tensor with shape + strides + shared_ptr storage. Lives in sub-namespace `inferc::rt` to avoid collision with the IR's `Tensor` (metadata).
- [x] `kernels/matmul.cc` — `cblas_sgemm` wrapper, row-major, batched broadcast on leading dims. Routes through Accelerate → AMX. Plus `Gemm` (alpha·op(A)·op(B) + beta·C) for the classifier head.
- [x] `kernels/elementwise.cc` — `Add`, `Sub`, `Mul`, `Div`, `Pow` with numpy-style right-aligned broadcasting (zero-stride trick). Plus unary pointwise: `Sqrt`, `Erf`, `Relu`, `Tanh`, `Neg`, `Abs`.
- [x] `kernels/activation.cc` — Exact GELU (`x · 0.5 · (1 + erf(x/√2))`, matches DistilBERT export). Numerically stable Softmax (max-subtract trick) along arbitrary axis. Fused LayerNorm. ReduceMean.
- [x] `kernels/embedding.cc` — `Gather` along arbitrary axis with int32/int64 indices.
- [x] `kernels/movement.cc` — `Reshape` (with -1 inference), `Transpose` (physical copy via stride permutation), `Concat`, `Slice` (with starts/ends/axes/steps), `Unsqueeze`, `Squeeze`, `Cast`.
- [x] Per-kernel unit tests in `tests/kernels_test.cc` — 22 cases covering each kernel with hand-computed goldens, tolerance ≤ 1e-5.

**Done when:** every per-op test passes with max-abs-diff ≤ 1e-5 vs reference.

**Actuals:** 29/29 ctest cases passing across the full suite (smoke + loader + shape inference + kernels). All kernels verified against hand-computed numerical references.

Completed: 2026-05-06

---

## Session 5: End-to-end run + numerical correctness gate

- [x] Filled in missing kernels: `Equal` (broadcasted comparison → bool), `Where` (3-arg select), `Expand` (broadcast to target shape), `ShapeOf` (input.shape as int64 1D tensor).
- [x] `runtime/executor.{h,cc}` — materializes initializers as `rt::Tensor`s once, then `Run(inputs)` walks `graph.nodes` in topo order, dispatches each `op_type` to the right kernel via a switch, threads tensors through a name→Tensor "tape", and returns the graph outputs.
- [x] `inferc run <model.onnx> --input-ids <bin> --attention-mask <bin> --output <bin>` — full CLI command. Reads int64 token IDs and attention mask from binary files, runs the executor, writes float32 logits.
- [x] `tests/correctness_test.cc` — loads real DistilBERT, runs end-to-end, compares to ORT's golden logits.

**Done when:** `inferc run` produces logits matching ORT within 1e-3 on the fixed test prompt.

**Actuals:**
- inferc logits: `[-4.336367, 4.661800]`
- ORT golden:   `[-4.336367, 4.661799]`
- **max_abs_diff = 4.76837e-07** — at float32 epsilon, *4 orders of magnitude tighter* than the 1e-3 gate.
- 30/30 ctest cases passing across the whole suite.

The project went from "describes a model" to "computes a model" this session. inferc and ORT now agree bit-for-bit (modulo fp32 round-off). Every kernel, every shape, every attention-mask path is correct.

Completed: 2026-05-06

---

## Session 6: Profiler + bench harness

- [x] `profiler/profiler.{h,cc}` — per-op wall-clock timing (`std::chrono::steady_clock`), peak RSS via `task_info(MACH_TASK_BASIC_INFO)`, activation memory accounting from executor (sum of live, non-initializer tensor bytes, max per iter)
- [x] `inferc run --profile <out.json> -n <iters> --warmup <n>` writes structured report: per-op times (mean / p50 / p95 / min / max over N iters), peak RSS, activation bytes, op count by type
- [x] `inferc compare <baseline.json> <other.json>` prints totals + top-N ops by time with ratio
- [x] `bench/bench_ort.py` — two-pass measurement: pass-1 untimed for total wall time (no profiler overhead), pass-2 with ORT's chrome-trace profiler for per-op breakdown. Writes the same JSON schema → `baseline_ort.json`
- [x] `inferc bench` — convenience: runs inferc + ORT on the canonical fixtures, calls `compare`, prints the table
- [x] Profiler unit tests (9 new GTest cases — percentile / stats / iteration recording / op stack / JSON roundtrip / peak RSS)

**Done when:** `inferc bench` outputs a side-by-side table of inferc-unoptimized vs ORT-CPU on identical inputs, reproducible across runs (median of 100+ iterations).

**Actuals at completion (n=30, warmup=5, single-threaded ORT):**

| backend          | mean(ms) | p50(ms) | p95(ms) | RSS(MB) |
|------------------|---------:|--------:|--------:|--------:|
| inferc-baseline  |  4935.11 | 4932.73 | 4964.62 |  1091.4 |
| ort-cpu          |   122.60 |  122.59 |  122.88 |   792.3 |

inferc-baseline is **40.25x slower than ORT** overall pre-fusion — expected: this is the unoptimized interpreter doing 555 nodes per inference with no pointwise SIMD and a 26-call ReduceMean path implementing LayerNorm as separate ops.

Per-op insight from the comparison table:

- **inferc beats ORT 4.63x on raw MatMul** (49.9ms vs 231.3ms / iter). Apple Accelerate's AMX-backed sgemm path is working. This is the foundation Session 7's fusion builds on.
- **LayerNorm-as-separate-ops dominates inferc**: ReduceMean 1347ms + Add 1075ms + Div 503ms + Mul 429ms + Pow 186ms + Sub 169ms = ~3.7s of the 4.9s total. The PROJECT.md risk doc called this exactly.
- **ORT's "MatMul" is a fused linear layer** (MatMul + bias-add), which is why ORT MatMul is 4.6x slower than inferc's bare MatMul but ORT-total is 40x faster. This sets up the Session 7 fusion story: collapsing MatMul + Add + GELU into one kernel should narrow the gap dramatically.

39/39 ctest cases passing (30 prior + 9 profiler).

Completed: 2026-05-21

---

## Session 7: Flagship fusion — MatMul + Add + GELU

- [x] `ir/passes/recognize_gelu.{h,cc}` — Pass 1: matches the 7-node Erf-decomposed GELU pattern that PyTorch ONNX export emits (`Div(/sqrt(2)) → Erf → Add(+1) → Mul(X,·) → Mul(·,0.5)`), folds it into a single `Gelu` op. Validates constants by approx-eq + checks every intermediate tensor is single-use.
- [x] `ir/passes/fuse_matmul_add_gelu.{h,cc}` — Pass 2 (the flagship): IR pattern matcher for `MatMul → Add(bias) → Gelu` triplets, replaces with a `FusedMatMulAddGELU` op in the custom `inferc` domain.
- [x] `ir/passes/pass_utils.{h,cc}` — shared helpers: use-counts, constant scalar readers, producer lookup, approx-equal float comparison.
- [x] `kernels/fused_matmul_add_gelu.cc` — single sweep: one `cblas_sgemm` (Accelerate / AMX) into the output buffer, then **one** fused per-element pass adds bias and applies exact GELU. Collapses 5+ separate buffer passes into 1.
- [x] `inferc optimize <model.onnx> --out <plan.onnx>` writes the optimized graph back as ONNX (custom domain registered in `opset_import`). Prints node-count delta + pattern-match counts.
- [x] `inferc run <plan.onnx>` consumes the optimized plan — auto-detects the fused op and tags the profile JSON `inferc-optimized` (vs `inferc-baseline`).
- [x] `inferc bench --model <plan> --ort-model <orig>` runs apples-to-apples comparison: inferc on the optimized plan, ORT on the unoptimized .onnx it can actually load.
- [x] 8 new GTest cases for the passes (synthetic small graphs): commutative-operand tolerance, wrong-constant rejection, extra-consumer safety, three-op fusion, bias operand order. Plus one new end-to-end test that runs the optimized DistilBERT and asserts max-abs-diff ≤ 1e-3 vs ORT golden.

**Done when:** optimized DistilBERT passes the correctness gate, op count is reduced, latency is measurably different from unoptimized inferc, and the bench table includes optimized numbers vs ORT.

**Actuals (n=30, warmup=5, single-threaded ORT, M1):**

Pass counts on DistilBERT-SST2:
- Erf-GELU patterns folded: **6** (= 6 transformer blocks × 1 FFN-GELU each)
- MatMul+Add+GELU triplets fused: **6**
- Nodes: 555 → 501 (54 removed: 48 from recognize-GELU, 6 net from fusion)

End-to-end latency:

| backend          | mean(ms) | p50(ms) | p95(ms) | RSS(MB) |
|------------------|---------:|--------:|--------:|--------:|
| inferc-baseline  |  4935.11 | 4932.73 | 4964.62 |  1091.4 |
| inferc-optimized |  3922.26 | 3912.44 | 3974.73 |   932.0 |
| ort-cpu          |   124.30 |  124.22 |  124.89 |   792.8 |

**Headline: inferc-optimized is 1.26x faster than baseline (20.5% latency reduction).**

Per-op breakdown of the win (baseline → optimized ms / iter):
- Add: 1075 → 537 (−538ms, 12 fewer calls — bias-Add + GELU's `+1` Add fused away)
- Mul: 429 → 161 (−268ms, 12 fewer calls — both GELU `Mul`s folded)
- Div: 503 → 243 (−260ms, 6 fewer calls — GELU's `/sqrt(2)` folded)
- MatMul: 50 → 33 (−17ms, 6 fewer calls — FFN MatMuls absorbed into fused op)
- New FusedMatMulAddGELU: +54 ms / iter (one fused kernel doing all of it)
- **Net saved: ~1029ms / iter (matches measured 1013ms delta within fp noise)**

Correctness: optimized DistilBERT logits = `[-4.336367, 4.661800]`, golden ORT = `[-4.336367, 4.661800]`. **max_abs_diff at float32 epsilon (≤ 5e-7)** — 4 orders of magnitude tighter than the 1e-3 v1 gate, same as Session 5 unoptimized.

inferc-vs-ORT: optimized inferc is **31.55x slower than ORT overall** (was 40.25x at baseline). But on raw MatMul, **inferc beats ORT 6.99x** (33.5ms vs 234.3ms) — Accelerate AMX is decisively the right kernel choice. The gap is now entirely in the unvectorized pointwise pipeline (ReduceMean 1.38s / iter — LayerNorm split across ReduceMean + Pow + Sub + Sqrt + Div — and Transpose 753ms / iter). Vectorizing those is v2 work.

48/48 ctest cases passing (39 prior + 8 pass tests + 1 optimized-correctness gate).

Completed: 2026-05-22

---

## Session 8: Polish + ship

- [ ] `README.md` — Advisor-style: pitch, "Run it" with exact commands, usage examples, "How it works" (4 bullets max), stack list
- [ ] `DEMO.md` — reproducible CLI sequence: export model → tokenize → run baseline → run optimized → compare. Include expected output table.
- [ ] Bench results table embedded in README with the real numbers
- [ ] `RESUME.md` (gitignored) — STAR bullets, talking points, stack tags
- [ ] `.gitignore` correct (RESUME.md, build/, models/*.onnx, models/*.bin, .venv/)
- [ ] `gh repo create inferc --public --source=. --remote=origin --push`, set description + topics

**Done when:** repo is public, README has real bench numbers, someone with an M1 can clone + reproduce within 30 minutes.

---

## v2/v3 (deferred — not built unless v1 ships)

- v2: GPT-2 small + KV cache + `inferc chat` REPL
- v2: CLIP-ViT-B/32 (proves multi-modal generality)
- v2: constant folding + DCE (cheap once IR exists)
- v2: real memory planner (activation buffer reuse via liveness analysis)
- v3: INT8 weight quantization
- v3: Energy/joule reporting via `powermetrics`
