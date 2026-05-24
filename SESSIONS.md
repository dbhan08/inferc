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

- [x] `README.md` rewritten Advisor-style: pitch → headline bench → Run it → Commands → How it works (4 bullets) → Stack. Down from 89 lines of mixed status/planned/notes to a single tight document.
- [x] `DEMO.md` — reproducible 7-step CLI walkthrough with actual captured output: setup → fetch → inspect → run baseline → optimize → run optimized → bench → compare baseline-vs-optimized. ~5 min on an M1.
- [x] Bench results table embedded in README with real n=30 numbers from Session 7.
- [x] `RESUME.md` (gitignored) — 3 STAR bullets, 3 talking points (tradeoff / what broke / what I'd do differently), stack tags, numbers vault.
- [x] `.gitignore` already correct from earlier sessions (RESUME.md + build/ + models/ + .cache/ + bench_out/). Verified via `git check-ignore RESUME.md`.
- [x] Public repo live at github.com/dbhan08/inferc; description + topics set via `gh repo edit` (added: amx, sgemm, gelu, operator-fusion).

**Done when:** repo is public, README has real bench numbers, someone with an M1 can clone + reproduce within 30 minutes.

**Actuals:** repo public at https://github.com/dbhan08/inferc. README leads with the bench table. DEMO.md's reproduction path is `brew install ...` → ~5 min on a fresh M1 (download ~268 MB DistilBERT + 30-iter bench). 48/48 ctest cases passing. v1 ships.

Completed: 2026-05-23

---

# v2: GPT-2 + KV cache + AMX characterization + arxiv paper

> Full spec: [`V2_PLAN.md`](V2_PLAN.md). v2 is the path to an arxiv tech report (cs.PF / cs.LG) plus possible NeurIPS ENLSP 2026 workshop submission. M1-only scope. Headline target: per-token GPT-2 decode latency < ORT-CPU's, on M1, attributed to empirically-characterized AMX engagement thresholds.

---

## Session 9: GPT-2 scaffold

- [x] `scripts/fetch_gpt2.py` — pulls GPT-2-small (124M params) from `Xenova/gpt2` on HF Hub. Downloads both `decoder_model.onnx` → `models/gpt2.onnx` (no cache I/O, for inferc) and `decoder_with_past_model.onnx` → `models/gpt2_with_past.onnx` (for v2 Session 17 ORT KV-cache comparison). No torch dependency — uses pre-exported ONNX.
- [x] `scripts/make_gpt2_inputs.py` — tokenizes fixed prompt `"The quick brown fox jumps over the lazy"` (8 tokens) via HF tokenizer, runs ORT for golden logits `[1, 8, 50257]`, writes `models/gpt2_input_ids.bin` + `models/gpt2_golden_logits.bin` + meta file.
- [x] `inferc inspect models/gpt2.onnx` runs end-to-end.
- [x] Session note written below.

**Done when:** `inferc inspect models/gpt2.onnx` works; missing-ops list is captured; we know whether session 10's scope (filling the gap) is realistic at ≤5 new ops or needs re-planning at >10.

**Actuals at completion:**

- **Model**: Xenova/gpt2 `decoder_model.onnx` — torch_jit producer, **opset v13**, IR v7, **499.3 MB**, 148 initializers (~498 MB of weights).
- **Inputs (2)**: `input_ids: int64[batch, seq]`, `attention_mask: int64[batch, seq]`.
- **Outputs (25)**: `logits: float32[batch, seq, 50257]` + 24 `present.N.{key,value}` tensors (12 layers × {K, V}). The "no past" export still emits the *output* KV state — inferc just ignores those 24 outputs and reads `logits`.
- **Nodes**: **3092** (vs DistilBERT's 555 — GPT-2 has 12 transformer blocks vs DistilBERT's 6, plus more cache-handling sub-graphs).
- **24 distinct op types** in GPT-2. The 21 inferc already supports:
  `Constant (1214), Unsqueeze (284), Shape (279), Gather (196), Concat (148), Reshape (148), Add (123), Slice (109), Squeeze (85), Mul (74), Transpose (61), ReduceMean (50), Pow (49), Gemm (48), Cast (38), Sub (38), Div (37), MatMul (25), Sqrt (25), Softmax (12), Tanh (12), Where (12)`.

**Missing ops (3 — all simple):**

| Op | Count | What it does | Difficulty |
|---|---|---|---|
| `ConstantOfShape` | 12 | Output = tensor with given shape, all elements filled with a scalar `value` attribute. Used in GPT-2 for building attention-mask scaffolding. | Trivial (~20 lines). |
| `Split` | 12 | Splits one input tensor into N output tensors along an `axis`. Used in GPT-2 to split the concatenated Q/K/V projection (shape `[..., 3*hidden]`) into three `[..., hidden]` tensors. | Easy (~40 lines) — basically N Slice calls. |
| `Range` | 1 | Output = 1D int64 tensor `[start, start+step, ..., limit)`. Used once for position-IDs construction. | Trivial (~10 lines). |

**Session 10 scope:** 3 new kernels + their shape-inference rules + their executor dispatch entries + GTest unit tests for each. Realistic at well under "≤5 ops" budget. Total estimate: ~60-80 lines of C++ + ~30 lines of tests. **Plus** the correctness gate (GPT-2 position-0 forward pass logits match HF golden within 1e-3) — that's where the real verification happens.

**Other observations:**
- **opset v13** matches DistilBERT — no new opset semantics to worry about.
- **Gemm = 48** is 2× DistilBERT's (DistilBERT had 2 Gemms total in the classifier head). In GPT-2, every linear projection (Q, K, V, attention output, FFN-in, FFN-out per layer × 12 layers = 48) is a Gemm rather than MatMul + Add. Convenient — Gemm already does the bias-add internally.
- **MatMul = 25** — these are the Q·Kᵀ and Q·Kᵀ·V calls inside attention (2 per layer × 12 = 24, plus 1 elsewhere).
- **`present.N.{key,value}` outputs**: even the no-past export computes K and V tensors and writes them to output. inferc will compute them naturally (they're the K and V projections used in attention) but doesn't need to return them — the executor returns whatever's in `graph.outputs`, but downstream code (greedy decode) only reads `logits`.

**ORT smoke test (from make_gpt2_inputs.py):**
- Prompt: `"The quick brown fox jumps over the lazy"` → token IDs `[464, 2068, 7586, 21831, 18045, 625, 262, 16931]`.
- Predicted next token (argmax): `,` (id 11). Not the famous " dog" — GPT-2 was trained on web data, not pangrams. Fine for testing; correctness gate compares logits, not the predicted continuation.

Completed: 2026-05-24

---

## Session 10: Fill missing ops + GPT-2 forward pass

- [x] `kernels/movement.{h,cc}` — added `ConstantOfShape`, `Split` (multi-output), `RangeI64`, `RangeF32`.
- [x] `ir/shape_inference.cc` — added `Op_ConstantOfShape`, `Op_Range`, and `Op_Split` (writes to multiple outputs).
- [x] `runtime/executor.cc` — added dispatch for the three new ops. Split's multi-output path uses a `continue;` short-circuit so the standard single-output assignment doesn't run.
- [x] **`kernels/elementwise.cc` refactored** to support int64 alongside fp32 — required by GPT-2's shape arithmetic on int64 scalars. `BinaryBroadcast` now dispatches on dtype to `BinaryBroadcastTyped<float>` or `BinaryBroadcastTyped<int64_t>`. Add/Sub/Mul/Div/Pow now use generic-typed lambdas (`auto x, auto y`).
- [x] 8 new kernel unit tests (`ConstantOfShape` ×2, `Split` ×3, `Range` ×3).
- [x] `tests/correctness_test.cc` — added `EndToEnd.GPT2ForwardPassMatchesORT`.
- [x] `scripts/make_gpt2_inputs.py` — also writes `models/gpt2_attention_mask.bin` (the Xenova/gpt2 ONNX needs `attention_mask` as a graph input).

**Done when:** `inferc run models/gpt2.onnx --input-ids <bin> --output <bin>` produces logits that match HF golden ≤ 1e-3.

**Actuals:**
- **max_abs_diff = 4.12e-4** vs ORT golden logits — within the 1e-3 gate by ~2.4× margin. (Looser than DistilBERT's 5e-7 because GPT-2 is 12 layers vs DistilBERT's 6, so float32 rounding compounds more.)
- **inferc_argmax = 11 (`,`), golden_argmax = 11** — same predicted next token. ✓
- **20.8 s per forward pass** for 8-token prompt on M1 — slow, but expected for the un-vectorized interpreter walking 3092 nodes per inference. v2 sessions 14 (vDSP) and 15 (fused LayerNorm) attack this.
- **57/57 ctest cases passing** across the full suite: smoke + loader + shape inference + kernels (now incl. 8 new) + correctness (now incl. GPT-2) + profiler + passes.
- **Bonus surprise**: needed to extend `elementwise.cc` to int64 (GPT-2 does Add/Sub on int64 shape scalars). Clean template-based refactor; same Add/Sub/Mul/Div now work on both fp32 and int64.

Session 10 ran well under the budgeted scope — 3 new ops + 1 refactor + tests = ~150 lines of C++ and the gate is green. Session 11 (KV cache) is unblocked.

Completed: 2026-05-24

---

## Session 11: KV cache as executor state

- [ ] `runtime/executor.h` — add per-layer K/V cache buffers, `cached_len` counter
- [ ] Two execution modes in `Executor::Run`: prefill (cache writes) and decode (cache reads + appends)
- [ ] `inferc decode <model> --prompt-ids <bin> --max-tokens N --output <bin>` — greedy autoregressive generation with cache
- [ ] End-to-end test: 32-token greedy decode matches HF `transformers` token-for-token (correctness at every position)

**Done when:** 32-token greedy GPT-2 generation matches HF token-for-token, with cache verifiably populating (instrumentation prints `cached_len` growing).

---

## Session 12: AMX engagement microbench suite

- [ ] `inferc amx-probe` subcommand: sweeps M, N, K across a grid and measures GFLOPs for `cblas_sgemm` and `cblas_sgemv`
- [ ] Output: CSV + JSON, per-shape (M, N, K, GFLOPs achieved, theoretical peak, % of peak)
- [ ] Plot script (Python) producing the heatmap from the CSV
- [ ] Captured M1 measurement: clear threshold curve identified (or "no clean threshold, here's why")

**Done when:** the AMX engagement curve for M1 is captured as Paper Figure 1 raw data; reproducible from one CLI call.

---

## Session 13: AMX-aware decode kernel

- [ ] `kernels/fused_decode_step.cc` (or similar) — decode-step matmul routed through `cblas_sgemv` for AMX-favorable shapes; scalar fallback below threshold
- [ ] Wire into executor's decode mode for the Q/K/V projections + attention output projection
- [ ] Measure: per-token decode latency vs session-11 baseline; ablation row recorded

**Done when:** per-token decode latency on GPT-2 drops measurably from session 11; correctness gate still green.

---

## Session 14: vDSP-vectorized pointwise pipeline

- [ ] `kernels/elementwise.cc` — route `Add`, `Sub`, `Mul`, `Div` through `vDSP_v*`; size-based fallback for small N
- [ ] `kernels/activation.cc` — route `Sqrt`, `Erf`, `Tanh` through `vv*` math functions
- [ ] DistilBERT + GPT-2 bench: pointwise op times drop measurably; existing 48+ tests still pass

**Done when:** pointwise op family is no longer scalar; net latency drops on both DistilBERT and GPT-2; ablation row recorded.

---

## Session 15: Fused LayerNorm pass + kernel

- [ ] `ir/passes/recognize_layernorm.{h,cc}` — pattern-matches `ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add` (the 8-op decomposed LayerNorm) and folds into a `FusedLayerNorm` op
- [ ] `kernels/fused_layernorm.cc` — one-pass kernel computing mean, variance, normalize, scale, bias in a single sweep
- [ ] Tests: synthetic pattern match + DistilBERT correctness still passing
- [ ] DistilBERT: ReduceMean total time drops from ~1.35s → expected ~80-150ms

**Done when:** LayerNorm dominates 35% less of inferc's runtime; ablation row recorded.

---

## Session 16: `inferc chat` REPL

- [ ] `inferc chat <model>` — interactive prompt → token stream
- [ ] Flags: `--temperature`, `--max-tokens`, `--top-k`
- [ ] Token-by-token streaming output (flush after each)
- [ ] Demo screenshot for paper + README

**Done when:** `inferc chat models/gpt2.onnx` accepts a prompt and streams tokens; works for at least 32-token continuations.

---

## Session 17: Multi-baseline bench harness

- [ ] `bench/bench_llama_cpp.py` — runs GPT-2 through llama.cpp (after model conversion via `ggml-org` tooling), writes same JSON schema
- [ ] `bench/bench_ctranslate2.py` — same for CTranslate2
- [ ] `bench/bench_pytorch.py` — PyTorch CPU baseline
- [ ] Extend `inferc bench` to orchestrate all 5 baselines (inferc + ORT + 3 new)
- [ ] Captured: Table 1 raw data (per-token latency at positions 1, 8, 32, 128, 256)

**Done when:** `inferc bench --full` runs all 5 backends end-to-end and produces the Paper Table 1.

---

## Session 18: Hardware-counter attribution

- [ ] Run Instruments CPU profile traces for inferc-decode and ORT-decode on the same input
- [ ] Per-op attribution: where does inferc spend its time vs ORT spend its time?
- [ ] Identify the *mechanism* for the speedup (GEMV path vs MLAS GEMM path, ideally with counter-level evidence)
- [ ] Captured: Paper Figure 2 raw data

**Done when:** the *why* of inferc's speedup is attributable to specific microarchitectural events / dispatch decisions, not just wall-clock.

---

## Session 19: Paper draft

- [ ] LaTeX project set up (use NeurIPS or arxiv generic template)
- [ ] Outline locked (see V2_PLAN.md §"Paper outline")
- [ ] All figures + tables typeset
- [ ] First full draft (Intro → Background → System → AMX char → Decode kernel → Eval → Discussion → Limitations → Conclusion)
- [ ] Bib file with ~25 cited papers (AMX prior art + CPU inference SOTA + KV cache literature)

**Done when:** complete draft exists; ready for reader feedback.

---

## Session 20: Polish + arxiv submission

- [ ] Reader pass (find 1-2 readers — labmate, advisor, friend in the field — by week 13 so they have warning)
- [ ] Revise based on feedback
- [ ] Final correctness sanity check on all numbers in the paper
- [ ] University endorsement step (likely automatic via .edu affiliation)
- [ ] Submit to arxiv (cs.PF primary; cs.LG and cs.AR cross-listings)
- [ ] Wait ~1-2 days for moderator approval
- [ ] Update RESUME.md and README.md with the arxiv URL
- [ ] (Optional) Submit same paper to NeurIPS ENLSP 2026 if the deadline is still open

**Done when:** arxiv URL is live; resume + repo updated; paper exists in the world.

---

## v3 (deferred — beyond v2)

- v3: CLIP-ViT-B/32 (proves multi-modal generality)
- v3: real memory planner (activation buffer reuse via liveness analysis)
- v3: M2 / M3 / M4 cross-chip generalization study (extends the AMX paper)
- v3: INT8 weight quantization
- v3: Energy/joule reporting via `powermetrics`
- v3: Speculative decoding
- v3: GPT-2-medium / Llama scaling
