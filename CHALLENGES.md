# Challenge log

A running record of the non-obvious bugs, performance pathologies, and
false-starts hit while building inferc — what broke, why, the fix, and the
lesson. Kept because the *why-it-broke* is the interesting part (and the best
interview / paper material). Newest first.

---

## C9 — fp16 compute would make GPT-2 *slower* on M1 (Session 16)

- **Plan:** implement full fp16 inference via Apple's BNNS `BNNSMatMul`, expecting
  the usual "half precision = ~2× throughput" win.
- **Premise test (before integrating):** extended `inferc amx-probe` to benchmark
  BNNS fp16 GEMM vs the fp32 `cblas_sgemm` AMX path across the shape sweep.
- **Finding:** fp16 does **not** win for GPT-2-small's shapes. At the hidden dim
  (N=K=768) fp16 peak is 0.90× fp32; fp16 only beats fp32 at N=K≥1536. Worse, at
  the **decode shape (M=1, NK=768)** fp16 is **9 GFLOPs vs fp32's 81 — ~9× slower**
  (BNNS has high per-call overhead and doesn't engage AMX for a single row). The
  fp32 AMX `sgemm` path is already better for everything GPT-2-small does.
- **Decision:** do **not** build full fp16 compute — it would regress latency.
  fp16 *storage* (memory halving) remains an option, but compute stays fp32.
- **Lesson:** the same as C7 — test the premise cheaply first. A ~5-minute probe
  extension replaced a multi-session integration that would have made things
  worse. "Lower precision is faster" is not a law; it depends on whether the
  vendor's fp16 kernel engages the matrix unit for your shapes.

## C8 — `Contiguous()` deep-copied already-contiguous tensors (Session 15)

- **Symptom:** GPT-2 per-token decode spent ~20 ms in `Gather` and ~20 ms in
  `MatMul` — absurd, since at decode the embedding gather copies *one* 768-float
  row and the attention matmuls are tiny.
- **Root cause:** `Tensor::Contiguous()` always made a full copy, even when the
  tensor was already contiguous ("caller asked for one"). Every `Gather` of the
  `[50257, 768]` embedding (154 MB) and every `MatMul`/`Gemm` on the tied
  LM-head weight (154 MB) deep-copied the whole weight *per call*.
- **Fix:** when already contiguous, `Contiguous()` returns a shared-storage view
  (`return *this`) instead of copying. Every kernel treats the result as
  read-only input and writes to a separate output buffer, so sharing is safe.
- **Impact:** GPT-2 decode 106 → 58.5 ms/token (1.8×). 5.2× off ORT (was 13×).
- **Lesson:** a "defensive copy" in a hot path is a silent perf bug. Share
  read-only buffers; copy only when a kernel actually mutates in place (none do).

## C7 — The flat-execution-plan hypothesis was wrong (Session 14)

- **Symptom:** decode was assumed to be *overhead-bound* — a 3124-node graph walk
  with a `std::map<string,Tensor>` tape and per-op allocation. Plan: a flat
  integer-indexed execution plan, expected ~5–10×.
- **Root cause / finding:** swapping the tape `std::map → unordered_map` changed
  nothing (146 → 156 ms, noise). The graph walk / lookups are *not* the cost —
  it's real kernel work (matmul, LayerNorm, the Contiguous copies of C8).
- **Fix:** abandoned the flat-plan refactor; redirected to kernel-level work
  (fused LayerNorm, the Contiguous fix).
- **Lesson:** test the cheap version of a big refactor's premise *before*
  building it. Five minutes of `unordered_map` saved days of flat-plan rewrite.

## C6 — Profiler's O(n²) activation accounting hid the real costs (Session 13→14)

- **Symptom:** profiled decode showed "Constant" ops costing ~889 ms and matmul
  near-zero — led to the wrong conclusion "matmul is <1% of decode."
- **Root cause:** the profiler summed live-tensor bytes after *every* op by
  scanning the whole tape — O(tape) per op, O(n²) per inference. On a 3124-node
  graph this dwarfed real op time and inflated the cheap (early) ops.
- **Fix:** `Profiler::SetTrackActivationBytes(false)` for latency profiling. The
  accurate breakdown: matmul ~35%, LayerNorm/pointwise ~36%, Gather ~14%.
- **Lesson:** instrument the instrument. Verify profiler overhead is negligible
  before trusting per-op numbers — a bad profiler produces confidently-wrong
  conclusions.

## C5 — AMX microbench measured the wrong sgemv variant (Session 12→13)

- **Symptom:** Session 12's microbench showed `cblas_sgemv` faster than a
  single-row `cblas_sgemm`, so the "AMX-aware decode kernel" routed M==1 Gemms
  through sgemv. It made GPT-2 decode *slower* (160 vs 148 ms).
- **Root cause:** the microbench measured `sgemv` in `CblasNoTrans` form
  (`y = A·x`, contiguous). But GPT-2's weights are `transB=0` (`[K,N]`), so the
  matrix-vector product needs `CblasTrans` (column-strided access), which
  Accelerate does *not* accelerate — slower than sgemm.
- **Fix:** gate the GEMV dispatch to `trans_b==true` (the layout where the fast
  NoTrans path applies). GPT-2 stays on sgemm.
- **Lesson:** microbenchmark the exact variant the real workload uses. Weight
  layout (transB) decides which BLAS path you actually hit.

## C4 — Transpose allocated a `Shape` per element (Session 13)

- **Symptom:** GPT-2 decode was ~14.8 s/token; the profile blamed `Transpose`
  for ~85%.
- **Root cause:** GPT-2's tied LM head exports as `Transpose(wte [50257,768]) →
  MatMul`, recomputed every step — through a Transpose kernel that heap-allocated
  a `Shape` (a `std::vector`) *per output element*: 38.6 M allocations per step.
- **Fix:** (1) rewrote Transpose to be allocation-free (odometer-stepped input
  offset); (2) added a constant-folding pass so `Transpose(initializer)` is
  computed once at build time, not per step.
- **Impact:** 14349 → 148 ms/token (97×).
- **Lesson:** per-element heap allocation in a hot loop is catastrophic at scale;
  and constants should never be recomputed every inference.

## C3 — Concat heap overflow, latent for 7 sessions (Session 11)

- **Symptom:** GPT-2 KV-cache decode SIGABRT'd (heap corruption).
- **Root cause:** `Concat` had a dead first loop that wrote past the output
  buffer whenever a tensor had `outer > 1`. Never fired in DistilBERT — all its
  Concats are along axis 0 (`outer == 1`). GPT-2 concatenates past⨁new K/V along
  the sequence axis of a `[1,12,…,64]` tensor → `outer == 12` → overflow.
- **Fix:** removed the dead loop; the correct second loop does all the work.
- **Lesson:** "all v1 tests pass" ≠ "kernels are correct on all shapes." A new
  model exercises new shape regimes that expose latent bugs.

## C2 — DistilBERT had 555 nodes, not ~100 (Session 2)

- **Symptom:** estimated ~100 nodes; the real model had 555 with 24 op types.
- **Root cause:** the ONNX export decomposes high-level ops — GELU exports as a
  7-node Erf chain, LayerNorm as ~9 nodes, etc.
- **Fix:** none needed; reset expectations and built shape inference + kernels for
  all 24 op types. (Later the decompositions became the *fusion* targets.)
- **Lesson:** never estimate op count from the architecture diagram; inspect the
  actual exported graph. The decompositions you find are future fusion wins.

## C1 — int64 elementwise needed for shape arithmetic (Session 10)

- **Symptom:** GPT-2 forward pass failed — `Add`/`Sub`/`Mul` invoked on int64
  tensors (shape scalars), but the elementwise kernels were fp32-only.
- **Root cause:** GPT-2's dynamic-shape plumbing does integer arithmetic on
  shape values (int64), unlike DistilBERT.
- **Fix:** templated `BinaryBroadcast` over dtype; Add/Sub/Mul/Div/Pow now
  dispatch to fp32 or int64.
- **Lesson:** "elementwise = float math" is an encoder-only assumption; decoder
  shape-handling needs integer elementwise.
