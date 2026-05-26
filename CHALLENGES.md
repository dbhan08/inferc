# Challenge log

A running record of the non-obvious bugs, performance pathologies, and
false-starts hit while building inferc — what broke, why, the fix, and the
lesson. Kept because the *why-it-broke* is the interesting part (and the best
interview / paper material). Newest first.

---

## C12 — the ORT gap was naive memory kernels, not fusion (Session 18)

- **Question:** why is inferc 1.3× slower than ORT-CPU on DistilBERT — what to fix?
- **Profiling (the right move):** dumped ORT's *optimized* graph and per-op times.
  Findings: (1) ORT's CPU EP does **not** fuse attention — its graph is the same
  shape as inferc's; (2) ORT spends **~115 ms in MatMul and ~7 ms on everything
  else**, while inferc *beats* ORT's matmul (~2×, via AMX) but spent ~100 ms on
  non-matmul kernels. The gap was entirely **naive memory kernels**, not fusion.
- **Fix:** Transpose and broadcast-Add still copied single 4-byte elements with
  strided reads. Added a **contiguous-run** path (bulk `memcpy` / `vDSP_vadd`
  when the inner axis is contiguous). Transpose 19.5→5.7 ms, Add 21.7→<5 ms.
  DistilBERT 170 → 113 ms.
- **The honesty caveat (account for the comparison!):** "beats ORT" only holds
  **single-threaded** (113 vs 122 ms). We had pinned ORT to 1 thread; ORT scales
  to **35 ms on 8 cores** while inferc is single-threaded → ORT is 3.2× faster on
  the full machine. And the matmul win is **Apple AMX (hardware)**, not algorithm
  (ORT's portable MLAS uses NEON on ARM). Verified inferc is genuinely
  single-threaded (latency flat vs `VECLIB_MAXIMUM_THREADS`), so the 1-thread
  comparison is at least *fair*.
- **Lessons:** (1) profile the competitor's *structure* before optimizing — the
  obvious assumption (ORT wins via fusion) was wrong. (2) State perf claims with
  their confounds: thread count and hardware path. A single-threaded + AMX win on
  M1 is real but is NOT "beats multi-core ORT." The bench harness now takes
  `--threads` so the full-machine comparison is reproducible, not hidden.

## C11 — fast-erf GELU approximation was *slower* (Session 18, reverted)

- **Hypothesis:** `FusedMatMulAddGELU` (49 ms, DistilBERT's biggest op) was bound
  by the ~2.4M per-element `std::erf` calls in its GELU sweep. Plan: replace with
  the Abramowitz-Stegun erf approximation, batching its `exp(-z²)` through
  vectorized `vvexpf`.
- **Result:** it got *slower* — `FusedMatMulAddGELU` 49 → 64 ms. Reverted.
- **Why:** the premise was wrong. clang's libm `erf` is already efficient, so erf
  wasn't the bulk — the op is largely **sgemm-bound** (the FFN GEMM at N=3072 is
  cache-bound, ~50% of AMX peak per the amx-probe sweep). The approximation added
  two extra memory passes over a 1.5 MB buffer plus a per-element divide, which
  outweighed any erf saving.
- **Lesson:** measure the *split* before optimizing a fused op — don't assume the
  transcendental dominates. (And, again: bench the kernel in isolation before
  committing to an approach.) This is also why DistilBERT's last ~30% to ORT is
  hard: it's sgemm/memory tail + attention-materialization, not a single
  pathology.

## C10 — broadcast elementwise recomputed the offset per element (Session 17)

- **Symptom:** DistilBERT spent **Add 357 ms + Where 169 ms + Expand 119 ms + Div
  69 ms ≈ 714 ms** of an 838 ms inference — i.e. most of the time in "pointwise"
  ops, despite tiny per-element work.
- **Root cause:** the *broadcast* path of these kernels walked the output with an
  `IndexIterator` and recomputed each input's flat offset (`BroadcastOffset`,
  O(rank)) for *every* element. The attention-mask Add/Where on `[1,12,128,128]`
  (~2.4 M elements) paid that O(rank) cost millions of times. (Equal-shape inputs
  already had a tight fast path — these were the broadcasting ops.)
- **Fix:** odometer-step each input's offset (precompute per-output-axis strides,
  0 on broadcast axes; add on increment, subtract on rollover) — O(1) amortized
  per element. Same technique as the Transpose fix (C4). Plus vDSP (`vDSP_vadd`
  etc.) for the equal-shape float path and vForce (`vvsqrtf`/`vvtanhf`) for unary
  transcendentals.
- **Impact:** Add 357→24 ms, Where 169→12 ms, Expand 119→7 ms, Div 69→5 ms.
  DistilBERT 838→170 ms — from **6.5× to 1.29× off ORT**.
- **Lesson:** broadcasting is a hot loop; never recompute strided offsets
  per element. (And `vDSP_vsub`/`vDSP_vdiv` take the B operand *first* — a
  classic arg-order footgun the equal-shape kernel tests guard against.)

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
