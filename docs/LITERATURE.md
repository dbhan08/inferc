# Literature & prior-art map (novelty gate)

Compiled before committing to a paper, to answer one question honestly: **is this
already done?** Fresh benchmark numbers (this machine, reproduced) at the bottom.

## Reproduced results (M1, 4P+4E, ORT 1.25.1, n=30) — not "AI slop", re-run 2026-05-26

| workload | inferc | ORT 1-thread | ORT all-core (8) | verdict |
|---|---|---|---|---|
| DistilBERT-SST2, B=1 S=128 | **40.5 ms** | 121.4 ms (**3.00× faster**) | 34.6 ms (**1.17× slower**) | byte-exact |
| GPT-2 decode, B=1 (per token) | 28.4 ms | 11.2 ms (**2.5× slower**) | 10.8 ms | known overhead gap |
| custom AMX GEMM, FFN 128×3072×768 | 115 GFLOPs | — | Accelerate 788 (**0.15×**) | bit-exact, issue-bound |

Key confirmed fact: **ORT-CPU/MLAS on ARM64 uses its own NEON kernels**
(`aarch64/SgemmKernelNeon.S`; NEON/DOT/I8MM/SVE dispatch), **not Apple Accelerate,
not AMX**. So inferc's 3× win over single-thread ORT is *AMX-via-Accelerate vs
MLAS-NEON*; all-core ORT closes the gap with 8-way NEON parallelism. (Intel "AMX"
in ORT 1.14 is the x86 Xeon AMX — unrelated to Apple AMX.)

## Prior art (what's already published)

**AMX characterization & microbenchmarks**
- Zhou, J. *Performance Analysis of the Apple AMX Matrix Accelerator.* MIT MEng
  thesis, 2025. — AMX load/store/compute throughput, matrix vs vector mode, ILP
  effects; a GEMM with masked outer products + overlapping tiles that beats Accelerate.
- *Apple vs. Oranges: Evaluating the Apple Silicon M-Series for HPC.* arXiv 2502.05317.
- philipturner/amx-benchmarks; corsix/amx + dougallj (RE of the ISA).

**Custom AMX GEMM**
- Zhou 2025 (dense fp32, beats Accelerate on select shapes).
- *Accelerating Sparse Ternary GEMM for Quantized ML on Apple Silicon.* arXiv
  2510.06957 — custom AMX kernels for ternary quant, ~50% peak on M1, vs BLAS/ORT.

**LLM/transformer inference characterization on Apple Silicon**
- *Benchmarking and Characterization of LLM Inference on Apple Silicon.* ACM POMACS
  2025 (10.1145/3771563). M2 Ultra/Max, M4 Pro vs NVIDIA; 8B–405B; 14 quant schemes;
  ALU util / bandwidth / cache residency. **Claims "first thorough characterization
  of Apple Silicon for on-device inference."** GPU/unified-memory focused, large models.
- *Profiling LLM Inference on Apple Silicon: A Quantization Perspective.* arXiv 2508.08531.
- *Benchmarking On-Device ML on Apple Silicon with MLX.* arXiv 2510.18921 —
  BERT/RoBERTa/XLM-R, MLX vs PyTorch, CPU & GPU (no AMX, no ORT, no fusion).

**Accelerate(=AMX)-backed inference & custom CPU transformer runtimes**
- llama.cpp / ggml — Apple Silicon path uses Accelerate `cblas_sgemm` (∴ AMX) + NEON + Metal.
- Apple BNNS Graph — real-time ML inference on CPU.
- *Fast DistilBERT on CPUs.* arXiv 2211.07715 — custom sparse/quant transformer
  runtime, up to **4.1× over ORT** (x86, not Apple).

## Honest novelty assessment

Nearly every *scientific* angle inferc could claim is already published:
- Apple-Silicon inference characterization → POMACS 2025 (explicitly claims "first").
- Custom AMX GEMM (dense & ternary) → Zhou 2025, arXiv 2510.06957.
- Accelerate(AMX)-backed inference → llama.cpp (in production).
- Custom CPU transformer runtime beating ORT → Fast-DistilBERT (x86).
- ORT-CPU underuses AMX (NEON only) → derivable from MLAS source.

## Lit scan 2026-06-07 — additional prior art + angle assessment

**Other hand-written Apple AMX GEMMs (rigor — must position vs these):**
- Tencent **ncnn** `gemm_arm_amx.cpp` (+ `gemm_fp16s_amx.h`, `gemm_bf16s_fp16s_amx.h`)
  — production fp16/bf16 AMX kernels for Apple Silicon. **fp16/bf16, not fp32**,
  so NOT a Paper-1 baseline; it is the baseline-to-beat for the fp16 Paper 2
  (alongside BNNS fp16). Acknowledge in related work.
- Zhou MIT thesis 2025 (cited) — masked outer products.
- Sparse-Ternary AMX, arXiv 2510.06957 (cited).

**Quantized/sparse AMX (Paper-2 direction; richer than fp16 alone):**
- **SparAMX**, arXiv 2502.12444 — unstructured sparsity on AMX, decode-phase,
  INT8 (Intel AMX). Sparse-prefill is an open Apple-AMX angle.
- Fine-grained **codebook quantization for Arm CPUs**, arXiv 2501.00032.

**GEMM techniques assessed for OUR M=128 prefill (and why they don't apply):**
- **Space-filling-curve / cache-oblivious traversal** (SFC-CA, arXiv 2601.16294;
  ~1.8x on large-M CPU LLM GEMM): needs 2D operand reuse. At M=128, A is
  L2-resident and B is streamed once (no reuse), and LM-head is **AMX-compute-
  bound** (Accelerate ~900 ≈ AMX peak ~924), so traversal order cannot help.
  Confirms N≫K is at the compute ceiling, not a cache-blocking problem.
- **Split-K** (GemLite/TorchAO): alternative parallel axis over K. Marginal for
  us — N-panel splitting already saturates the P-cluster at M=128, N≥2048.

**Recent Apple-Silicon eval (related work to add):**
- Native LLM/MLLM inference at scale on Apple Silicon, arXiv 2601.19139.
- Domain-specific architectures on Apple Silicon, arXiv 2511.13450.
- CPUs outperform GPUs for on-device LLM inference, arXiv 2505.06461.

**Net:** the literature confirms the BLIS+packing approach and confirms there is
**no fp32 algorithmic lever for N≫K** (compute-bound at the AMX ceiling). The
N≫K headroom is precision/sparsity → Paper 2. Paper-1 fp32 baselines
(Accelerate AMX + OpenBLAS NEON) are correct; add an ncnn acknowledgment.

## (original novelty assessment, May 2026)

**The unoccupied sliver:** a *byte-exact, ONNX-graph-level* CPU runtime on M1 that
isolates **why ORT-CPU leaves AMX on the table** for small encoder models (MLAS-NEON
vs Accelerate), with graph-fusion + an honest engineering teardown (the negative
results: fp16 doesn't engage, naive AMX is issue-bound and plateaus at 0.15×
Accelerate, decode is overhead-bound). This is a **measurement + systems-engineering
study of known components** — workshop/tech-report/portfolio grade, *not* a novel
research contribution. No false-novelty claim survives contact with the prior art above.
