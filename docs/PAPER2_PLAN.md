# Paper 2 — Plan

**Working title:** *A Fused Codebook-Gather Matmul on the Apple AMX Coprocessor: Low-Bit LLM GEMM Faster than the Production Engine at Small-Batch Prefill*

**Companion to Paper 1** ("Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX"). Paper 1 = bit-exact fp32 prefill GEMM beating Accelerate. Paper 2 = a *new AMX primitive* for low-bit inference.

---

## 1. Thesis (one sentence)
Apple AMX's undocumented **indexed-load** turns the matrix instruction into a **fused codebook-gather** — one operand is gathered per-lane from a 16-entry register codebook *inside* the outer product, free — which we exploit to run non-uniform 4-bit (NF4/codebook) LLM GEMM **up to ~4× faster single-thread (≈1.5× multi-thread) than llama.cpp's production Q4 kernel at small-batch/prefill on the M1**, bit-exact and quantizer-agnostic.

## 2. Contributions (what is genuinely ours)
1. **A reverse-engineered mechanism:** the AMX MATFP/MATINT indexed-load (bit 53) gathers a per-lane 16-entry codebook value into the outer product, fused, at plain-matmul throughput. First characterization + first use (corsix documents the bitfield only; Zhou's thesis skips it).
2. **A kernel** that realizes non-uniform 4-bit codebook GEMM with this primitive, profiled to ~74% of the AMX MATFP ceiling, with batch-adaptive blocking.
3. **A fair, apples-to-apples evaluation** vs the production engine (ggml/llama.cpp Q4) on M1 — including where it *loses* (decode) and why (lane utilization, AMX block count).

**Not claimed:** a new quantization algorithm, or SOTA quantization *quality*. Quality is portable/upstream; we *consume* it.

## 3. The mechanism (Section 3)
- AMX indexed-load encoding (bit 53 enable, bit 48 = 4-bit indices, bit 47 = X/Y, bits 49–51 = table reg). Verified semantics: operand register holds indices; `x[i] = codebook[idx[i]]`; outer product proceeds. Bench: `amx_matfp_indexed.cc` (gather verified bit-exact), `amx_matfp_indexed_cost.cc` (free: indexed ≈ plain MATFP, ~1.15 cyc).
- **Correction worth stating:** FMA does *not* support indexed-load (corsix `fma.c`); only MATFP/MATINT do. (`amx_indexed_*` document the dead-end.)
- Distinct from: NEON `tbl`/genlut (separate op), and from the matmul-engine alternatives below.

## 4. Speed results (Section 4 — the core)
Apples-to-apples vs **ggml Q4_0** on M1 (NEON DOTPROD — verified: no Apple-AMX/Accelerate in ggml's quant path; M1 `FEAT_I8MM=0`). Raced ggml's **best** path (repacked `q4_0_4x4`, not the per-row strawman). K=2048, N=8192.

**Optimized M-curve, ST + MT** (`amx_mcurve_opt.cc`, `amx_mcurve_opt_mt.cc`; all ggml measured):

| M | our ST | ggml ST | **ST×** | our MT | ggml MT | **MT×** |
|---|---|---|---|---|---|---|
| 1 | 0.48 | 0.25 | 0.52× | 0.37 | 0.18 | 0.48× (decode → NEON) |
| 4 | 0.51 | 0.61 | 1.21× | 0.37 | 0.25 | 0.67× |
| 16 | 0.51 | 2.02 | **3.94×** | 0.38 | 0.59 | **1.57×** |
| 32 | 1.11 | 4.02 | 3.61× | 0.74 | 1.09 | 1.47× |
| 64 | 2.95 | 8.03 | 2.72× | 1.99 | 2.06 | 1.04× |

(ms, K=2048 N=8192. ST/MT = our speedup over ggml at that thread count.)

- **ST: win M≥4, up to ~3.9× (peak M=16); lose only M=1 decode. MT: win the M=16-32 sweet spot (~1.5×), tie at M=64, lose decode/small.**
- Mechanistic: ggml Q4 = M independent dot-products (~linear in M); our outer-product amortizes the weight across the batch. M<16 wastes the 16-wide AMX M-dimension.
- **Optimization story (profile-driven):** baseline M=16 was issue/load-bound (~22% of peak, 3 AMX instr/MATFP). Latency hypothesis (bank-ILP) failed (1.03×) → re-diagnosed load-bound → (a) N-tile blocking amortizes the A load, (b) idx-amortization: 4 tiles' indices in one X register feed 4 MATFPs (1 LDX), → 1.5 instr/MATFP, **3.3× to 1155 GFLOP/s**. Requires compile-time-specialized blocking (runtime-param = 2× slower). Benches: `amx_codebook_ilp.cc`.
- **Multi-thread** (full curve above): wins ~1.5x at M=16-32, ties M=64 — honestly capped: M1 has ~2 AMX blocks (P+E), so AMX MT scales ~1.2×, while ggml NEON scales ~3× across 8 cores. **State this limit plainly.**

## 5. Quality results (Section 5 — portable, upstream)
The kernel is **bit-exact to `A·dequant(W)`** (verified on real OPT-125M weights end-to-end, `amx_real_kernel_test.cc`, 5.2e-7), so it reproduces *any* scalar codebook quantizer's quality exactly — quality is hardware-independent.
- Standard eval (OPT-125M, full wikitext-2, ctx 2048, fp16 head; baseline 27.66 ≈ published FP16 27.65): our NF4+GPTQ per-channel **+7.39%** vs published vanilla GPTQ **+12.4%** — beats the standard method. SOTA GANQ is +3.4% (~2× better); since it's non-uniform scalar, **its codebook runs on our kernel too.** Bench: `gptq_eval.py`, `codebook_perplexity.py`.
- **Scalar ceiling (state honestly):** vector quant (AQLM/QuIP#, <1%) cannot run on the scalar gather — out of scope.

## 6. Honest scope / limitations (own these in the paper)
- **Regime:** wins small-batch/prefill (M≥4); decode (M=1) belongs to NEON GEMV (lane waste).
- **MT:** advantage shrinks (AMX block count vs NEON core count).
- **Hardware:** M1-class AMX (undocumented `.word`); M4 uses SME (different).
- **Quality:** our quantizer is GPTQ-class, not SOTA; the kernel is quantizer-agnostic.

## 7. Related work / positioning
- **AMX RE:** corsix/amx (encodings, no costs), Zhou MIT thesis 2025 (throughput, skips indexed-load/genlut). We add the indexed-load cost + use.
- **Matrix-engine low-bit:** ARM SME `LUTI4→MOPA` is **two instructions** (lookup → vector reg → separate matmul); ours fuses gather into the matmul in one. T-MAC/LUT-GEMM table the *dot-product* (different mechanism). FLUTE (GPU) gathers a value but as a *separate* step. Fused codebook-gather-into-MAC otherwise exists only in research ASIC/FPGA.
- **Quantizers (upstream, we consume):** GPTQ (2210.17323), AWQ, GANQ (2501.12956), NF4/QLoRA.
- **M1 4-bit baseline:** ggml/llama.cpp Q4_0 (repacked DOTPROD).

## 8. Figures / tables
- F1: the mechanism (indexed-load → fused gather diagram).
- F2: M-curve speedup vs ggml (the headline) — ST and MT.
- T1: indexed vs plain MATFP cost (free-gather evidence).
- T2: quality vs published GPTQ/GANQ (standard eval).
- F3: profile/optimization waterfall (baseline → blocked → idx-amort).

## 9. Remaining experiments
**Done:** full ST+MT M-curve (`amx_mcurve_opt{,_mt}.cc`); ggml M=32 measured (4.02 ms). The optimized
reference kernel is `amx_mcurve_opt{,_mt}.cc` (specialized templated blocking); the `gptq`/`pergroup`
files are the earlier per-channel/per-group *correctness* demos (kept for the bit-exact + accuracy story).
**Optional / future:** a second shape (real attn/FFN dims) and second model for robustness; real-model
end-to-end tok/s (kernel integrated into a runtime) — larger effort, not needed for the core claim.

## 10. Honesty guardrails
- Lead with the *mechanism + fair speed*, not a quality-SOTA claim.
- Always cite the **fair (repacked) ggml**, never the per-row strawman.
- State the decode loss and the MT block-limit up front, not buried.
- "Quantizer-agnostic; quality is upstream and portable" — don't claim we out-quantize anyone.

## Repro
All benches in `bench/amx/`. ggml baseline: build llama.cpp ggml (CPU), `ggml_q4_repack_bench.cpp`. Quality: `gptq_eval.py` (needs torch/transformers/datasets, CPU). Pushed through current HEAD.
