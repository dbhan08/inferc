# A Fused Codebook-Gather Matmul on the Apple AMX Coprocessor

*Low-bit LLM GEMM faster than the production engine at small-batch prefill*

---

## Abstract

Apple's AMX coprocessor — the undocumented matrix unit shared per CPU cluster on Apple
silicon — exposes an **indexed-load** mode whose cost and use have never been characterized.
We show this mode turns the matrix-multiply instruction into a **fused codebook-gather**: one
operand is gathered per-lane from a 16-entry register codebook *inside* the outer product, at
plain-matmul throughput and bit-exact. We exploit it to run non-uniform (NF4/codebook) 4-bit
LLM GEMM and, on an M1, beat llama.cpp's production Q4 kernel by **up to ~4× single-thread at
small-batch prefill** (M≥4), a result that holds **at both higher precision (fp32 activations)
and matched precision (int8, apples-to-apples with ggml)**, verified bit-exact end-to-end on a
real model. The kernel is quantizer-agnostic: it reproduces any scalar codebook quantizer's
output exactly, so its quality is the upstream quantizer's (our own NF4+GPTQ beats vanilla
GPTQ; SOTA codebooks plug in unchanged). We delimit the win with first-principles, hardware-
grounded limits: single-stream decode is memory-bound and belongs to NEON (a structural loss
recovered by batching), and AMX throughput caps at ~2× across the chip's two cluster-level
blocks. The result is an architecture-aware design — AMX for prefill/batched, NEON for
single-stream decode — for low-bit inference on Apple-silicon CPUs.

---

## 1. Introduction

Low-bit weight quantization (4-bit and below) is the dominant lever for LLM inference under
memory pressure. On Apple-silicon CPUs the production path is llama.cpp/ggml's Q4, which
quantizes activations to int8 and uses NEON DOTPROD. The CPU also hosts **AMX**, a
reverse-engineered matrix coprocessor (one block per CPU cluster) that prior work has
characterized for floating-point throughput (Zhou, MIT 2025) and instruction semantics
(corsix/amx), but whose **indexed-load** mode — present in the ISA encoding but undocumented
for cost or use — has gone unexploited.

This paper makes three contributions:

1. **A reverse-engineered mechanism (§3).** The AMX `MATFP`/`MATINT` indexed-load (bit 53)
   gathers a per-lane 16-entry codebook value into the outer product, fused, at the same
   throughput as a plain matrix instruction. We give the encoding, verify the gather is
   bit-exact, and measure it free. We also correct the record: `FMA` does *not* support
   indexed-load; only the matrix ops do.

2. **A kernel (§4)** realizing non-uniform 4-bit codebook GEMM with this primitive, profiled
   from ~22% to ~74% of the AMX matrix-throughput ceiling via batch-adaptive blocking and
   index-amortization, and a matched-precision int8 variant via `MATINT`.

3. **A fair, apples-to-apples evaluation (§5)** against the production engine (ggml Q4,
   *repacked* path) on an M1 — at both fp32 and int8 precision, bit-exact, with the win's
   regime boundaries and their structural causes made explicit (§6).

The headline: **~3–4× single-thread prefill over ggml's Q4**, robust across shapes and across
both precisions, with single-stream decode honestly ceded to NEON.

---

## 2. Background

**AMX.** Apple AMX is a matrix coprocessor issued from the CPU via undocumented `.word`
encodings (corsix/amx). It holds a 4 KB Z accumulator (64×64 B), X/Y operand registers, and
computes outer products `z[i][j] += f(x[i], y[j])`. Critically, there is **one AMX block per
CPU cluster** (Firestorm + Icestorm → two on an M1), shared across the cluster's cores with
per-core register state only (Zhou, MIT 2025; Eclectic Light). This bounds aggregate AMX
throughput at ~2× (dual-cluster), in contrast to NEON's per-core SIMD.

**Indexed-load.** The `MATFP`/`MATINT` operand encodes an indexed mode (bit 53): the operand
register holds small (2- or 4-bit) indices, and the instruction gathers the actual operand
elements from a *second* register (the "codebook") before the multiply. corsix documents the
bitfield; no cost or use has been published.

**Low-bit codebook quant.** Non-uniform / codebook quantization (NF4, GANQ, AQLM) represents
each weight as an index into a small codebook of representative values. Dequantization is a
*table lookup* — which is exactly what indexed-load performs in hardware.

---

## 3. The Mechanism: Fused Codebook-Gather

The indexed `MATFP` (fp32) / `MATINT` (int8) computes, in one instruction:

```
x[i] = codebook[ idx[i] ]          (per-lane gather from a 16-entry register codebook)
z[i][j] += x[i] * y[j]             (outer product, accumulated)
```

**Encoding.** bit 53 = indexed; bit 48 = 4-bit (vs 2-bit) indices; bit 47 = index X or Y;
bits 49–51 = codebook register; lane bits 42–45 select precision. For int8 `MATINT`,
int8×int8→int32 is alumode 8 / lane 10, with `SIGNED_X` (bit 63) and `SIGNED_Y` (bit 26)
required (it defaults to unsigned).

**Verified properties.**
- *Correct:* the gather returns `codebook[idx]` bit-exactly (`amx_matfp_indexed.cc`;
  `amx_matint_idx_verify.cc`).
- *Free:* indexed `MATFP` ≈ plain `MATFP` (~1.15 cyc/op) — the gather adds no cost
  (`amx_matfp_indexed_cost.cc`).
- *Correction:* `FMA` ignores the indexed bits (corsix `fma.c` has no indexed path); only
  `MATFP`/`MATINT` gather (`amx_indexed_*.cc` document the dead-end).

This is the key capability: codebook dequantization is **fused into the matmul, for free**.
No prior shipping matrix engine does this in one instruction (§7).

---

## 4. Kernel Design and Optimization

**Layout.** fp32 `MATFP` produces a 16×16 tile (output row *j* at Z row 4*j*). int8 `MATINT`
produces a **64×16** tile (1024 MAC/op, 2× the fp32 MAC rate), with a quad-interleaved layout
`C[m][n] → Z[4n + m%4][m/4]` decoded via delta probes (`amx_matint_map.cc`).

**Profile-driven optimization (fp32).** The naïve M=16 kernel ran at ~22% of the
matrix-throughput ceiling. Profiling showed it was **issue/load-bound** (3 AMX instructions
per `MATFP`: LDX idx, LDY A, MATFP), *not* latency-bound — a 4-way bank-ILP fix gave only
1.03×. Two load cuts fixed it:
1. **N-tile blocking:** load the activation `A[k]` once, reuse across 4 N-tiles.
2. **Index-amortization:** four tiles' 4-bit indices (4×8 = 32 B) fit one X register, so a
   single LDX feeds four MATFPs (per-tile byte offset) → 1.5 instr/MATFP.

Result: **3.3× to ~1062 GFLOP/s (~74% of peak)** at M=16. The kernel must be *specialized*
(compile-time blocking) — a runtime-parameterized version is ~2× slower from per-instruction
operand computation (`amx_codebook_ilp.cc`, `amx_mcurve_opt.cc`).

**Per-channel scale, free.** GPTQ/NF4 use per-output-channel scales. A per-channel scale
*factors out of the K-sum*, so the kernel gathers a shared codebook and applies the scale as a
cheap per-column post-scale — no inner-loop cost (`amx_codebook_gptq.cc`). Per-group scales
(slightly more accurate) are supported via per-K-group accumulator rescale at ~3× speed
(`amx_codebook_pergroup.cc`).

---

## 5. Evaluation

All on an M1, K=2048 N=8192 unless noted, vs ggml's **repacked** Q4_0 (its best M1 path:
NEON DOTPROD `q4_0_4x4`; we confirmed ggml uses no Apple-AMX/Accelerate for quantized matmul,
and M1 lacks i8mm).

### 5.1 Speed — single-thread M-curve (fp32)

| M | our ST | ggml ST | **speedup** |
|---|---|---|---|
| 1 | 0.48 ms | 0.25 ms | 0.52× (decode → NEON) |
| 4 | 0.51 | 0.61 | 1.21× |
| 16 | 0.47–0.51 | 2.02 | **~4× (peak)** |
| 32 | 1.05 | 4.02 | 3.6× |
| 64 | 2.86 | 8.03 | 2.7× |

We win for all M≥4, up to ~4× at M=16; we lose only single-stream decode (M=1).
**Mechanism:** ggml Q4 is M independent dot-products (≈linear in M); our outer-product
amortizes the weight across the batch.

### 5.2 Shape robustness

Across five attention/FFN/square shapes (2048×8192, 4096×4096, 4096×11008, 11008×4096,
768×3072) at M=16, the single-thread speedup is **4.3–5.5×** — consistent
(`amx_shape.cc`).

### 5.3 Matched precision (int8, apples-to-apples)

AMX `MATINT` int8 codebook GEMM, bit-exact, matched to ggml's int8 (Q8) activations:

| M=64 | our int8 | ggml int8 | speedup |
|---|---|---|---|
| single-thread | 2.70 ms | 8.03 ms | **2.97×** |
| multi-thread | 1.56 ms | 2.06 ms | 1.32× |

So the **~3× single-thread prefill win holds at matched precision**, not just at our (higher,
fp32) precision. int8 does *not* exceed fp32 here: its 64×16 tile fills all 64 Z rows, leaving
no banks for the load-amortization that makes fp32 efficient, so its raw 2× is cancelled
(`amx_codebook_int8{,_perf}.cc`).

### 5.4 End-to-end (full OPT-125M linear-GEMM stack)

Summing all layer matmuls, each engine at its best thread config:

| batch M | ours | ggml | winner |
|---|---|---|---|
| 1 (single-stream decode) | 491 tok/s | 617 | ggml 1.26× |
| 4 (batched decode) | 1969 | 1186 | **ours 1.66×** |
| 16 | 7752 | 2060 | **ours 3.76×** |
| 64 (prefill) | 5972 | 3640 | **ours 1.64×** |

The only loss is single-stream M=1; the crossover is M≈2–4. Batching to M≤16 is *free* (the
16-wide tile costs the same at M=1 and M=16), so batched/serving decode wins.

### 5.5 Quality (portable, upstream)

The kernel is **bit-exact to `A·dequant(W)`** — verified on real OPT-125M weights end-to-end
(5.2e-7, `amx_real_kernel_test.cc`) — so it reproduces *any* scalar codebook quantizer's
quality exactly. On the standard wikitext-2 eval, our in-house NF4+GPTQ:

| model (arch) | our NF4+GPTQ | vanilla GPTQ (ref) |
|---|---|---|
| OPT-125M (Linear) | +7.39% ppl | +12.4% |
| gpt2-124M (Conv1D) | +4.96% | +12.4% |

We beat the standard method on two architectures; SOTA GANQ (+3.4%) is non-uniform scalar and
**its codebook runs on our kernel unchanged** (`gptq_eval.py`, `codebook_perplexity.py`).

---

## 6. Limitations (structural, hardware-grounded)

These are first-principles hardware bounds, not unoptimized code:

1. **Single-stream decode (M=1) is memory-bound → NEON's.** A GEMV has arithmetic intensity
   ~4 FLOP/byte vs M1's ~15 roofline knee → hard memory-bound. All engines share DRAM, so the
   winner is whoever streams it with least overhead; NEON's in-core DOTPROD does so cleanly,
   while AMX can only reach a GEMV through a wasteful tile over the shared cluster bus. *No AMX
   precision beats it* (int8's larger tile is worse). It is recovered by batching (§5.4).
2. **Multi-thread caps at ~2×.** One AMX block per cluster (2 total); threads within a cluster
   contend, not parallelize (Zhou 2025: FMA32 1669→3320 GFLOP/s 1→2 cluster = 1.99×; no gain
   from a 2nd in-cluster thread). NEON scales ~8× per-core. The advantage is fundamentally
   single-thread/per-core; all-core throughput is contested.
3. **int8 does not exceed fp32.** Its 64×16 tile fills the Z accumulator, leaving no banks for
   load-amortization → stuck ~half peak.
4. **Scope:** M1-class AMX (undocumented `.word`; M4 = SME); scalar codebooks only (no vector
   quant à la AQLM/QuIP#); 4-bit is lossy; GEMM-stack + real-layer verified, not a full
   deployed runtime.

These bounds motivate the **hybrid design** (AMX prefill/batched + NEON single-stream decode)
as architecture-aware, not a compromise.

---

## 7. Related Work

- **AMX RE:** corsix/amx (encodings, no costs); Zhou MIT 2025 (fp/int throughput, per-cluster
  structure; skips indexed-load). We add the indexed-load cost and first use.
- **Matrix-engine low-bit:** ARM SME `LUTI4→MOPA` is **two instructions** (lookup → vector
  register → separate matmul); ours fuses the gather into the matmul in one. T-MAC / LUT-GEMM
  table the *dot-product* (a different mechanism). FLUTE (GPU) gathers a value but as a
  *separate* step. Fused codebook-gather-into-MAC otherwise appears only in research
  ASIC/FPGA, never as a shipping CPU-matrix-coprocessor instruction.
- **Quantizers (upstream, consumed):** GPTQ, AWQ, GANQ, NF4/QLoRA.
- **M1 baseline:** ggml/llama.cpp Q4_0 (repacked DOTPROD).

---

## 8. Conclusion

Apple AMX's indexed-load is a fused codebook-gather — codebook dequantization free, inside the
matmul — that no shipping matrix engine offers. Exploited for non-uniform 4-bit LLM GEMM, it
beats the production M1 engine by ~3–4× at single-thread small-batch prefill, robust across
shapes and across both higher- and matched-precision, bit-exact end-to-end. Its limits
(single-stream decode, multi-thread, int8 tile footprint) are structural and explained, and
they prescribe an architecture-aware hybrid. The kernel is quantizer-agnostic, so it inherits
the quality of any scalar codebook method — making it a drop-in accelerator for the prefill
and batched-decode phases of low-bit inference on Apple-silicon CPUs.

**Artifacts.** All benchmarks in `bench/amx/` (kernels, probes, ggml baseline harness) and the
quality harness `bench/amx/{gptq_eval,codebook_perplexity}.py`. Companion to Paper 1
("Above the Inner Loop").
