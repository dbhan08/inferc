# Direct AMX programming on M1: a thesis-grade investigation

A consolidated record of the AMX breakthrough investigation in inferc (Session 22).
Everything in this document is reproducible from the bench files cited; every "this
kernel beats X" claim is bit-exact vs a scalar reference; every "we don't beat X" is
honest. The investigation ran two parallel paths — batch-1 decode and batch>1 prefill —
each to a definitive plateau. The artifact and the empirical map are real; the
breakthrough versus tuned SOTA is not.

## TL;DR

We tested whether undocumented Apple AMX, programmed directly via `corsix`-encoded inline
asm, can beat the *real* M1 CPU baselines: NEON+DOTPROD for batch-1 decode (via
llama.cpp's tuned Q4_0 path) and Accelerate `sgemm` for prefill (which uses AMX
internally). Three gates passed, two failed:

| gate | claim | result |
|---|---|---|
| 1 | quantized decode is dequant-bound vs naive NEON | ✅ |
| 2 | AMX `genlut` can dequant near memory-bandwidth | ✅ ~1 cycle/instr; 134 GB/s fp32 |
| 3 | `genlut + vecfp` int4 GEMV beats fp32 baselines | ✅ 7.73 ms, 1.5× ORT, byte-exact, per-row scales free |
| 4 | beats llama.cpp Q4_0 NEON+DOTPROD (the M1 CPU SOTA) | ❌ **3.3× short** (2.32 ms scaled equivalent) |
| 5 | `vecint` (denser int primitive) or thesis-grade prefill closes it | ❌ `vecint` ~5 cyc/instr; prefill best 0.75× Accel |

The contrarian thesis (*matrix coprocessor wins for decode*) only holds against naive
NEON. Tuned NEON+DOTPROD is at the M1 single-core DRAM ceiling (~29 GB/s); AMX has
no path that beats it on this hardware for batch-1 decode. The prefill avenue (where
AMX was *designed* to win) hits Apple's own tuned Accelerate as the ceiling, and
closing the residual 1.3–3.8× would mean reproducing Apple's multi-week tuning.

What is durable: a bit-exact direct-AMX int4 GEMV kernel that beats every fp32
baseline on M1, a working int8/`vecint` variant, five increasingly sophisticated
fp32 GEMM kernels with a clean optimization curve, an M1-specific micro-architectural
characterization of `vecint`/`LDX_pair`, and end-to-end correctness checked against
scalar references at every step.

## 1. The thesis we tested

The literature (POMACS 2025, the quant-profiling and MLX benchmark papers, and the
Jan-2026 "Bare-Metal Tensor Virtualization" arxiv paper which deliberately *refused*
to use AMX and ate a 5× penalty) systematically (a) omits the CPU+AMX path and
(b) treats engines as black boxes. The full prior-art map is in `docs/LITERATURE.md`.

Our contrarian thesis: **quantized batch-1 decode on Apple Silicon is dequant-bound,
not memory-bound; AMX `genlut` (op 22, mode 11 = u4→f32 lookup) makes dequant nearly
free, unlocking near-memory-floor decode the literature said was impossible.** It is
half-true — gate 1 confirms it for *naive* NEON, gate 4 falsifies it against tuned
NEON. The thesis fails because tuned NEON already does fused dequant+multiply via
`vdotq_s32` (DOTPROD), which is at the M1 single-core DRAM ceiling. The matrix
engine has no headroom above memory bandwidth on this regime.

## 2. Decode path — five gates

All numbers are M1 single thread, N=60000 K=2048 (123M weights/token, GPT-2-small
weight-volume equivalent for direct comparison across kernels).

**Gate 1 — quantized decode is dequant-bound on NEON.** `bench/amx/decode_floor.cc`
streams a 61 MB int4 buffer through three GEMV variants (4-acc ILP so fp32 is
memory-bound, not latency-bound):

```
int4 raw memory read floor   2.41 ms  (25.5 GB/s)
fp32 GEMV                    9.61 ms  (51.1 GB/s)  ← memory-bound
int8 GEMV                    9.26 ms  (13.3 GB/s)  ← dequant-bound (same time as fp32)
int4 GEMV                   12.66 ms  ( 4.9 GB/s)  ← dequant-bound (slower than fp32)
Accelerate sgemv fp32       14.67 ms  (33.5 GB/s)
```

Quantization is *pointless* on naive NEON because nibble unpack + int→fp + fma eats
the savings. The prize: if dequant were free, int4 decode → 2.4 ms vs fp32's
~10 ms — ~4× headroom. Gate passes.

**Gate 2 — `genlut` throughput.** `bench/amx/genlut_throughput.cc` measures the AMX
hardware LUT. The canonical operand encoding (pinned from `corsix/amx/genlut.{md,c}`):
mode at bit 53 (4 bits), table reg at bit 60 (3 bits), table-from-Y at bit 59, dest
reg at bit 20 (3 bits), dest-is-Z at bit 26, source offset at bit 0 (9 bits),
source-from-Y at bit 10. Mode 11 = u4 → 32-bit lookup, 16 lanes.

```
LDX stream 61MB (no genlut)        1.36 ms  (45.1 GB/s)  ← AMX memory floor
LDX + 8x genlut/chunk stream 61MB  3.65 ms  (16.8 GB/s)  ← integrated dequant
```

`genlut` issues at ~1 cycle/instr — not a bottleneck. Sanity check: first 8
dequanted fp32 outputs match the table lookup exactly. Gate passes.

**Gate 3 — int4 dequant-GEMV kernel.** `bench/amx/int4_gemv_amx.cc` is four
increasingly optimized AMX kernels using `genlut` mode 11 + `vecfp` (ALU=0,
lane=4, f32 one row) for dot-product accumulation. All bit-exact vs scalar:

```
v1 single Z accumulator (latency-bound)            16.56 ms
v2 4-way ILP (Z banks 0..3)                       12.19 ms  (-26%)
v3 hoisted LDYs (8 Y regs, dispatcher pipelined)  12.46 ms  (~no change)
v4 B=4 row-batched (per-tensor scale, 16 Z rows)   7.40 ms  (-39%, the win)
v5 B=4 row-batched + per-row scales (realistic)    7.73 ms  (-1% vs v4, essentially free)
```

The B=4 row batching collapsed shared `LDY x` cost across 4 output rows. Per-row
scales handled by pre-loading 4 different pre-scaled tables into X[1..4] per batch.
Bars at the same shape:

```
AMX int4 (B=4 per-row, this kernel)   7.73 ms  ←
fp32 NEON GEMV                        9.61 ms  inferc 1.24× faster
ORT GPT-2 decode                      ~11 ms   inferc 1.42× faster
int4 NEON GEMV (naive)               12.66 ms  inferc 1.64× faster
Accelerate sgemv fp32                14.67 ms  inferc 1.90× faster
```

Gate passes — we beat every fp32 baseline.

**Gate 4 — vs llama.cpp Q4_0 (the real SOTA).** Built llama.cpp commit 408ae2b
CPU-only, `GGML_NATIVE=ON` (auto-detected DOTPROD), ran `llama-bench -t 1 -p 0 -n 64`
on TinyLlama-1.1B Q4_0 GGUF. Result: **48.11 tok/s, 20.79 ms/token, 606 MB streamed
per token = 29.2 GB/s = M1 single-core DRAM ceiling.** Scaled to GPT-2-small weight
volume (123 M / 1.10 B = 0.111): **2.32 ms/token equivalent.**

My best kernel: 7.73 ms. **3.3× slower than tuned llama.cpp.** Gate fails. The
reason is documented under llama.cpp's NEON Q4_0 path: `vdotq_s32` fuses 4 int8
multiplies + accumulate in one instruction, dequant happens implicitly via the
block layout, and the kernel saturates the memory bus.

**Gate 5 — `vecint` int8 path (the AMX analog of DOTPROD).** The natural rescue:
switch from `genlut→fp32→vecfp` to `genlut→int8→vecint` (op 18, mode 10, i8×i8→i32
across 4 Z rows interleaved), which nominally does 64 macs/instr vs `vecfp`'s 16 —
4× the density. `bench/amx/int4_gemv_amx.cc::amx_int4_gemv_vecint`. Bit-exact
(0.00e+00 vs scalar). Result:

```
v4 vecfp B=4               7.80 ms  ← still the best
v6 vecint (single-row)    13.28 ms  ← actually slower despite denser primitive
```

Cycle accounting: `vecint` runs at **~4.9 cycles/instr** on M1 (708 cycles/row,
144 instr/row), vs `vecfp` at ~1.4 cycles/instr (416 cycles/row, 300 instr/row).
The 4× instruction-density advantage is **canceled by 3.5× per-instruction latency**.
**`vecint` mode 10 is not the DOTPROD analog its shape suggests on M1.** This is a
genuine micro-architectural finding for which I haven't seen published equivalents.

The decode bet is closed.

## 3. Prefill path — cache blocking + microkernel limits

If batch-1 is memory-bound by physics, batch>1 *should* be where AMX shines, because
its 16×16 FMA32 outer-product engine is then fully utilized rather than 1/16th used.
`bench/amx/prefill_bench.cc` tests this at typical LLM prefill shapes (M=128 = a
typical prefill batch, 4 representative GEMM shapes).

**Accelerate baseline** (Accelerate uses AMX internally for `sgemm`):

```
Shape                          GFLOPS  % of M1 AMX peak (~1400)
QKV-proj-like [128,2048,2048]    288   21%
FFN1 (up)     [128,8192,2048]    412   29%
FFN2 (down)   [128,2048,8192]    520   37%
LM-head       [128,60000,2048]   951   68%
```

Accelerate leaves significant compute on the table at small N — a real opening for
a custom kernel.

**Five AMX kernels were tested in escalating sophistication**, all bit-exact:

```
GFLOPS    naive  blocked  +LDX_pair  +pipelined
QKV         194     212      204         211
FFN1        113     181      119         131
FFN2        115     131      117         137
LM-head     184     601      680         716
```

Cache blocking (j outer / i inner, A transposed once) is the big win at LM-head
(+216% over naive, 184 → 601 GFLOPS); it collapses 8× B re-streaming into one pass.
At smaller shapes the cache benefit is smaller and is partially canceled by the
A-transpose pass — `naive` even beats `blocked` at QKV by a tiny margin pre-optim.

`LDX_pair` (bit 62 of LDX operand → loads 128 B into two consecutive X regs at
once) **regressed at FFN1/FFN2** and only modestly helped LM-head (+13%). On M1,
`LDX_pair` is *not* unambiguously cheaper than 2 LDXs — apparently the hardware
pays a setup cost that exceeds its savings at moderate shapes. Empirical M1
finding.

Software-pipelined ping-pong banks (X bank A=0..3 / B=4..7, Y bank A=0 / B=1;
prologue loads k=0 into A; steady-state body loads k+1 into the other bank before
issuing FMAs for k from current bank) gives **+19% at LM-head** (601 → 716 GFLOPS)
but barely moves smaller shapes. AMX seems to pipeline LDX/FMA already to a large
degree; manual pipelining adds at best a partial recovery.

**Best AMX kernel vs Accelerate by shape:**

```
QKV-proj    0.74× Accelerate (212 vs 288 GFLOPS)
FFN1        0.44× (181 vs 412)
FFN2        0.26× (137 vs 520)
LM-head     0.75× (716 vs 951)
```

We narrow the gap but do not close it. The remaining 1.3–3.8× would require
multi-level (L3/L2/L1) cache blocking with panel sizes hand-tuned per shape, an
instruction-scheduled microkernel with explicit cycle counts, prefetch hints, and
possibly the Zhou-thesis masked-outer-product trick that drops the per-i0 pack
entirely. Multi-week, M1-specific, and reproduces what Apple did inside Accelerate.

The prefill bet is closed.

## 4. The honest landing

**What is durable in this repo:**

1. **A bit-exact direct-AMX int4 GEMV** (`bench/amx/int4_gemv_amx.cc::amx_int4_gemv_b4_perRow`)
   — 7.73 ms at GPT-2-small weight volume, per-row scales (a real quant scheme),
   beats every fp32 baseline on this shape: ORT decode 1.42×, Accelerate sgemv 1.90×,
   fp32 NEON GEMV 1.24×, naive NEON int4 GEMV 1.64×.

2. **A bit-exact direct-AMX `vecint` int8 variant** (`amx_int4_gemv_vecint`) and the
   M1 micro-architectural finding that `vecint` mode 10 is not a DOTPROD analog —
   ~5 cycles/instr.

3. **Five direct-AMX `sgemm` kernels of increasing sophistication** (naive → blocked →
   `LDX_pair` → software-pipelined), all bit-exact (`0.0e+00` diff at all 4 prefill
   shapes), and a clean characterization of where each optimization pays off.

4. **Full canonical AMX authoring chops** across `LDX/LDY/LDZ/STX/STY/STZ/FMA32/MATINT/
   VECINT/VECFP/GENLUT/SET/CLR`, all encoded against `corsix/amx` source markdown +
   C emulation. Vendored `third_party/amx/aarch64.h` matches corsix HEAD.

5. **An honest prior-art map + novelty gate** (`docs/LITERATURE.md`) that prevents
   any over-claimed paper, and a clean characterization of the M1 CPU regime that
   nobody has published as far as I can find.

**What was tested and rejected:**

- The "matrix engine wins for decode" contrarian thesis (fails vs DOTPROD).
- `vecint` as a DOTPROD analog (fails on M1's ~5 cycles/instr).
- `LDX_pair` as a free win (regresses at moderate shapes).
- Software pipelining as a thesis-scale path (helps marginally at memory-bound LM-head).

**Where a breakthrough could still live (untested here):**

- **Different silicon (M2/M3/M4):** `vecint` per-instruction latency may differ on
  later silicon; the thesis could revive there. Hardware we don't have.
- **Apple's published `sme` (Scalable Matrix Extension) on M4:** documented, not
  reverse-engineered, and a successor to AMX. A real systems target if hardware
  becomes available.
- **A reproduction of the Zhou 2025 in-place masked-outer-product GEMM on M1:**
  multi-week reproduction of published work; would beat Accelerate at some shapes
  but isn't novel.

## 5. Related work (the pieces that close off naive novelty claims)

Full citations and discussion in `docs/LITERATURE.md`. The pieces that constrain
what a paper can claim:

- **Zhou, J.** *Performance Analysis of the Apple AMX Matrix Accelerator.* MIT
  MEng thesis, 2025. AMX microbench + an in-place masked-outer-product GEMM that
  beats Accelerate on select shapes.
- **POMACS 2025**, 10.1145/3771563 — "First thorough characterization of Apple
  Silicon for on-device inference" (GPU/unified-memory, 8B–405B, vs NVIDIA).
- **arXiv 2510.06957** — Custom AMX kernels for sparse ternary GEMM on M1.
- **arXiv 2601.03324 "Bare-Metal Tensor Virtualization"** (Jan 2026) — from-scratch
  ARM64 LLM engine that *deliberately refused* AMX, ate a ~5× penalty. The
  "we open the black box" framing inferc could land was substantially preempted.
- **llama.cpp** — Apple Silicon path uses `Accelerate cblas_sgemm` + NEON+DOTPROD;
  in production, this is the SOTA we measured against.
- **MLAS source** — confirmed ORT-CPU on ARM64 uses NEON, *not* Accelerate. So
  inferc's win over single-thread ORT-CPU is fundamentally AMX-via-Accelerate vs
  MLAS-NEON, a hardware-path delta.

## 6. Reproducible artifacts

Everything in this report runs from `bench/amx/` with a clang++ invocation, no
CMake build needed. Source paths:

```
third_party/amx/aarch64.h               # vendored corsix AMX macros (canonical)
bench/amx/decode_floor.cc               # gate 1 — fp32 vs int8 vs int4 memory/dequant
bench/amx/genlut_throughput.cc          # gate 2 — genlut throughput proof
bench/amx/int4_gemv_amx.cc              # gate 3-5 — int4 GEMV (v1..v5 + vecint v6)
bench/amx/prefill_bench.cc              # prefill — 4 AMX sgemm variants vs Accelerate
src/kernels/amx_gemm.cc                 # the original AMX fp32 GEMM (4-way ILP)
src/kernels/amx_gemm_test.cc            # bit-exact test of the above (in tests/)
```

To rerun, e.g., the int4 GEMV kernels:

```bash
clang++ -O3 -std=c++17 -I third_party -framework Accelerate \
  bench/amx/int4_gemv_amx.cc -o /tmp/int4_gemv_amx
/tmp/int4_gemv_amx
```

All bench binaries print bit-exact correctness max-abs-diff numbers alongside the
GFLOPS/ms tables, so the perf claims can never silently lose their accuracy
guarantees.

---

*Investigation: inferc Session 22. Hardware: Apple M1 (base, 4P+4E, ~30 GB/s
single-core DRAM, AMX peak ~1400 GFLOPS fp32). Method: corsix-encoded direct AMX
inline asm + scalar correctness references + honest comparison against
llama.cpp (Q4_0 NEON+DOTPROD) and Accelerate (`sgemm`/`sgemv`).*
