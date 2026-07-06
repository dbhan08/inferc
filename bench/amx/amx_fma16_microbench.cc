// FMA16 load-issue microbench -- the Paper-2 gate experiment.
//
// Paper 1 established the M1 AMX fp32 inner loop is LOAD-ISSUE BOUND: the instant
// any operand load interleaves with the FMA32 stream, single-thread throughput
// pins to ~670 GFLOPS (~9.8 cyc per 4 FMA32 = per 2048 FLOP) and nothing moves it
// (amx_microbench.cc). The only fp32 levers are above the loop (pre-pack, 2nd block).
//
// Paper 2 thesis: fp16 is the one knob that moves the inner loop. One FMA16 outer
// product is fed by just 1 LDX + 1 LDY (a 64-byte fp16 column each), vs the fp32
// tile's 3 loads per equivalent compute -- so fp16 needs FEWER loads per MAC and
// MIGHT slip under the load-issue wall, even though FMA16 itself issues at only
// ~1 per 2 cycles (Zhou thesis; meekolab teardown) versus FMA32's 1 per cycle.
//
// THE GATE: does fp16 escape the load-issue bound, or share it?
//   * cyc/FMA16 ~flat as loads are added  -> loads hide under the 2-cyc FMA16
//     issue: fp16 ESCAPES the wall. Build the kernel.
//   * cyc/FMA16 RISES with each load      -> loads serialize with FMA16 just as
//     in fp32: same wall, no fp16 headroom. Fold Paper 2 into a Paper-1 appendix.
//
// Honest caveat: the per-instruction MAC count of matrix-mode FMA16 is contested
// (corsix "8 Z registers" vs Zhou/meekolab "32 registers"). So the ENGINE-TRUTH
// metric here is cyc/FMA16 and how it scales with loads -- that alone answers the
// gate. The GFLOPS column assumes 32x32=1024 MAC/FMA16 (the Zhou/meekolab read)
// for intuition only; real-shape fp16 GFLOPS is measured separately by BNNS in
// amx_fp16_ceiling.cc. Encoding from corsix/amx fma.md: Z row 20-25, X off 10-18,
// Y off 0-8, skip-Z bit 27, matrix mode = bit 63 clear, bit 62 = f32 accumulate.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }

// FMA16 operand. zrow: Z row (bits 20-25). xoff/yoff: byte offsets. skipZ: bit 27
// (1 = Z=X*Y, no accumulate read -> decouples the FMA latency chain, so we measure
// pure ISSUE throughput, the analog of the fp32 bench's 4 independent Z banks).
// f32acc: bit 62 (1 = fp16-in/fp32-accumulate, the widening mode the real kernel
// needs; it writes 2x the Z bytes and may issue slower -- tested explicitly below).
static inline uint64_t Fma16(int zrow, int xoff, int yoff, bool skipZ, bool f32acc) {
  return (uint64_t(zrow) << 20) | (uint64_t(xoff) << 10) | uint64_t(yoff) |
         (skipZ ? (1ULL << 27) : 0) | (f32acc ? (1ULL << 62) : 0);
}
// One FMA16 group = 4 independent outer products to Z rows 0,8,16,24. Each matrix
// FMA16 spans up to 8 Z registers (corsix), so spacing by 8 keeps them disjoint;
// with skip-Z this gives ILP without an accumulate dependency, isolating issue.
static inline void Fma16x4(int xoff, int yoff, bool f32acc) {
  AMX_FMA16(Fma16(0,  xoff, yoff, true, f32acc));
  AMX_FMA16(Fma16(8,  xoff, yoff, true, f32acc));
  AMX_FMA16(Fma16(16, xoff, yoff, true, f32acc));
  AMX_FMA16(Fma16(24, xoff, yoff, true, f32acc));
}

int main() {
  // L1-resident fp16 operands. X/Y each hold 32 fp16 = 64 bytes = one register.
  // 8 registers' worth (512B) so loads can target X0..7 / Y0..7 without aliasing.
  __fp16* A = nullptr; __fp16* B = nullptr;
  if (posix_memalign((void**)&A, 128, 256 * sizeof(__fp16)) ||
      posix_memalign((void**)&B, 128, 256 * sizeof(__fp16))) { std::printf("alloc\n"); return 1; }
  for (int i = 0; i < 256; ++i) { A[i] = (__fp16)(i * 1e-3f); B[i] = (__fp16)(i * 1e-3f); }
  const uint64_t a = reinterpret_cast<uint64_t>(A), b = reinterpret_cast<uint64_t>(B);

  const int64_t ITERS = 20'000'000;            // each iteration = 4 FMA16
  const double FMA16S = (double)ITERS * 4;
  const double FLOP = FMA16S * 1024 * 2;        // ASSUMES 1024 MAC/FMA16 (see header)
  const double GHZ = 3.2;                       // M1 P-core

  auto report = [&](const char* tag, double t, int loads_per_fma) {
    double g = FLOP / t / 1e9;
    double cyc_fma = t * GHZ * 1e9 / FMA16S;    // cycles per FMA16 -- ENGINE TRUTH
    std::printf("%-28s %6.0f GFLOPS   %5.2f cyc/FMA16  (%d ld : 1 fma16)\n",
                tag, g, cyc_fma, loads_per_fma);
  };

  AMX_SET();
  for (int i = 0; i < 1000; ++i) Fma16x4(0, 0, true);   // warmup

  // ---- f32-accumulate (bit62=1): the mode the real fp16-in/fp32-acc kernel uses ----
  std::printf("== fp16-in / fp32-accumulate (the kernel's mode) ==\n");

  // 0 loads: pure FMA16 issue ceiling.
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) Fma16x4(0, 0, true);
    report("0:1  pure FMA16 (ceiling)", sec(clk::now() - t0), 0);
  }
  // 1 load : 1 FMA16 -- one LDX per FMA16 (64B fp16 column of B).
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL << 56)); AMX_LDX((b + 64) | (1ULL << 56));
      AMX_LDX((b + 128) | (2ULL << 56)); AMX_LDX((b + 192) | (3ULL << 56));
      Fma16x4(0, 0, true);
    }
    report("1:1  +1 LDX / fma16", sec(clk::now() - t0), 1);
  }
  // 2 loads : 1 FMA16 -- the deployed 32x32 fp16 tile (1 LDX + 1 LDY per k).
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL << 56)); AMX_LDX((b + 64) | (1ULL << 56));
      AMX_LDX((b + 128) | (2ULL << 56)); AMX_LDX((b + 192) | (3ULL << 56));
      AMX_LDY(a | (0ULL << 56)); AMX_LDY((a + 64) | (1ULL << 56));
      AMX_LDY((a + 128) | (2ULL << 56)); AMX_LDY((a + 192) | (3ULL << 56));
      Fma16x4(0, 0, true);
    }
    report("2:1  +LDX+LDY (32x32 tile)", sec(clk::now() - t0), 2);
  }
  // 2 loads DECOUPLED: load into X4..7 / Y4..7, which the FMA16s (read X0/Y0) do
  // not depend on. ~ceiling => any slowdown above is a fixable data hazard (pipeline
  // it). ~the 2:1 row => structural load/FMA issue contention, not pipelineable.
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX((b + 0) | (4ULL << 56)); AMX_LDX((b + 64) | (5ULL << 56));
      AMX_LDX((b + 128) | (6ULL << 56)); AMX_LDX((b + 192) | (7ULL << 56));
      AMX_LDY((a + 0) | (4ULL << 56)); AMX_LDY((a + 64) | (5ULL << 56));
      AMX_LDY((a + 128) | (6ULL << 56)); AMX_LDY((a + 192) | (7ULL << 56));
      Fma16x4(0, 0, true);   // reads X0/Y0, independent of the loads above
    }
    report("2:1  decoupled (no dep)", sec(clk::now() - t0), 2);
  }
  // BATCH G=4: 8 loads then 4 FMA16 (one load<->fma switch per 4 k instead of per
  // k). If batching beats the 2:1 row, the switch cost dominates and the kernel
  // should load-batch; if equal, contention is per-instruction.
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      for (int r = 0; r < 4; ++r) { AMX_LDX((b + 64 * r) | (uint64_t(r) << 56));
                                    AMX_LDY((a + 64 * r) | (uint64_t(r) << 56)); }
      for (int r = 0; r < 4; ++r) AMX_FMA16(Fma16(8 * r, 0, 0, true, true));
    }
    report("2:1  batch G=4", sec(clk::now() - t0), 2);
  }

  // ---- f16-accumulate (bit62=0): does the narrower accumulate issue faster? ----
  std::printf("== fp16-in / fp16-accumulate (ceiling reference) ==\n");
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) Fma16x4(0, 0, false);
    report("0:1  pure FMA16 f16-acc", sec(clk::now() - t0), 0);
  }
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL << 56)); AMX_LDX((b + 64) | (1ULL << 56));
      AMX_LDX((b + 128) | (2ULL << 56)); AMX_LDX((b + 192) | (3ULL << 56));
      AMX_LDY(a | (0ULL << 56)); AMX_LDY((a + 64) | (1ULL << 56));
      AMX_LDY((a + 128) | (2ULL << 56)); AMX_LDY((a + 192) | (3ULL << 56));
      Fma16x4(0, 0, false);
    }
    report("2:1  +LDX+LDY f16-acc", sec(clk::now() - t0), 2);
  }

  AMX_CLR();
  std::printf(
      "\nGATE: cyc/FMA16 ~flat across 0:1 / 1:1 / 2:1  -> loads hide under the\n"
      "2-cyc FMA16 issue: fp16 ESCAPES the load-issue wall, build the kernel.\n"
      "cyc/FMA16 RISES per load -> same wall as fp32, fold into Paper-1 appendix.\n"
      "Compare against fp32 (amx_microbench.cc): ~670 GFLOPS loaded, ~9.8 cyc per\n"
      "2048 FLOP. fp16 wins iff it does the same FLOP in materially fewer cycles.\n");
  free(A); free(B);
  return 0;
}
