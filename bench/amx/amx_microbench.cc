// Is the AMX inner loop LOAD-bound or FMA-bound? Decides whether the 32x32
// retiling (which cuts loads 3->2 per 4 FMAs) can help via instruction issue.
//
// Method: keep tiny A/B buffers L1-resident and reissue the same loads every
// iteration, so memory bandwidth/latency is NOT the limiter -- only whether a
// load instruction competes with an FMA for the AMX issue slot. Sweep the
// load:FMA ratio; all configs do the SAME 4 FMA32 / iteration (2048 FLOP).
//
//   0:4  pure FMA (peak issue)            <- ceiling
//   1:4  + 1 LDX_pair
//   2:4  + 1 LDX_pair + 1 LDY_pair        <- the 32x32 tile's load count
//   3:4  + 2 LDX_pair + 1 LDY             <- the current 16x64 tile's load count
//
// If 0:4 ~= 2:4 ~= 3:4  -> FMA-bound: loads overlap, 32x32 won't help (issue).
// If throughput rises as loads drop      -> load-bound: 32x32's 3->2 is the lever.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}
// full FMA32 operand with explicit X and Y byte offsets (Y offset = bits 0-9).
static inline uint64_t Fma(int z, int xoff, int yoff, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xoff) << 10) | (uint64_t(yoff)) | (f ? (1ULL << 27) : 0);
}
static const uint64_t PAIR = 1ULL << 62;

int main() {
  // L1-resident operands. 64-byte aligned. X holds 64 fp32 (4 regs), Y 32 fp32.
  float* A = nullptr; float* B = nullptr;
  if (posix_memalign((void**)&A, 128, 64 * sizeof(float)) ||
      posix_memalign((void**)&B, 128, 64 * sizeof(float))) { std::printf("alloc\n"); return 1; }
  for (int i = 0; i < 64; ++i) { A[i] = float(i) * 1e-3f; B[i] = float(i) * 1e-3f; }
  const uint64_t a = reinterpret_cast<uint64_t>(A), b = reinterpret_cast<uint64_t>(B);

  const int64_t ITERS = 30'000'000;          // each iteration = 4 FMA32 = 2048 FLOP
  const double FLOP = (double)ITERS * 4 * 512;
  const double GHZ = 3.2;                      // M1 P-core, for cycles/iter estimate

  auto report = [&](const char* tag, double t, int loads) {
    double g = FLOP / t / 1e9;
    double cyc_iter = t * GHZ * 1e9 / ITERS;   // cycles per 4-FMA iteration
    std::printf("%-26s %6.0f GFLOPS   %4.2f cyc/iter  (%d ld + 4 fma)\n", tag, g, cyc_iter, loads);
  };

  AMX_SET();
  // warmup
  for (int i = 0; i < 1000; ++i) { AMX_FMA32(Fma32Op(0,0,false)); }

  // 0:4 pure FMA
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_FMA32(Fma32Op(0,0,false)); AMX_FMA32(Fma32Op(1,64,false));
      AMX_FMA32(Fma32Op(2,128,false)); AMX_FMA32(Fma32Op(3,192,false));
    }
    report("0:4  pure FMA (ceiling)", sec(clk::now()-t0), 0);
  }
  // 1:4  + 1 LDX_pair
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL<<56) | PAIR);
      AMX_FMA32(Fma32Op(0,0,false)); AMX_FMA32(Fma32Op(1,64,false));
      AMX_FMA32(Fma32Op(2,128,false)); AMX_FMA32(Fma32Op(3,192,false));
    }
    report("1:4  +1 LDX_pair", sec(clk::now()-t0), 1);
  }
  // 1:4 DECOUPLED: load into X6/X7 which the FMAs (read X0..3) do NOT depend on.
  // Separates load-USE latency (data hazard, fixable by software pipelining) from
  // a structural switch/issue penalty (load and FMA can't co-issue, not fixable).
  //   ~= pure FMA  -> hazard-only: pipelining hides it -> big win available.
  //   ~= 1:4 above -> structural: pipelining won't recover it.
  {
    AMX_LDX(b | (0ULL<<56) | PAIR); AMX_LDX((b+128) | (2ULL<<56) | PAIR);  // preload X0..3
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (6ULL<<56) | PAIR);   // -> X6/X7, not read by the FMAs below
      AMX_FMA32(Fma32Op(0,0,false)); AMX_FMA32(Fma32Op(1,64,false));
      AMX_FMA32(Fma32Op(2,128,false)); AMX_FMA32(Fma32Op(3,192,false));
    }
    report("1:4  decoupled (no dep)", sec(clk::now()-t0), 1);
  }
  // 2:4  + 1 LDX_pair + 1 LDY_pair  (the 32x32 tile load count)
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL<<56) | PAIR);
      AMX_LDY(a | (0ULL<<56) | PAIR);
      AMX_FMA32(Fma32Op(0,0,false)); AMX_FMA32(Fma32Op(1,64,false));
      AMX_FMA32(Fma32Op(2,128,false)); AMX_FMA32(Fma32Op(3,192,false));
    }
    report("2:4  +LDX_pair+LDY_pair", sec(clk::now()-t0), 2);
  }
  // 3:4  + 2 LDX_pair + 1 LDY  (the current 16x64 tile load count)
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL<<56) | PAIR);
      AMX_LDX((b+128) | (2ULL<<56) | PAIR);   // +128 bytes = B[32], 128-aligned
      AMX_LDY(a | (0ULL<<56));
      AMX_FMA32(Fma32Op(0,0,false)); AMX_FMA32(Fma32Op(1,64,false));
      AMX_FMA32(Fma32Op(2,128,false)); AMX_FMA32(Fma32Op(3,192,false));
    }
    report("3:4  +2 LDX_pair+1 LDY", sec(clk::now()-t0), 3);
  }
  // 3:4 unpaired  (2 LDX + 1 LDY, no pairing) -- isolate the pairing effect
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL<<56));
      AMX_LDX((b+64) | (1ULL<<56));           // +64 bytes = B[16], 64-aligned
      AMX_LDY(a | (0ULL<<56));
      AMX_FMA32(Fma32Op(0,0,false)); AMX_FMA32(Fma32Op(1,64,false));
      AMX_FMA32(Fma32Op(2,128,false)); AMX_FMA32(Fma32Op(3,192,false));
    }
    report("3:4  unpaired (2LDX+1LDY)", sec(clk::now()-t0), 3);
  }
  // 32x32 ALTERNATING (G=1): per k, LDX_pair + LDY_pair + 4 FMA (the candidate
  // kernel's inner loop, switching load<->fma every k).
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(b | (0ULL<<56) | PAIR);   // X0,X1
      AMX_LDY(a | (0ULL<<56) | PAIR);   // Y0,Y1
      AMX_FMA32(Fma(0,  0,  0,false)); AMX_FMA32(Fma(1, 64,  0,false));
      AMX_FMA32(Fma(2,  0, 64,false)); AMX_FMA32(Fma(3, 64, 64,false));
    }
    report("32x32 alt  G=1", sec(clk::now()-t0), 2);
  }
  // 32x32 BATCHED (G=4): load 4 k-steps into X0..7/Y0..7, THEN 16 FMA. One
  // load<->fma switch per 4 k instead of per k. Same instr count as 4x the G=1
  // body; if batching recovers throughput toward the ceiling, switch cost is the
  // bottleneck and this is the microkernel design.
  {
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS / 4; ++i) {
      for (int kk = 0; kk < 4; ++kk) {           // load phase: 4 k-steps
        AMX_LDX(b | (uint64_t(2*kk) << 56) | PAIR);
        AMX_LDY(a | (uint64_t(2*kk) << 56) | PAIR);
      }
      for (int kk = 0; kk < 4; ++kk) {           // fma phase: 4 quadrants x 4 k
        int xb = (2*kk) * 64, yb = (2*kk) * 64;
        AMX_FMA32(Fma(0, xb,      yb,      false));
        AMX_FMA32(Fma(1, xb + 64, yb,      false));
        AMX_FMA32(Fma(2, xb,      yb + 64, false));
        AMX_FMA32(Fma(3, xb + 64, yb + 64, false));
      }
    }
    report("32x32 batch G=4", sec(clk::now()-t0), 2);
  }
  AMX_CLR();
  std::printf("\nload-bound if GFLOPS rises as loads drop; FMA-bound if 0:4~=2:4~=3:4.\n"
              "32x32 (2 loads) vs 16x64 (3 loads): compare the 2:4 and 3:4 rows.\n");
  free(A); free(B);
  return 0;
}
