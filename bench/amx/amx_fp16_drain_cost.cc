// STEP 2: drain-cost microbench -- THE decisive experiment for the novel method.
// Blocked-fp16 accumulation needs, every b k-steps, a "drain": 32 STZ of the fp16
// Z tile (step-1 layout: even rows 0..62) + a NEON fp16->fp32 convert-and-add into
// a fp32 spill. The whole novelty claim ("error correction is FREE in the FMA16
// issue shadow") rides on whether that drain hides. For each block size b we time
// the SAME inner loop WITH and WITHOUT the drain; the ratio = drain efficiency.
//
//   with/without ~= 1.0  -> drain hides in the issue shadow: novel 2x kernel ALIVE.
//   with/without << 1.0  -> drain serializes: falls back to the ~1.4x f32-acc kernel.
//
// Anchors: pure f16-acc ceiling ~3,027 GFLOPS; f32-acc ~1,500. A drained kernel
// that stays well above ~1,500 at an accurate b (b<=32 -> <=8.8e-4) beats the safe
// kernel AND is more accurate than BNNS Graph (1.4e-3).

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Fma16(int zrow, int xoff, int yoff, bool skipZ) {  // f16-acc (bit62=0)
  return (uint64_t(zrow) << 20) | (uint64_t(xoff) << 10) | uint64_t(yoff) | (skipZ ? (1ULL << 27) : 0);
}

int main() {
  __fp16* A = nullptr; __fp16* B = nullptr; __fp16* scratch = nullptr; float* spill = nullptr;
  posix_memalign((void**)&A, 128, 64); posix_memalign((void**)&B, 128, 64);
  posix_memalign((void**)&scratch, 128, 32 * 64);   // 32 even Z rows x 64B = 1024 fp16
  posix_memalign((void**)&spill, 128, 1024 * sizeof(float));
  for (int i = 0; i < 32; ++i) { A[i] = (__fp16)(i * 1e-3f); B[i] = (__fp16)(i * 1e-3f); }
  std::memset(spill, 0, 1024 * sizeof(float));
  const uint64_t a = (uint64_t)A, b_ = (uint64_t)B, zs = (uint64_t)scratch;

  const int K = 2048;            // one tile-pass depth
  const int REP = 60000;         // repeats for stable timing
  const double GHZ = 3.2;
  const int Bs[] = {8, 16, 32, 64, 128};

  auto drain = [&]() {           // 32 STZ (even rows) + NEON fp16->fp32 accumulate
    for (int r = 0; r < 32; ++r) AMX_STZ((zs + (uint64_t)r * 64) | ((uint64_t)(2 * r) << 56));
    for (int i = 0; i < 1024; i += 8) {
      float16x8_t h = vld1q_f16(scratch + i);
      float32x4_t lo = vaddq_f32(vld1q_f32(spill + i),     vcvt_f32_f16(vget_low_f16(h)));
      float32x4_t hi = vaddq_f32(vld1q_f32(spill + i + 4), vcvt_high_f32_f16(h));
      vst1q_f32(spill + i, lo); vst1q_f32(spill + i + 4, hi);
    }
  };

  AMX_SET();
  for (int i = 0; i < 2000; ++i) AMX_FMA16(Fma16(0, 0, 0, true));   // warm

  std::printf("%-6s %-14s %-14s %-12s %-s\n", "b", "drained GFLOPS", "no-drain GFLOPS",
              "efficiency", "accuracy 1.45e-4*sqrt(b)");
  for (int b : Bs) {
    // -------- WITH drain --------
    auto t0 = clk::now();
    for (int rep = 0; rep < REP; ++rep)
      for (int k0 = 0; k0 < K; k0 += b) {
        for (int k = 0; k < b; ++k) {
          AMX_LDX(b_ | (0ULL << 56)); AMX_LDY(a | (0ULL << 56));
          AMX_FMA16(Fma16(0, 0, 0, k == 0));      // accumulate into even-row tile
        }
        drain();
      }
    double tw = sec(clk::now() - t0);
    // -------- WITHOUT drain (same FMA16/load count) --------
    auto t1 = clk::now();
    for (int rep = 0; rep < REP; ++rep)
      for (int k0 = 0; k0 < K; k0 += b) {
        for (int k = 0; k < b; ++k) {
          AMX_LDX(b_ | (0ULL << 56)); AMX_LDY(a | (0ULL << 56));
          AMX_FMA16(Fma16(0, 0, 0, k == 0));
        }
      }
    double tn = sec(clk::now() - t1);

    double flop = (double)REP * K * 1024 * 2;      // K FMA16 x 1024 MAC x 2
    double gw = flop / tw / 1e9, gn = flop / tn / 1e9;
    std::printf("%-6d %-14.0f %-14.0f %-12.0f%% %.1e\n", b, gw, gn, 100.0 * gw / gn,
                1.45e-4 * std::sqrt((double)b));
  }
  AMX_CLR();
  std::printf("\nefficiency ~100%% => the 32-STZ drain hides in the FMA16 issue shadow\n"
              "(novel 'free correction' result holds). Pick largest b at efficiency~100%%\n"
              "AND accuracy below ~1e-3: that's the operating point. Compare drained\n"
              "GFLOPS to f32-acc ~1,500 (the fallback) and f16-acc ~3,027 (the ceiling).\n");
  free(A); free(B); free(scratch); free(spill);
  return 0;
}
