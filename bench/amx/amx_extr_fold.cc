// THE VERDICT BUILD: EXTRH + VECFP-mode3 in-register fp32 fold for blocked-fp16.
// EXTRH (Z->X) is ~1 cyc (cheap); ALL stores are ~31 cyc; so the fp32 accumulator
// must live in Z and be folded in-register. VECFP mode 3 widens fp16->fp32 writing
// 2 Z rows per fp16 row, so the fp32 accumulator is 2x the fp16 tile's row count --
// they can't both be 32x32 in the 64-row Z file. The fitting layout is a 32x16
// output tile (fp16 tile = 16 rows, fp32 accumulator = 32 rows, total 48 <= 64),
// which HALVES the per-FMA16 compute (512 MAC vs 1024). Question: does the cheap
// EXTRH fold still net out above the fp32-acc path (~1,500), or does the half-width
// penalty erase it? Measure the steady-state inner-loop throughput, sweep b.
//
// Encodings: FMA16 f16-acc (bit62=0); EXTRH op8 (Zrow<<20)|(1<<26)|xoff, dest X;
// VECFP mode3 (3<<42)|(zrow<<20)|(xoff<<10)|yoff -- X,Y read f16, +-> f32 Z 2 rows.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Fma16(int z, bool skipZ) { return (uint64_t(z) << 20) | (skipZ ? (1ULL << 27) : 0); }
static inline uint64_t Extrh(int zrow, int xoff) { return (uint64_t(zrow) << 20) | (1ULL << 26) | (uint64_t(xoff) & 0x1FF); }
static inline uint64_t Vecfp3(int zrow, int xoff, int yoff) {  // f16->f32 widening accumulate, 2 rows
  return (3ULL << 42) | (uint64_t(zrow) << 20) | (uint64_t(xoff) << 10) | uint64_t(yoff);
}

int main() {
  __fp16* ones = nullptr; posix_memalign((void**)&ones, 128, 64);
  for (int i = 0; i < 32; ++i) ones[i] = (__fp16)1.0f;
  const uint64_t y1 = (uint64_t)ones;

  const int K = 2048, REP = 80000; const double GHZ = 3.2;
  const int Bs[] = {16, 32, 64, 128};
  AMX_SET();
  AMX_LDY(y1 | (7ULL << 56));                 // Y[7] = fp16 ones (for the fold multiply)
  for (int i = 0; i < 2000; ++i) AMX_FMA16(Fma16(0, true));

  std::printf("32x16-tile EXTRH+VECFP3 fold (512 MAC/FMA16).  fp32-acc anchor ~1,500 GFLOPS\n");
  std::printf("%-6s %-16s %-s\n", "b", "GFLOPS", "accuracy 1.45e-4*sqrt(b)");
  for (int b : Bs) {
    auto t0 = clk::now();
    for (int rep = 0; rep < REP; ++rep)
      for (int k0 = 0; k0 < K; k0 += b) {
        // ILP FIX: 4 INDEPENDENT fp16 accumulation tiles (z=0,8,16,24) interleaved ->
        // hides the FMA16 4-cyc latency the single-chain version was bound on.
        for (int k = 0; k < b; ++k) {
          AMX_FMA16(Fma16(0,  k == 0)); AMX_FMA16(Fma16(8,  k == 0));
          AMX_FMA16(Fma16(16, k == 0)); AMX_FMA16(Fma16(24, k == 0));
        }
        // DRAIN interleaved: 8 EXTRH then 8 VECFP3 (break the EXTRH->VECFP dependency)
        for (int r = 0; r < 8; ++r) AMX_EXTRX(Extrh(r, (r % 8) * 64));
        for (int r = 0; r < 8; ++r) AMX_VECFP(Vecfp3(32 + 2 * r, (r % 8) * 64, 7 * 64));
      }
    double t = sec(clk::now() - t0);
    double flop = (double)REP * K * 4 * 1024 * 2;  // 4 indep FMA16 x 1024 MAC/k
    std::printf("%-6d %-16.0f %.1e\n", b, flop / t / 1e9, 1.45e-4 * std::sqrt((double)b));
  }
  AMX_CLR();
  std::printf("\nVERDICT: if best GFLOPS > ~1,500 -> EXTRH fold beats fp32-acc despite\n"
              "half-width: BREAKTHROUGH. If <= ~1,500 -> the fp32-accumulator's 2x Z\n"
              "footprint forces half-width tiles that erase the cheap-extract win.\n");
  free(ones); return 0;
}
