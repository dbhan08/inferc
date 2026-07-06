// Does STZ hide in the FMA16 issue shadow the way LDX/LDY did? This is the single
// fact that decides whether a software-PIPELINED drain (overlap block A's STZ-out
// with block B's FMA16) could rescue the blocked-fp16 method after the un-pipelined
// drain-cost bench showed 13-71% efficiency. Mirror the load-issue microbench:
// measure cyc/FMA16 for pure FMA16 vs FMA16 + interleaved STZ.
//   cyc/FMA16 flat  -> STZ co-issues in the shadow: pipelined drain viable.
//   cyc/FMA16 rises -> STZ serializes with FMA16: the method is dead.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Fma16(int z, bool skipZ) {  // f16-acc, 4 independent tiles for ILP
  return (uint64_t(z) << 20) | (skipZ ? (1ULL << 27) : 0);
}

int main() {
  void* scratch = nullptr; posix_memalign(&scratch, 128, 64 * 64);
  const uint64_t zs = (uint64_t)scratch;
  const int64_t ITERS = 20'000'000; const double GHZ = 3.2; const double FMA16S = (double)ITERS * 4;
  auto rep = [&](const char* tag, double t, int stz) {
    std::printf("%-30s %6.2f cyc/FMA16   (%d STZ : 4 FMA16)\n", tag, t * GHZ * 1e9 / FMA16S, stz);
  };
  AMX_SET();
  for (int i = 0; i < 2000; ++i) AMX_FMA16(Fma16(0, true));
  // pure: 4 independent fp16 tiles (Z 0,8,16,24), skip-Z -> issue ceiling
  { auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_FMA16(Fma16(0,true)); AMX_FMA16(Fma16(8,true)); AMX_FMA16(Fma16(16,true)); AMX_FMA16(Fma16(24,true)); }
    rep("0 STZ  pure FMA16 (ceiling)", sec(clk::now()-t0), 0); }
  // + 1 STZ per 4 FMA16
  { auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_STZ(zs | (0ULL<<56));
      AMX_FMA16(Fma16(0,true)); AMX_FMA16(Fma16(8,true)); AMX_FMA16(Fma16(16,true)); AMX_FMA16(Fma16(24,true)); }
    rep("1 STZ : 4 FMA16", sec(clk::now()-t0), 1); }
  // + 2 STZ per 4 FMA16
  { auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_STZ(zs | (0ULL<<56)); AMX_STZ((zs+64) | (1ULL<<56));
      AMX_FMA16(Fma16(0,true)); AMX_FMA16(Fma16(8,true)); AMX_FMA16(Fma16(16,true)); AMX_FMA16(Fma16(24,true)); }
    rep("2 STZ : 4 FMA16", sec(clk::now()-t0), 2); }
  // + 4 STZ per 4 FMA16 (1:1 -- a tile drained over its own compute window)
  { auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_STZ(zs|(0ULL<<56)); AMX_STZ((zs+64)|(1ULL<<56)); AMX_STZ((zs+128)|(2ULL<<56)); AMX_STZ((zs+192)|(3ULL<<56));
      AMX_FMA16(Fma16(0,true)); AMX_FMA16(Fma16(8,true)); AMX_FMA16(Fma16(16,true)); AMX_FMA16(Fma16(24,true)); }
    rep("4 STZ : 4 FMA16", sec(clk::now()-t0), 4); }
  // reference: 1 LDX : 4 FMA16 (we KNOW loads hide -> baseline for "hides")
  { void* buf=nullptr; posix_memalign(&buf,128,64); const uint64_t bb=(uint64_t)buf;
    auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) {
      AMX_LDX(bb | (0ULL<<56));
      AMX_FMA16(Fma16(0,true)); AMX_FMA16(Fma16(8,true)); AMX_FMA16(Fma16(16,true)); AMX_FMA16(Fma16(24,true)); }
    rep("1 LDX : 4 FMA16 (hides ref)", sec(clk::now()-t0), 0); free(buf); }
  AMX_CLR();
  std::printf("\nIf STZ rows track the LDX 'hides' reference (flat cyc/FMA16) -> pipelined\n"
              "drain is viable, method lives. If cyc/FMA16 rises ~linearly with STZ count\n"
              "-> STZ serializes, no shadow, blocked-fp16 cannot beat f32-acc. Decisive.\n");
  free(scratch); return 0;
}
