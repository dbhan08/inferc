// Z-evacuation cost shootout -- the decisive "design around the STZ wall" probe.
// Blocked-fp16 (accurate 2x) died because STZ costs ~27 cyc to evacuate the fp16 Z.
// But STZ (store-to-memory) is one of several ways out of Z. EXTRH (op 8, bit26=1)
// moves a Z row into an X/Y register -- register-to-register, no memory. If EXTRH
// hides in the FMA16 issue shadow (like LDX does) instead of costing like STZ, an
// in-register fp32 fold becomes cheap and the accurate-fp16 2x kernel is revived.
// Measure cyc/FMA16 for: pure / +EXTRH / +STZ (expensive ref) / +LDX (free ref).
// Encoding: EXTRH = op8, (Zrow<<20)|(1<<26)|dest-off; bit10=0 -> dest X.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Fma16(int z, bool skipZ) { return (uint64_t(z) << 20) | (skipZ ? (1ULL << 27) : 0); }
// EXTRH: extract Z row `zrow` -> X register at byte `xoff` (dest X: bit10=0).
static inline uint64_t Extrh(int zrow, int xoff) {
  return (uint64_t(zrow) << 20) | (1ULL << 26) | (uint64_t(xoff) & 0x1FF);
}

int main() {
  void* mem = nullptr; posix_memalign(&mem, 128, 64 * 64);
  const uint64_t zs = (uint64_t)mem;
  const int64_t ITERS = 20'000'000; const double GHZ = 3.2; const double F = (double)ITERS * 4;
  auto rep = [&](const char* tag, double t) { std::printf("%-30s %6.2f cyc/FMA16\n", tag, t * GHZ * 1e9 / F); };
  AMX_SET();
  for (int i = 0; i < 2000; ++i) AMX_FMA16(Fma16(0, true));
#define BODY AMX_FMA16(Fma16(0,true)); AMX_FMA16(Fma16(8,true)); AMX_FMA16(Fma16(16,true)); AMX_FMA16(Fma16(24,true))
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ BODY; } rep("0 extra  pure FMA16", sec(clk::now()-t0)); }
  // +1 EXTRH per 4 FMA16 (Z row 0 -> X4 @ byte 256)
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_EXTRX(Extrh(0,256)); BODY; } rep("1 EXTRH : 4 FMA16", sec(clk::now()-t0)); }
  // +2 EXTRH
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_EXTRX(Extrh(0,256)); AMX_EXTRX(Extrh(2,320)); BODY; } rep("2 EXTRH : 4 FMA16", sec(clk::now()-t0)); }
  // +4 EXTRH
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_EXTRX(Extrh(0,256)); AMX_EXTRX(Extrh(2,320)); AMX_EXTRX(Extrh(4,384)); AMX_EXTRX(Extrh(6,448)); BODY; } rep("4 EXTRH : 4 FMA16", sec(clk::now()-t0)); }
  // other Z/X evacuation + store paths (does ANY cheap store-to-memory exist?)
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_STX(zs|(0ULL<<56)); BODY; } rep("1 STX : 4 FMA16", sec(clk::now()-t0)); }
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_STY(zs|(0ULL<<56)); BODY; } rep("1 STY : 4 FMA16", sec(clk::now()-t0)); }
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_STZI(zs|(0ULL<<56)); BODY; } rep("1 STZI : 4 FMA16", sec(clk::now()-t0)); }
  // references
  { auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_STZ(zs|(0ULL<<56)); BODY; } rep("1 STZ : 4 FMA16 (slow ref)", sec(clk::now()-t0)); }
  { void* b=nullptr; posix_memalign(&b,128,64); const uint64_t bb=(uint64_t)b;
    auto t0=clk::now(); for(int64_t i=0;i<ITERS;++i){ AMX_LDX(bb|(0ULL<<56)); BODY; } rep("1 LDX : 4 FMA16 (free ref)", sec(clk::now()-t0)); free(b); }
  AMX_CLR();
  std::printf("\nEXTRH ~ LDX (flat) => cheap in-register Z-evacuation EXISTS: design the\n"
              "fp32-fold around it, accurate-fp16 2x kernel revived. EXTRH ~ STZ (~27 cyc)\n"
              "=> all Z-evacuation is expensive, blocked-fp16 is truly dead.\n");
  free(mem); return 0;
}
