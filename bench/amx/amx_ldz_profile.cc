// PROFILE (not assume): the correct in-reg square added an LDZ (zero Z) per op that
// I never cost-measured. Is LDZ cheap (a load, ~1 cyc) or expensive (Z-side, ~31 cyc
// like the stores)? And within the 4-op square (EXTRH-X, EXTRH-Y, LDZ, VECFP), which
// op actually dominates? Measure each op's marginal cyc, so the 0.72% conclusion is
// backed by a named cycle cost -- or reveals a cheaper zeroing path.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t Vfp(int alu,int z,int xo,int yo){return (uint64_t(alu)<<47)|(4ULL<<42)|(uint64_t(z)<<20)|(uint64_t(xo)<<10)|uint64_t(yo);}
static inline uint64_t ExX(int z,int xo){return (uint64_t(z)<<20)|(1ULL<<26)|(uint64_t(xo)&0x1FF);}
static inline uint64_t ExY(int z,int yo){return (uint64_t(z)<<20)|(1ULL<<26)|(1ULL<<10)|(uint64_t(yo)&0x1FF);}

int main(){
  void* m=nullptr; posix_memalign(&m,128,64*64); const uint64_t a=(uint64_t)m;
  const int64_t IT=20'000'000; const double GHZ=3.2; const double N=IT*4;
  auto rep=[&](const char* t,double s){std::printf("%-30s %5.2f cyc/op\n",t,s*GHZ*1e9/N);};
  AMX_SET();
  for(int i=0;i<2000;++i) AMX_VECFP(Vfp(0,0,0,0));
#define B AMX_VECFP(Vfp(0,0,0,0)); AMX_VECFP(Vfp(0,1,0,0)); AMX_VECFP(Vfp(0,2,0,0)); AMX_VECFP(Vfp(0,3,0,0))
  // marginal cost of each op vs a pure-VECFP baseline (4 VECFP/iter)
  {auto t0=clk::now();for(int64_t i=0;i<IT;++i){B;}rep("baseline: 4 VECFP",sec(clk::now()-t0));}
  {auto t0=clk::now();for(int64_t i=0;i<IT;++i){AMX_LDZ(a|(0ULL<<56));B;}rep("+1 LDZ : 4 VECFP",sec(clk::now()-t0));}
  {auto t0=clk::now();for(int64_t i=0;i<IT;++i){AMX_LDX(a|(0ULL<<56));B;}rep("+1 LDX : 4 VECFP (ref)",sec(clk::now()-t0));}
  {auto t0=clk::now();for(int64_t i=0;i<IT;++i){AMX_EXTRX(ExX(0,0));B;}rep("+1 EXTRH-X : 4 VECFP",sec(clk::now()-t0));}
  {auto t0=clk::now();for(int64_t i=0;i<IT;++i){AMX_EXTRX(ExY(0,0));B;}rep("+1 EXTRH-Y : 4 VECFP",sec(clk::now()-t0));}
  // the FULL correct 4-op square unit, repeated 4x/iter (matches the real exp inner)
  {auto t0=clk::now();for(int64_t i=0;i<IT;++i){
     AMX_EXTRX(ExX(0,0));AMX_EXTRX(ExY(0,0));AMX_LDZ(a|(0ULL<<56));AMX_VECFP(Vfp(0,0,0,0));
     AMX_EXTRX(ExX(1,0));AMX_EXTRX(ExY(1,0));AMX_LDZ(a|(1ULL<<56));AMX_VECFP(Vfp(0,1,0,0));
     AMX_EXTRX(ExX(2,0));AMX_EXTRX(ExY(2,0));AMX_LDZ(a|(2ULL<<56));AMX_VECFP(Vfp(0,2,0,0));
     AMX_EXTRX(ExX(3,0));AMX_EXTRX(ExY(3,0));AMX_LDZ(a|(3ULL<<56));AMX_VECFP(Vfp(0,3,0,0));
   }rep("FULL 4-op square (per square)",sec(clk::now()-t0));}
  AMX_CLR();
  std::printf("\nIf LDZ ~1 cyc -> exp's 4 ops are irreducible, 0.72x stands (mechanism = op count).\n"
              "If LDZ ~31 cyc -> the ZEROING is the killer; find a cheaper clear (VECFP force-0,\n"
              "double-buffer rows, etc.) -> the conclusion was premature.\n");
  free(m); return 0;
}
