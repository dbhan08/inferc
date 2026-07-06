// The LDZ profile showed the correct in-reg exp is LATENCY-bound on the VECFP->EXTRH
// read-after-write chain (EXTRH jumped 1.8->17 cyc when it reads a just-written Z),
// NOT op-count bound (LDZ is cheap). Latency-bound => more ILP hides it. The attention
// bench used 16 independent rows; is that enough to hide the ~15-cyc chain latency, or
// does WIDER ILP keep dropping cyc/exp? Sweep W = independent rows processed level-major.
// If cyc/exp falls as W grows past 16, the 0.72x was ILP-limited and recoverable.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t Vfp(int z,int xo,int yo){return (4ULL<<42)|(uint64_t(z)<<20)|(uint64_t(xo)<<10)|uint64_t(yo);}
static inline uint64_t ExX(int z,int xo){return (uint64_t(z)<<20)|(1ULL<<26)|(uint64_t(xo)&0x1FF);}
static inline uint64_t ExY(int z,int yo){return (uint64_t(z)<<20)|(1ULL<<26)|(1ULL<<10)|(uint64_t(yo)&0x1FF);}

int main(){
  void* zb=nullptr; posix_memalign(&zb,128,64); const uint64_t z0=(uint64_t)zb; for(int i=0;i<16;++i)((float*)zb)[i]=0;
  const int M=8; const double GHZ=3.2;
  std::printf("m=%d squaring exp, level-major ILP sweep. Z has 64 rows.\n", M);
  std::printf("%-8s %-14s %s\n","W rows","cyc/exp-row","note");
  for(int W : {4,8,16,32,48,56}){
    int REP = 12'000'000 / W;                    // ~constant total work
    AMX_SET(); for(int i=0;i<2000;++i) AMX_VECFP(Vfp(0,0,0));
    auto t0=clk::now();
    for(int r=0;r<REP;++r){
      for(int qsq=0;qsq<M;++qsq)                  // each level across all W rows = W-way ILP
        for(int j=0;j<W;++j){ int xb=(j%8)*64;
          AMX_EXTRX(ExX(j,xb)); AMX_EXTRX(ExY(j,xb)); AMX_LDZ(z0|((uint64_t)j<<56)); AMX_VECFP(Vfp(j,xb,xb)); }
    }
    double t=sec(clk::now()-t0); AMX_CLR();
    double cyc_per_exp = t*GHZ*1e9/((double)REP*W);   // cycles to do one row's full m-squaring exp
    std::printf("%-8d %-14.1f %s\n", W, cyc_per_exp, W>=48?"(near Z capacity)":"");
  }
  std::printf("\nIf cyc/exp-row keeps dropping past W=16 -> the attention exp is ILP-starved\n"
              "at 16 rows; process more independent rows (multi-keyblock) to recover. If flat\n"
              "by 16 -> 16-way already saturates, the 4-op square is the real floor (0.72x stands).\n");
  free(zb); return 0;
}
