// SUPERSEDED / NEGATIVE RESULT: FMA does NOT support indexed-load (corsix fma.c has no
// indexed path). The "free" timing here was a plain FMA with the indexed bits ignored;
// amx_indexed_verify.cc confirmed it does NOT gather. The real fused gather is on
// MATFP/MATINT -> see amx_matfp_indexed.cc. Kept to document the RE correction.
//
// NEW-ANOMALY HUNT: indexed-load FMA. bit53 makes the FMA gather its X operand
// through 4-bit indices into a table register (bits49-51) -- a per-lane gather of a
// 16-entry codebook FUSED into the multiply-accumulate. If an indexed FMA issues at
// ~the same rate as a plain FMA, then int4-CODEBOOK GEMM costs the SAME as fp32 GEMM
// with the dequant FREE -- which would flip the int4-prefill loss (0.73x via genlut,
// which added ops) into a win, and relieve the load-issue bound (operand gathered,
// not loaded). Measure cyc/op: indexed vs plain. enc: base FMA | bit53 | bit48(4b) |
// bit47(0=index X) | tblreg<<49.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t Fma32(int z,int xo,int yo){return (uint64_t(z)<<20)|(uint64_t(xo)<<10)|uint64_t(yo);}
static inline uint64_t IdxFma32(int z,int xo,int yo,int tbl){           // indexed-X 4-bit gather FMA
  return Fma32(z,xo,yo)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)tbl<<49);
}

int main(){
  float* buf=nullptr; posix_memalign((void**)&buf,128,512);
  for(int i=0;i<128;++i) buf[i]=(float)(i%16)*0.1f;     // table + indices source
  const uint64_t b=(uint64_t)buf;
  const int64_t IT=20'000'000; const double GHZ=3.2; const double N=IT*4;
  auto rep=[&](const char* t,double s){std::printf("%-28s %5.2f cyc/op\n",t,s*GHZ*1e9/N);};

  AMX_SET();
  AMX_LDX(b|(0ULL<<56)); AMX_LDX((b+64)|(1ULL<<56));      // X0=indices src, X1=table
  AMX_LDY(b|(0ULL<<56));
  for(int i=0;i<2000;++i) AMX_FMA32(Fma32(0,0,0));

  // baseline: 4 plain FMA32 (independent Z banks)
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){
      AMX_FMA32(Fma32(0,0,0));AMX_FMA32(Fma32(1,64,0));AMX_FMA32(Fma32(2,128,0));AMX_FMA32(Fma32(3,192,0)); }
    rep("plain FMA32 (baseline)",sec(clk::now()-t0)); }
  // indexed: 4 indexed-X FMA32 gathering from table reg X[1]
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){
      AMX_FMA32(IdxFma32(0,0,0,1));AMX_FMA32(IdxFma32(1,0,0,1));AMX_FMA32(IdxFma32(2,0,0,1));AMX_FMA32(IdxFma32(3,0,0,1)); }
    rep("indexed-X FMA32 (gather)",sec(clk::now()-t0)); }

  AMX_CLR();
  std::printf("\nindexed ~= plain -> FREE fused gather: int4-codebook GEMM at fp32-FMA rate,\n"
              "dequant free, load-issue bound relieved. indexed >> plain -> gather serializes.\n");
  free(buf); return 0;
}
