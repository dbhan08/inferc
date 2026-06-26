// Indexed MATFP gather is VERIFIED correct (amx_matfp_indexed.cc). Now: is it FREE?
// Measure cyc/op for plain FMA32 (ref), plain MATFP (fp32), and INDEXED MATFP (fused
// 16-entry-codebook gather + outer product). If indexed MATFP ~= plain MATFP ~= FMA32,
// the gather is free -> int4-codebook GEMM at fp32-matmul rate, dequant fused, and the
// weight operand is GATHERED from a 16-fp32 register table (loaded once) instead of
// streamed from memory -> directly relieves the load-issue bound.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t Fma32(int z,int xo){return (uint64_t(z)<<20)|(uint64_t(xo)<<10);}
static inline uint64_t Matfp(int z,int xo,int yo){return (4ULL<<42)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}

int main(){
  float* buf=nullptr; posix_memalign((void**)&buf,128,512); for(int i=0;i<128;++i) buf[i]=(float)(i%16)*0.1f;
  const uint64_t b=(uint64_t)buf;
  const int64_t IT=15'000'000; const double GHZ=3.2; const double N=IT*4;
  auto rep=[&](const char* t,double s){std::printf("%-30s %5.2f cyc/op\n",t,s*GHZ*1e9/N);};
  AMX_SET();
  AMX_LDX(b|(0ULL<<56)); AMX_LDX((b+64)|(1ULL<<56)); AMX_LDY(b|(0ULL<<56));
  for(int i=0;i<2000;++i) AMX_FMA32(Fma32(0,0));
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_FMA32(Fma32(0,0));AMX_FMA32(Fma32(1,64));AMX_FMA32(Fma32(2,128));AMX_FMA32(Fma32(3,192)); } rep("plain FMA32 (ref)",sec(clk::now()-t0)); }
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_MATFP(Matfp(0,0,0));AMX_MATFP(Matfp(1,64,0));AMX_MATFP(Matfp(2,128,0));AMX_MATFP(Matfp(3,192,0)); } rep("plain MATFP fp32",sec(clk::now()-t0)); }
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_MATFP(MatfpIdxX(0,0,0,1));AMX_MATFP(MatfpIdxX(1,0,0,1));AMX_MATFP(MatfpIdxX(2,0,0,1));AMX_MATFP(MatfpIdxX(3,0,0,1)); } rep("INDEXED MATFP (gather)",sec(clk::now()-t0)); }
  AMX_CLR();
  std::printf("\nindexed MATFP ~= plain MATFP -> FREE fused gather (int4-codebook GEMM at\n"
              "fp32-matmul rate, dequant fused, weight gathered from reg not memory).\n");
  free(buf); return 0;
}
