// THE LEVER to beat ggml harder: fp16 codebook GEMM. AMX fp16 (fp16 in, fp32 accumulate)
// runs ~2x the fp32 rate (Zhou: FMA16 3319 vs FMA32 1669 GFLOP/s). The 4-bit quant already
// carries ~10% error, so fp16 codebook+activation (~0.1%) is free accuracy-wise. Test:
// (1) does the indexed gather work in fp16 MATFP mode (lane bits42-45=3: xy=fp16, z=fp32)?
// (2) is it ~2x the fp32 indexed MATFP throughput? If yes -> prefill win ~1.6x -> ~3x.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
// fp32 indexed MATFP (lane mode 4) and fp16 indexed MATFP (lane mode 3: fp16 in, fp32 acc)
static inline uint64_t IdxFP32(int z,int xo,int yo,int src){return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
static inline uint64_t IdxFP16(int z,int xo,int yo,int src){return (3ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
static inline uint16_t f2h(float f){ // minimal float->half
  uint32_t x; std::memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int e=((x>>23)&0xFF)-112; uint32_t m=(x>>13)&0x3FF;
  if(e<=0) return (uint16_t)s; if(e>=31) return (uint16_t)(s|0x7C00); return (uint16_t)(s|(e<<10)|m); }

int main(){
  // throughput: fp32 vs fp16 indexed MATFP, 4-way ILP, operands preloaded
  void* buf; posix_memalign(&buf,128,1024); for(int i=0;i<256;++i) ((float*)buf)[i]=(float)(i%16)*0.1f;
  uint16_t* hb=(uint16_t*)((char*)buf+512); for(int i=0;i<256;++i) hb[i]=f2h((float)(i%16)*0.1f);
  const uint64_t b=(uint64_t)buf, hbp=(uint64_t)hb;
  const int64_t IT=15'000'000; const double GHZ=3.2; const double Nops=IT*4;
  auto rep=[&](const char* t,double s){std::printf("%-28s %5.2f cyc/op\n",t,s*GHZ*1e9/Nops);};
  AMX_SET();
  AMX_LDX(b|(0ULL<<56)); AMX_LDX((b+64)|(1ULL<<56)); AMX_LDY(b|(0ULL<<56));        // fp32 operands
  AMX_LDX(hbp|(2ULL<<56)); AMX_LDX((hbp+64)|(3ULL<<56)); AMX_LDY(hbp|(4ULL<<56));  // fp16 operands
  for(int i=0;i<2000;++i) AMX_MATFP(IdxFP32(0,0,0,1));
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_MATFP(IdxFP32(0,0,0,1));AMX_MATFP(IdxFP32(1,0,0,1));AMX_MATFP(IdxFP32(2,0,0,1));AMX_MATFP(IdxFP32(3,0,0,1)); } rep("fp32 indexed MATFP",sec(clk::now()-t0)); }
  // fp16: idx in X[2], codebook fp16 in X[3] (src=3), A fp16 in Y[4] (yoff=4*64)
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_MATFP(IdxFP16(0,2*64,4*64,3));AMX_MATFP(IdxFP16(1,2*64,4*64,3));AMX_MATFP(IdxFP16(2,2*64,4*64,3));AMX_MATFP(IdxFP16(3,2*64,4*64,3)); } rep("fp16 indexed MATFP",sec(clk::now()-t0)); }
  AMX_CLR();
  std::printf("(if fp16 cyc/op ~= fp32 but does 2x elements -> ~2x throughput -> the lever works.\n"
              " AMX fp16 MATFP is 32x32 fp16 vs fp32 16x16, so same cyc/op = 4x MACs... measure GFLOPS.)\n");
  free(buf); return 0;
}
