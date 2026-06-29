// Precision-MATCH probe: ggml Q4 uses int8 activations (Q8 DOTPROD). To compare fairly we
// run int8 too -> AMX MATINT (int8xint8->int32, alumode 8 / lane mode 10). Q: is MATINT i8
// faster than MATFP fp32 on M1? (int8 packs more MACs/instruction.) If yes, matching ggml's
// precision is ALSO a speed lever. Measure cyc/op for plain MATFP fp32 vs MATINT i8, 4-way ILP.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t Matfp32(int z,int xo,int yo){return (4ULL<<42)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
// MATINT int8xint8->int32: alumode 8 (bits47-52) + lane mode 10 (bits42-45)
static inline uint64_t Matint8(int z,int xo,int yo){return (10ULL<<42)|(8ULL<<47)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}

int main(){
  void* buf; posix_memalign(&buf,128,1024); for(int i=0;i<1024;++i) ((uint8_t*)buf)[i]=(uint8_t)(i&0x7);
  const uint64_t b=(uint64_t)buf;
  const int64_t IT=15'000'000; const double GHZ=3.2; const double Nops=IT*4;
  auto rep=[&](const char* t,double s){std::printf("%-26s %5.2f cyc/op\n",t,s*GHZ*1e9/Nops);};
  AMX_SET();
  AMX_LDX(b|(0ULL<<56)); AMX_LDX((b+64)|(1ULL<<56)); AMX_LDY(b|(0ULL<<56));
  for(int i=0;i<2000;++i){ AMX_MATFP(Matfp32(0,0,0)); AMX_MATINT(Matint8(0,0,0)); }
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_MATFP(Matfp32(0,0,0));AMX_MATFP(Matfp32(1,64,0));AMX_MATFP(Matfp32(2,128,0));AMX_MATFP(Matfp32(3,192,0)); } rep("MATFP fp32 (16x16)",sec(clk::now()-t0)); }
  { auto t0=clk::now(); for(int64_t i=0;i<IT;++i){ AMX_MATINT(Matint8(0,0,0));AMX_MATINT(Matint8(1,64,0));AMX_MATINT(Matint8(2,128,0));AMX_MATINT(Matint8(3,192,0)); } rep("MATINT i8->i32",sec(clk::now()-t0)); }
  AMX_CLR();
  std::printf("(int8 packs more MACs/op: if MATINT i8 cyc/op ~ fp32 but >=2x MACs -> faster. Need MAC count.)\n");
  free(buf); return 0;
}
