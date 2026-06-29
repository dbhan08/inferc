// int8 codebook GEMM throughput (the matched-precision prefill widener). M=64 (fills the
// 64-wide int8 tile), K=2048 N=8192. Per N-tile(16): LDZ zero, K-loop {LDX A[k] (64 int8),
// LDY idx[k] (stride-4 packed), MATINT-indexed signed}, STZ. Time vs ggml int8 (M=64: ST
// 8.03 / MT 2.06 ms) and our fp32 (M=64 ST 2.86 ms). Correctness verified in amx_codebook_int8.cc.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatintIdxY(int z,int xo,int yo,int src){
  return (10ULL<<42)|(1ULL<<53)|(1ULL<<54)|(1ULL<<48)|(1ULL<<47)|(1ULL<<63)|(1ULL<<26)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
static int K,N,M,NT; static int8_t* Atp; static uint8_t* ixp; static int8_t* cbp; static int32_t* zb; static int32_t* Cp;

static void worker(int t0,int t1){
  alignas(64) int32_t z0[16]={0};
  AMX_SET(); AMX_LDY((uint64_t)cbp|(1ULL<<56));
  for(int t=t0;t<t1;++t){
    for(int r=0;r<64;++r) AMX_LDZ((uint64_t)z0|((uint64_t)r<<56));
    for(int k=0;k<K;++k){ AMX_LDX((uint64_t)(Atp+(size_t)k*64)|(0ULL<<56));
      AMX_LDY((uint64_t)(ixp+((size_t)t*K+k)*32)|(0ULL<<56)); AMX_MATINT(MatintIdxY(0,0,0,1)); }
    for(int r=0;r<64;++r) AMX_STZ((uint64_t)(Cp+((size_t)t*64*16)+r*16)|((uint64_t)r<<56));   // raw Z dump per tile
  }
  AMX_CLR();
}
int main(){
  K=2048;N=8192;M=64;NT=N/16;
  std::vector<int8_t> At((size_t)K*64,1); std::vector<int8_t> cb(64,1);
  std::vector<uint8_t> idxp((size_t)NT*K*32+64,1);
  std::vector<int32_t> C((size_t)NT*64*16); alignas(64) int32_t z0[16]={0};
  Atp=At.data();ixp=idxp.data();cbp=cb.data();zb=z0;Cp=C.data();
  auto best=[&](int T){ const int R=4,per=(NT+T-1)/T; double bb=1e30;
    for(int tr=0;tr<3;++tr){ auto t0=clk::now(); std::vector<std::thread> th;
      for(int i=0;i<T;++i){int a=i*per,b=std::min(NT,a+per); if(a<b) th.emplace_back([=](){for(int r=0;r<R;++r) worker(a,b);});}
      for(auto&x:th)x.join(); bb=std::min(bb,ms(clk::now()-t0)/R);} return bb; };
  double F=2.0*M*N*K;
  std::printf("int8 codebook GEMM M=%d K=%d N=%d:\n",M,K,N);
  for(int T:{1,2,4,8}){ double t=best(T); std::printf("  %d-thread: %.2f ms (%.0f G-int8op/s)\n",T,t,F/(t/1e3)/1e9); }
  std::printf("[ref M=64: ggml int8 ST 8.03 / MT 2.06 ms ; our fp32 ST 2.86 ms]\n");
  return 0;
}
