// Robustness: our optimized kernel at M=16 across real-model shapes (K,N from argv),
// ST + MT(8). Pair with ggml at the same shapes (ggml_q4_repack_bench takes argv too).
// Tests whether the small-batch speed win holds across attn/FFN shapes, not just one.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
static void worker(int blk0,int blk1,int K,int N,uint64_t CBp,uint64_t IB,uint64_t AT,uint64_t ZB,float* C){
  AMX_SET(); AMX_LDX(CBp|(1ULL<<56));               // M=16, MT=1, NB=4 (1x4 blocking)
  for(int blk=blk0;blk<blk1;++blk){
    for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+b)<<56));
    for(int k=0;k<K;++k){ AMX_LDX((IB+((size_t)(blk*K+k))*32)|(0ULL<<56)); AMX_LDY((AT+(size_t)k*16*4)|(0ULL<<56));
      for(int b=0;b<4;++b) AMX_MATFP(MatfpIdxX(b,b*8,0,1)); }
    for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C+((size_t)j*N+(4*blk+b)*16)*4)|((uint64_t)(4*j+b)<<56));
  }
  AMX_CLR();
}
int main(int argc,char**argv){
  int K=atoi(argv[1]),N=atoi(argv[2]),M=16,NT=N/16,NBB=NT/4;
  std::vector<float> At((size_t)K*M); for(auto&x:At)x=0.01f;
  std::vector<uint8_t> idxb((size_t)NBB*K*32+64,1);
  alignas(64) float cb[16]; for(int e=0;e<16;++e)cb[e]=0.1f*e; alignas(64) float zb[16]={0};
  std::vector<float> C((size_t)M*N,0); float* Cd=C.data();
  const uint64_t CBp=(uint64_t)cb,IB=(uint64_t)idxb.data(),AT=(uint64_t)At.data(),ZB=(uint64_t)zb;
  auto best=[&](int T){ const int R=40,per=(NBB+T-1)/T; double bb=1e30;
    for(int tr=0;tr<3;++tr){ auto t0=clk::now(); std::vector<std::thread> th;
      for(int i=0;i<T;++i){int a=i*per,b=std::min(NBB,a+per); if(a<b) th.emplace_back([=](){for(int r=0;r<R;++r) worker(a,b,K,N,CBp,IB,AT,ZB,Cd);});}
      for(auto&x:th)x.join(); bb=std::min(bb,ms(clk::now()-t0)/R);} return bb; };
  std::printf("%.3f %.3f\n", best(1), best(8));   // ourST ourMT
  return 0;
}
