// END-TO-END (GEMM-stack): full OPT-125M forward matmul cost, our codebook kernel vs the
// matmul time, at prefill (M=seq) and decode (M=1). LLM inference is matmul-dominated, so
// summing all layer GEMMs ~ the model's compute. Our kernel tries T=1..8 and reports best
// (AMX is dual-cluster ~2x max, confirmed); pair with ggml (ggq4r, 8T) summed over the same
// shapes in the driver script. tok/s = M / total_GEMM_ms. OPT-125M: 12 layers x {q,k,v,out
// [768,768], fc1 [768,3072], fc2 [3072,768]}.

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
template<int MT,int NB>
static void worker(int blk0,int blk1,int M,int K,int N,uint64_t CBp,uint64_t IB,uint64_t AT,uint64_t ZB,float* C){
  const int MP=MT*16;
  AMX_SET(); AMX_LDX(CBp|(1ULL<<56));
  for(int blk=blk0;blk<blk1;++blk){
    for(int bk=0;bk<MT*NB;++bk) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+bk)<<56));
    for(int k=0;k<K;++k){ AMX_LDX((IB+((size_t)(blk*K+k))*(NB*8))|(0ULL<<56));
      for(int mt=0;mt<MT;++mt){ AMX_LDY((AT+((size_t)k*MP+mt*16)*4)|(0ULL<<56));
        for(int nt=0;nt<NB;++nt) AMX_MATFP(MatfpIdxX(mt*NB+nt,nt*8,0,1)); } }
    for(int mt=0;mt<MT;++mt) for(int nt=0;nt<NB;++nt){ int tile=blk*NB+nt,b=mt*NB+nt;
      for(int j=0;j<16;++j){ int m=mt*16+j; if(m<M) AMX_STZ(((uint64_t)C+((size_t)m*N+tile*16)*4)|((uint64_t)(4*j+b)<<56)); } }
  }
  AMX_CLR();
}
static double time_shape(int K,int N,int M){
  int MT=(M+15)/16, NB=4/MT, MP=MT*16, NBB=(N/16)/NB;
  std::vector<float> At((size_t)K*MP,0.01f); std::vector<uint8_t> idxb((size_t)NBB*K*(NB*8)+64,1);
  alignas(64) float cb[16]; for(int e=0;e<16;++e)cb[e]=0.1f; alignas(64) float zb[16]={0};
  std::vector<float> C((size_t)M*N,0); float* Cd=C.data();
  const uint64_t CBp=(uint64_t)cb,IB=(uint64_t)idxb.data(),AT=(uint64_t)At.data(),ZB=(uint64_t)zb;
  auto disp=[&](int a,int b){ if(MT==1) worker<1,4>(a,b,M,K,N,CBp,IB,AT,ZB,Cd);
    else if(MT==2) worker<2,2>(a,b,M,K,N,CBp,IB,AT,ZB,Cd); else worker<4,1>(a,b,M,K,N,CBp,IB,AT,ZB,Cd); };
  auto bestT=[&](int T){ const int R=30,per=(NBB+T-1)/T; double bb=1e30;
    for(int tr=0;tr<3;++tr){ auto t0=clk::now(); std::vector<std::thread> th;
      for(int i=0;i<T;++i){int a=i*per,b=std::min(NBB,a+per); if(a<b) th.emplace_back([=](){for(int r=0;r<R;++r) disp(a,b);});}
      for(auto&x:th)x.join(); bb=std::min(bb,ms(clk::now()-t0)/R);} return bb; };
  double b=1e30; for(int T:{1,2,4,8}) b=std::min(b,bestT(T)); return b;   // best over thread counts
}
int main(int argc,char**argv){
  int M=atoi(argv[1]); const int L=12;
  // OPT-125M per-layer GEMM shapes (K,N): q,k,v,out [768,768] x4, fc1 [768,3072], fc2 [3072,768]
  int shapes[6][2]={{768,768},{768,768},{768,768},{768,768},{768,3072},{3072,768}};
  double total=0;
  for(int l=0;l<L;++l) for(int s=0;s<6;++s) total+=time_shape(shapes[s][0],shapes[s][1],M);
  std::printf("OPT-125M GEMM-stack (12 layers), M=%d: our best-AMX total = %.3f ms -> %.0f tok/s\n",
              M, total, M/(total/1e3));
  return 0;
}
