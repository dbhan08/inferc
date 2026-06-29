// Full optimized M-curve, ST and MT, vs the full fair ggml curve. Completes the speed
// figure (we previously had ours-MT only at M=16). Specialized templated kernel, threaded
// over N-tile blocks (spawn amortized). ggml repacked reference (measured): see table.
// K=2048 N=8192.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};

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

int main(){
  const int K=2048,N=8192,NT=N/16;
  const int Ms[5]={1,4,16,32,64};
  const double g1[5]={0.25,0.61,2.02,4.02,8.03}, g8[5]={0.18,0.25,0.59,1.09,2.06};  // measured fair ggml
  std::printf("OPTIMIZED M-curve ST+MT vs fair ggml, K=%d N=%d:\n",K,N);
  std::printf("%-4s | our 1T   our 8T  | ggml 1T  ggml 8T | ST x   MT x\n","M");
  for(int mi=0;mi<5;++mi){ int M=Ms[mi],MT=(M+15)/16,NB=4/MT,MP=MT*16,NBB=NT/NB; Rng r{0x100ULL+M};
    std::vector<float> At((size_t)K*MP,0.f),scale(N); for(int k=0;k<K;++k) for(int m=0;m<M;++m) At[(size_t)k*MP+m]=(float)nrm(r);
    for(auto&x:scale)x=0.2f+0.8f*(float)r.u();
    std::vector<uint8_t> idxb((size_t)NBB*K*(NB*8)+64,0);
    for(size_t i=0;i<(size_t)NBB*K*(NB*8);++i) idxb[i]=(uint8_t)(r.u()*256);
    std::vector<float> C((size_t)M*N,0);
    alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e]; alignas(64) float zb[16]={0};
    const uint64_t CBp=(uint64_t)cb,IB=(uint64_t)idxb.data(),AT=(uint64_t)At.data(),ZB=(uint64_t)zb;
    auto disp=[&](int blk0,int blk1){ if(MT==1) worker<1,4>(blk0,blk1,M,K,N,CBp,IB,AT,ZB,C.data());
      else if(MT==2) worker<2,2>(blk0,blk1,M,K,N,CBp,IB,AT,ZB,C.data()); else worker<4,1>(blk0,blk1,M,K,N,CBp,IB,AT,ZB,C.data()); };
    auto best=[&](int T){ const int R=40,per=(NBB+T-1)/T; double bb=1e30;
      for(int tr=0;tr<3;++tr){ auto t0=clk::now(); std::vector<std::thread> th;
        for(int i=0;i<T;++i){int a=i*per,b=std::min(NBB,a+per); if(a<b) th.emplace_back([=](){for(int r2=0;r2<R;++r2) disp(a,b);});}
        for(auto&x:th)x.join(); bb=std::min(bb,ms(clk::now()-t0)/R);} return bb; };
    double t1=best(1),t8=best(8);
    std::printf("%-4d | %6.3f  %6.3f | %6.2f   %6.2f  | %.2f  %.2f\n",M,t1,t8,g1[mi],g8[mi],g1[mi]/t1,g8[mi]/t8);
  }
  std::printf("(ST: AMX wins M>=4; MT: AMX block-limited (~2 blocks) vs ggml 8-core NEON)\n");
  return 0;
}
