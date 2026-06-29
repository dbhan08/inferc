// Full OPTIMIZED M-curve vs ggml (the apples-to-apples speed figure for the paper).
// General blocking: MT=ceil(M/16) M-tiles, NB=4/MT N-tiles per block (MT*NB<=4 Z banks).
// Per k: 1 LDX loads NB tiles' indices (idx-amort, xo=nt*8); MT LDY (A, amortized across
// the NB N-tiles); MT*NB MATFP. instr/MATFP = 1/(MT*NB)+1/NB+1 -> small-batch (high NB)
// is most load-efficient. Shared NF4 codebook + per-channel post-scale. Verify each M.
// Compare to FAIR repacked ggml (1-thread): M1 0.25 M4 0.65 M16 2.02 M64 8.18 ms.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// SPECIALIZED kernel: MT,NB compile-time constants so loops unroll + MATFP operands fold
// to constants (the runtime-param version was ~2x slower from per-MATFP scalar overhead).
template<int MT,int NB>
static void kern(int M,int K,int N,uint64_t CBp,uint64_t IB,uint64_t AT,uint64_t ZB,float* C,const float* scale){
  const int MP=MT*16, NT=N/16, NB_blocks=NT/NB;
  AMX_SET(); AMX_LDX(CBp|(1ULL<<56));
  for(int blk=0;blk<NB_blocks;++blk){
    for(int bk=0;bk<MT*NB;++bk) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+bk)<<56));
    for(int k=0;k<K;++k){ AMX_LDX((IB+((size_t)(blk*K+k))*(NB*8))|(0ULL<<56));
      for(int mt=0;mt<MT;++mt){ AMX_LDY((AT+((size_t)k*MP+mt*16)*4)|(0ULL<<56));
        for(int nt=0;nt<NB;++nt) AMX_MATFP(MatfpIdxX(mt*NB+nt, nt*8, 0, 1)); } }
    for(int mt=0;mt<MT;++mt) for(int nt=0;nt<NB;++nt){ int tile=blk*NB+nt, b=mt*NB+nt;
      for(int j=0;j<16;++j){ int m=mt*16+j; if(m<M) AMX_STZ(((uint64_t)C+((size_t)m*N+tile*16)*4)|((uint64_t)(4*j+b)<<56)); } }
  }
  AMX_CLR(); for(int m=0;m<M;++m) for(int n=0;n<N;++n) C[(size_t)m*N+n]*=scale[n];
}

int main(){
  const int K=2048,N=8192,NT=N/16;
  const double ggml[5]={0.25,0.65,2.02,/*M32~*/4.0,8.18};   // fair repacked ggml 1T (M=32 interp)
  const int Ms[5]={1,4,16,32,64};
  Rng r0{0x4242ULL};
  std::vector<float> Wn((size_t)K*N); for(auto&x:Wn) x=(float)nrm(r0);   // base weights
  std::printf("OPTIMIZED M-curve, K=%d N=%d, single-thread (vs fair repacked ggml 1T):\n",K,N);
  std::printf("%-5s %-6s %-12s %-11s %s\n","M","MTxNB","our ms","GFLOP/s","ggml ms / speedup");

  for(int mi=0;mi<5;++mi){ int M=Ms[mi]; int MT=(M+15)/16, NB=4/MT, MP=MT*16, NB_blocks=NT/NB;
    Rng r{0x100ULL+M};
    std::vector<float> At((size_t)K*MP,0.f), scale(N), Wdq((size_t)K*N);
    for(int k=0;k<K;++k) for(int m=0;m<M;++m) At[(size_t)k*MP+m]=(float)nrm(r);
    for(auto&x:scale) x=0.2f+0.8f*(float)r.u();
    // idx per N-block: NB tiles' idx contiguous per k (NB*8 bytes), one LDX feeds NB MATFPs
    std::vector<uint8_t> idxb((size_t)NB_blocks*K*(NB*8)+64,0);
    for(int blk=0;blk<NB_blocks;++blk) for(int k=0;k<K;++k) for(int nt=0;nt<NB;++nt){ int tile=blk*NB+nt;
      for(int c=0;c<16;++c){ int n=tile*16+c; int e=(int)(r.u()*16); if(e>15)e=15;
        idxb[((size_t)(blk*K+k))*(NB*8) + nt*8 + c/2] |= (uint8_t)(e&0xF)<<((c&1)*4); Wdq[(size_t)k*N+n]=scale[n]*NF4[e]; } }
    std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*MP+m];
    std::vector<float> C((size_t)M*N,0);
    alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e]; alignas(64) float zb[16]={0};
    const uint64_t CBp=(uint64_t)cb,IB=(uint64_t)idxb.data(),AT=(uint64_t)At.data(),ZB=(uint64_t)zb;

    auto run=[&](){
      if(MT==1) kern<1,4>(M,K,N,CBp,IB,AT,ZB,C.data(),scale.data());
      else if(MT==2) kern<2,2>(M,K,N,CBp,IB,AT,ZB,C.data(),scale.data());
      else kern<4,1>(M,K,N,CBp,IB,AT,ZB,C.data(),scale.data()); };

    run(); double num=0,den=0; for(int m=0;m<M;++m) for(int n=0;n<N;++n){ double t=0; for(int k=0;k<K;++k) t+=(double)A[(size_t)m*K+k]*Wdq[(size_t)k*N+n]; double e=C[(size_t)m*N+n]-t; num+=e*e; den+=t*t; }
    double rel=std::sqrt(num/den);
    auto best=[&](auto fn){fn();double b=1e30;for(int i=0;i<5;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
    double t=best(run); double F=2.0*M*N*K;
    std::printf("%-5d %dx%-4d %-12.3f %-11.0f %.2f / %.2fx %s\n",M,MT,NB,t,F/(t/1e3)/1e9,ggml[mi],ggml[mi]/t, rel<1e-3?"":"[ERR]");
  }
  std::printf("(M=1 GEMV: AMX wastes lanes, ggml wins; small-batch M=4-32: our regime; M=64 ggml repacked closes)\n");
  return 0;
}
