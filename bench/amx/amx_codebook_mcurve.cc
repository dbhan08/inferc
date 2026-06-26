// Full M-curve: where does the codebook GEMM win, and where does it cross to a loss?
// Sweeps M=1..128. M<16 underutilizes the 16-wide outer product (GEMV regime -> AMX
// wastes lanes, DOTPROD territory). M=16..64 weight-memory-bound (codebook wins). Large
// M compute-bound (int4 loses to Accelerate's tuned AMX). M>64 needs >4 Z banks -> done
// in passes of 4 M-tiles. Weight-stationary within a pass. vs cblas_sgemm(fp32) same M.
// K=2048, N=8192. Single-thread both. Correctness-checked per M.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}

int main(){
  const int K=2048,N=8192,NT=N/16;
  Rng r{0x3030ULL};
  std::vector<float> B((size_t)K*N); for(auto&x:B) x=(float)nrm(r);
  std::vector<float> cb((size_t)NT*16+16,0.f); std::vector<uint8_t> idxp((size_t)NT*K*8+64,0);
  std::vector<float> Bdq((size_t)K*N);
  for(int t=0;t<NT;++t){ for(int e=0;e<16;++e) cb[t*16+e]=(float)nrm(r)*1.2f; std::sort(&cb[t*16],&cb[t*16+16]);
    for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; float w=B[(size_t)k*N+n];
      int be=0;float bd=1e30f;for(int e=0;e<16;++e){float d=std::fabs(cb[t*16+e]-w);if(d<bd){bd=d;be=e;}}
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(be&0xF)<<((c&1)*4); Bdq[(size_t)k*N+n]=cb[t*16+be]; } }
  alignas(64) float zb[16]={0}; const uint64_t CB=(uint64_t)cb.data(),IX=(uint64_t)idxp.data(),ZB=(uint64_t)zb;

  std::printf("K=%d N=%d, single-thread. M-curve (codebook int4 vs Accelerate fp32):\n",K,N);
  std::printf("%-5s %-12s %-14s %-9s %s\n","M","codebook ms","Accelerate ms","speedup","regime");
  for(int M : {1,4,16,32,64,128}){
    int MT=(M+15)/16, MP=MT*16;
    std::vector<float> At((size_t)K*MP,0.f), Ar((size_t)M*K), C((size_t)M*N,0), Cref((size_t)M*N), Cf((size_t)M*N);
    for(int k=0;k<K;++k) for(int m=0;m<M;++m){ float v=(float)nrm(r); At[(size_t)k*MP+m]=v; Ar[(size_t)m*K+k]=v; }
    const uint64_t AT=(uint64_t)At.data();
    auto cbk=[&](){ AMX_SET();
      for(int t=0;t<NT;++t){
        AMX_LDX((CB+(size_t)t*16*4)|(1ULL<<56));
        for(int p=0;p<MT;p+=4){ int nb=std::min(4,MT-p);
          for(int b=0;b<nb;++b) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+b)<<56));
          for(int k=0;k<K;++k){ AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));
            for(int b=0;b<nb;++b){ AMX_LDY((AT+((size_t)k*MP+(p+b)*16)*4)|(0ULL<<56)); AMX_MATFP(MatfpIdxX(b,0,0,1)); } }
          for(int b=0;b<nb;++b){ int gmt=p+b; for(int j=0;j<16;++j){ int mm=gmt*16+j; if(mm<M)
            AMX_STZ(((uint64_t)C.data()+((size_t)mm*N+t*16)*4)|((uint64_t)(4*j+b)<<56)); } }
        }
      }
      AMX_CLR(); };
    auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<4;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
    cbk();
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,Ar.data(),K,Bdq.data(),N,0.f,Cref.data(),N);
    double er=0; for(size_t i=0;i<(size_t)M*N;i+=97){double d=std::fabs(Cref[i]); er=std::max(er,std::fabs(C[i]-Cref[i])/(d+1e-6));}
    double tcb=best(cbk);
    double tfp=best([&](){cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,Ar.data(),K,B.data(),N,0.f,Cf.data(),N);});
    double sp=tfp/tcb;
    std::printf("%-5d %-12.2f %-14.2f %-9.2f %s%s\n",M,tcb,tfp,sp,
       sp>1.05?"codebook wins":(sp<0.95?"cblas wins":"~crossover"), er>1e-3?"  [ERR!]":"");
  }
  std::printf("\n(M<16 = GEMV/DOTPROD regime, AMX wastes lanes; M~16-64 mem-bound win; large M -> compute-bound.)\n");
  return 0;
}
