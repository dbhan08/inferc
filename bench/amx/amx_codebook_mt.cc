// RIGOR: multi-thread both (the fair baseline — single-thread cblas wasn't BW-saturating).
// Codebook GEMM parallelized over N-tiles (each thread: own AMX state, disjoint N-cols),
// vs Accelerate cblas_sgemm with its own internal threading. M=16, K=2048, N=8192,
// weight-memory-bound. If the 8x-less-weight-traffic advantage is real, it should HOLD
// (or grow) multi-threaded, since DRAM bandwidth is shared and codebook streams 8x less.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}

static int K,N,NT,M;
static float *cbp,*Atp; static uint8_t* ixp; static float* Cp;

static void worker(int t0,int t1){
  const uint64_t CB=(uint64_t)cbp,IX=(uint64_t)ixp,AT=(uint64_t)Atp;
  alignas(64) float zb[16]={0}; const uint64_t ZB=(uint64_t)zb;
  AMX_SET();
  for(int t=t0;t<t1;++t){
    AMX_LDX((CB+(size_t)t*16*4)|(1ULL<<56));
    for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));
    for(int k=0;k<K;++k){
      AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));
      AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));
      AMX_MATFP(MatfpIdxX(0,0,0,1));
    }
    for(int j=0;j<16;++j) AMX_STZ(((uint64_t)Cp+((size_t)j*N + t*16)*4)|((uint64_t)(4*j)<<56));
  }
  AMX_CLR();
}

int main(){
  K=2048;N=8192;NT=N/16;M=16;
  Rng r{0x99ULL};
  std::vector<float> B((size_t)K*N); for(auto&x:B) x=(float)nrm(r);
  std::vector<float> cb((size_t)NT*16+16,0.f); std::vector<uint8_t> idxp((size_t)NT*K*8+64,0);
  for(int t=0;t<NT;++t){ for(int e=0;e<16;++e) cb[t*16+e]=(float)nrm(r)*1.2f; std::sort(&cb[t*16],&cb[t*16+16]);
    for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; float w=B[(size_t)k*N+n];
      int be=0;float bd=1e30f;for(int e=0;e<16;++e){float d=std::fabs(cb[t*16+e]-w);if(d<bd){bd=d;be=e;}}
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(be&0xF)<<((c&1)*4);} }
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*M+m];
  std::vector<float> C((size_t)M*N,0), Cf((size_t)M*N);
  cbp=cb.data();Atp=At.data();ixp=idxp.data();Cp=C.data();

  auto run_mt=[&](int T){ std::vector<std::thread> th; int per=(NT+T-1)/T;
    for(int i=0;i<T;++i){int a=i*per,b=std::min(NT,a+per); if(a<b) th.emplace_back(worker,a,b);} for(auto&x:th)x.join(); };
  auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<5;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};

  std::printf("M=%d K=%d N=%d (weight fp32 67MB vs int4 8MB), multi-thread:\n",M,K,N);
  for(int T : {1,2,4,8}){ double t=best([&](){run_mt(T);}); std::printf("  codebook %d-thread: %.2f ms\n",T,t); }
  double tcb=best([&](){cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,B.data(),N,0.f,Cf.data(),N);});
  std::printf("  Accelerate cblas (all cores): %.2f ms\n",tcb);
  double tbestcb=best([&](){run_mt(8);});
  std::printf("speedup (codebook 8-thread vs Accelerate all-core): %.2fx %s\n", tcb/tbestcb, tcb/tbestcb>1?"<-- WINS MT":"(cblas wins MT)");
  return 0;
}
