// RIGOR PASS: M-sweep crossover for the codebook GEMM vs Accelerate. Weight-stationary
// (each 4-bit weight index loaded ONCE per (N-tile,k), reused across all M-tiles via the
// Z banks 4*j+mt) so weight traffic is independent of M -- otherwise re-reading weights
// per M-tile would fake away the memory advantage. As M grows the GEMM shifts from
// weight-memory-bound (codebook wins, 8x less weight traffic) to compute-bound (int4
// loses, our compute < Accelerate). Show WHERE the crossover is. K=2048, N=8192.
// M-tiles use Z bank mt (mt=0..M/16-1, max 4 -> M<=64). vs cblas_sgemm(fp32) same M.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  const int K=2048,N=8192,NT=N/16,MAXM=64;
  Rng r{0x7777ULL};
  std::vector<float> B((size_t)K*N); for(auto&x:B) x=(float)nrm(r);
  std::vector<float> cb((size_t)NT*16+16,0.f);
  std::vector<uint8_t> idxp((size_t)NT*K*8+64,0);
  std::vector<float> Bdq((size_t)K*N);
  for(int t=0;t<NT;++t){ for(int e=0;e<16;++e) cb[t*16+e]=(float)nrm(r)*1.2f; std::sort(&cb[t*16],&cb[t*16+16]);
    for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; float w=B[(size_t)k*N+n];
      int be=0;float bd=1e30f;for(int e=0;e<16;++e){float d=std::fabs(cb[t*16+e]-w);if(d<bd){bd=d;be=e;}}
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(be&0xF)<<((c&1)*4); Bdq[(size_t)k*N+n]=cb[t*16+be]; } }
  // At[K][MAXM] (transposed A); A[MAXM][K] for cblas
  std::vector<float> At((size_t)K*MAXM); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> A((size_t)MAXM*K); for(int m=0;m<MAXM;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*MAXM+m];
  alignas(64) float zb[16]={0}; const uint64_t CB=(uint64_t)cb.data(),IX=(uint64_t)idxp.data(),ZB=(uint64_t)zb;

  std::printf("K=%d N=%d, weight fp32 %.0fMB vs int4 %.0fMB. crossover sweep:\n",K,N,(double)K*N*4/1e6,(double)K*N/2/1e6);
  std::printf("%-5s %-14s %-16s %-8s %s\n","M","codebook ms","Accelerate ms","speedup","regime");
  for(int M : {16,32,48,64}){
    int MT=M/16;
    std::vector<float> C((size_t)M*N,0);
    const uint64_t AT=(uint64_t)At.data();
    auto cbk=[&](){ AMX_SET();
      for(int t=0;t<NT;++t){
        AMX_LDX((CB+(size_t)t*16*4)|(1ULL<<56));
        for(int mt=0;mt<MT;++mt) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+mt)<<56));   // zero banks
        for(int k=0;k<K;++k){
          AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));                       // weight idx ONCE per (t,k)
          for(int mt=0;mt<MT;++mt){
            AMX_LDY((AT+((size_t)k*MAXM + mt*16)*4)|(0ULL<<56));            // A col for M-tile mt
            AMX_MATFP(MatfpIdxX(mt,0,0,1));                                 // -> bank mt
          }
        }
        for(int mt=0;mt<MT;++mt) for(int j=0;j<16;++j)
          AMX_STZ(((uint64_t)C.data()+((size_t)(mt*16+j)*N + t*16)*4)|((uint64_t)(4*j+mt)<<56));
      }
      AMX_CLR(); };
    auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<4;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
    double tcb=best(cbk);
    std::vector<float> Cf((size_t)M*N);
    double tfp=best([&](){cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,B.data(),N,0.f,Cf.data(),N);});
    double sp=tfp/tcb;
    std::printf("%-5d %-14.2f %-16.2f %-8.2f %s\n",M,tcb,tfp,sp, sp>1.05?"mem-bound (codebook WINS)":(sp<0.95?"compute-bound (cblas wins)":"~crossover"));
  }
  std::printf("\n(single-thread both. intensity M/2 crosses M1's ~17 ratio near M~34 -> expect crossover M=32-48.)\n");
  return 0;
}
