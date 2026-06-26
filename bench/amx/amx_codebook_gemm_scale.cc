// MEMORY-BOUND codebook GEMM vs Accelerate fp32 — the regime the mechanism is FOR.
// Shape: M=16 (small batch -> weight-memory-bound: arith intensity M/2=8 FLOP/byte <
// M1's ~17 compute/BW ratio), K=2048, N=8192 (FFN1-ish). Weight B = 2048x8192 fp32 =
// 64 MB >> 12 MB L2, so the GEMM streams weights from DRAM. Codebook path stores B as
// 4-bit indices (8 MB) + per-N-tile 16-fp32 codebook, dequant fused via indexed MATFP,
// so it streams ~8x less weight memory. Compare wall-clock vs cblas_sgemm on fp32 B.
// If memory-bound, 8x less weight traffic should win despite our slower compute.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  const int M=16,K=2048,N=8192,NT=N/16;            // NT N-tiles of 16 cols
  Rng r{0x5151ULL};
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);   // A transposed [K][16]
  std::vector<float> B((size_t)K*N);  for(auto&x:B)  x=(float)nrm(r);   // fp32 weights (64MB)
  // quantize B to per-N-tile 16-entry codebook + 4-bit indices.
  std::vector<float> cb((size_t)NT*16 + 16, 0.f);
  std::vector<uint8_t> idxp((size_t)NT*K*8 + 64, 0);                    // [tile][k] -> 8 bytes (16 nibbles)
  std::vector<float> Bdq((size_t)K*N);                                  // dequantized (for ref)
  for(int t=0;t<NT;++t){
    // codebook = 16 sorted samples of this tile's weights (non-uniform)
    for(int e=0;e<16;++e) cb[t*16+e]=(float)nrm(r)*1.2f; std::sort(&cb[t*16],&cb[t*16+16]);
    for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; float w=B[(size_t)k*N+n];
      // nearest codebook entry
      int be=0; float bd=1e30f; for(int e=0;e<16;++e){float d=std::fabs(cb[t*16+e]-w); if(d<bd){bd=d;be=e;}}
      idxp[((size_t)t*K+k)*8 + c/2] |= (uint8_t)(be&0xF)<<((c&1)*4);
      Bdq[(size_t)k*N+n]=cb[t*16+be]; }
  }
  std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*M+m];

  alignas(64) float zb[16]={0}; std::vector<float> C((size_t)M*N,0), Cref((size_t)M*N,0);
  const uint64_t AT=(uint64_t)At.data(), CB=(uint64_t)cb.data(), IX=(uint64_t)idxp.data(), ZB=(uint64_t)zb;

  auto codebook_gemm=[&](){
    AMX_SET();
    for(int t=0;t<NT;++t){
      AMX_LDX((CB+(size_t)t*16*4)|(1ULL<<56));                          // codebook for this N-tile -> X[1]
      for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));          // zero Z (rows 4*j)
      for(int k=0;k<K;++k){
        AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));                     // 16 weight indices -> X[0]
        AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));                         // A col k -> Y[0]
        AMX_MATFP(MatfpIdxX(0,0,0,1));
      }
      for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N + t*16)*4)|((uint64_t)(4*j)<<56));
    }
    AMX_CLR();
  };

  // correctness (sample) vs cblas on the DEQUANTIZED weights
  codebook_gemm();
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,Bdq.data(),N,0.f,Cref.data(),N);
  double maxrel=0; for(size_t i=0;i<(size_t)M*N;i+=257){ double e=std::fabs(C[i]-Cref[i]),d=std::fabs(Cref[i]); double rl=d>1e-4?e/d:e; maxrel=std::max(maxrel,rl);}
  std::printf("correctness (codebook GEMM vs cblas-on-dequant): max-rel=%.2e %s\n", maxrel, maxrel<1e-3?"OK":"CHECK");

  auto best=[&](auto fn){ fn();fn(); double b=1e30; for(int i=0;i<5;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));} return b; };
  double tcb = best(codebook_gemm);
  std::vector<float> Cf((size_t)M*N);
  double tfp = best([&](){ cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,B.data(),N,0.f,Cf.data(),N); });

  std::printf("M=%d K=%d N=%d  weight: fp32 %.0fMB vs int4 %.0fMB (8x)\n", M,K,N, (double)K*N*4/1e6, (double)K*N/2/1e6);
  std::printf("codebook GEMM (indexed MATFP, 4-bit wt): %.2f ms\n", tcb);
  std::printf("Accelerate cblas_sgemm (fp32 wt):        %.2f ms\n", tfp);
  std::printf("speedup: %.2fx  %s\n", tfp/tcb, tfp/tcb>1?"<-- codebook WINS (memory-bound regime)":"(cblas faster here)");
  return 0;
}
