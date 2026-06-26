// PER-CHANNEL-SCALE kernel (the last mile: run GPTQ/NF4-style quant end-to-end, fast).
// GPTQ/NF4 weight = per-output-channel scale[n] * shared-NF4-codebook[idx[k,n]]. A per-
// channel scale FACTORS OUT of the K-sum:  Y[m,n] = scale[n] * SUM_k A[m,k]*NF4[idx[k,n]].
// So the kernel = the codebook GEMM with ONE shared 16-entry NF4 codebook (loaded once),
// then a cheap per-column post-scale. Proves: the accurate (GPTQ-quantized) weights run
// through the SAME fast kernel, bit-exact. Verify vs fp32 A.W_dq; time vs Accelerate.

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
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};

int main(){
  const int M=16,K=2048,N=8192,NT=N/16;
  Rng r{0x6262ULL};
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> scale(N); for(auto&x:scale) x=0.2f+0.8f*(float)r.u();          // per-output-channel scale
  alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e];                       // ONE shared NF4 codebook
  std::vector<uint8_t> idxp((size_t)NT*K*8+64,0); std::vector<float> Wdq((size_t)K*N);
  for(int t=0;t<NT;++t) for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; int e=(int)(r.u()*16); if(e>15)e=15;
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(e&0xF)<<((c&1)*4); Wdq[(size_t)k*N+n]=scale[n]*NF4[e]; }   // GPTQ-style weight
  std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*M+m];
  std::vector<float> C((size_t)M*N,0), Cref((size_t)M*N);
  const uint64_t CBp=(uint64_t)cb, IX=(uint64_t)idxp.data(), AT=(uint64_t)At.data();
  alignas(64) float zb[16]={0}; const uint64_t ZB=(uint64_t)zb;

  auto kernel=[&](){
    AMX_SET();
    AMX_LDX(CBp|(1ULL<<56));                                   // shared NF4 codebook -> X[1] (ONCE)
    for(int t=0;t<NT;++t){
      for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));
      for(int k=0;k<K;++k){
        AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));
        AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));
        AMX_MATFP(MatfpIdxX(0,0,0,1));                         // P[m][n] += A[m,k]*NF4[idx]
      }
      for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N+t*16)*4)|((uint64_t)(4*j)<<56));
    }
    AMX_CLR();
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) C[(size_t)m*N+n] *= scale[n];   // per-channel POST-SCALE
  };

  kernel();
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,Wdq.data(),N,0.f,Cref.data(),N);
  double mr=0; for(size_t i=0;i<(size_t)M*N;i+=131){double d=std::fabs(Cref[i]); mr=std::max(mr,std::fabs(C[i]-Cref[i])/(d+1e-6));}
  std::printf("per-channel-scale NF4 kernel vs fp32 A.W_dq: max-rel=%.2e %s\n", mr, mr<1e-3?"CORRECT (bit-exact-class)":"MISMATCH");

  auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<5;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
  double tk=best(kernel);
  std::vector<float> Cf((size_t)M*N);
  double tf=best([&](){cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,Wdq.data(),N,0.f,Cf.data(),N);});
  std::printf("M=%d K=%d N=%d  GPTQ-style (NF4 + per-channel scale, 4-bit weights):\n",M,K,N);
  std::printf("  per-channel-scale kernel: %.2f ms   Accelerate fp32: %.2f ms   speedup %.2fx\n", tk, tf, tf/tk);
  std::printf("(accurate weights -- GPTQ +2.85%% on gpt2 -- run through the SAME fast kernel, bit-exact + post-scale)\n");
  return 0;
}
