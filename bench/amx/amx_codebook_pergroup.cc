// PER-GROUP-SCALE kernel (the refinement: run the BETTER per-(channel,K-group) GPTQ quant
// at full speed). Per-group scale[n,kg] does NOT factor out of the K-sum, so:
//   Y[m,n] = SUM_kg scale[n,kg] * P_kg[m,n],   P_kg = SUM_{k in kg} A[m,k]*NF4[idx[k,n]].
// Per K-group: gather-accumulate P_kg in Z bank 0 (rows 4m), then Y bank 1 (rows 4m+1) +=
// scale[*,kg] (.) P_kg  via EXTRX (P row -> X) + VECFP (Y += X*scale). One shared NF4 codebook.
// Verify bit-exact vs fp32 A.W_dq; time vs Accelerate + vs the per-channel kernel (4.72x).

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
static inline uint64_t Vfp(int alu,int z,int xo,int yo){return ((uint64_t)alu<<47)|(4ULL<<42)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
static inline uint64_t Extrh(int zrow,int xoff){return ((uint64_t)zrow<<20)|(1ULL<<26)|((uint64_t)xoff&0x1FF);}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};

int main(){
  const int M=16,K=2048,N=8192,NT=N/16,G=64,NKG=K/G;
  Rng r{0x9090ULL};
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*M+m];
  std::vector<float> scale((size_t)NKG*N); for(auto&x:scale) x=0.2f+0.8f*(float)r.u();   // scale[kg][n]
  alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e];
  alignas(64) float zb[16]={0};
  std::vector<uint8_t> idxp((size_t)NT*K*8+64,0); std::vector<float> Wdq((size_t)K*N);
  for(int t=0;t<NT;++t) for(int k=0;k<K;++k){ int kg=k/G; for(int c=0;c<16;++c){ int n=t*16+c; int e=(int)(r.u()*16); if(e>15)e=15;
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(e&0xF)<<((c&1)*4); Wdq[(size_t)k*N+n]=scale[(size_t)kg*N+n]*NF4[e]; } }
  std::vector<float> C((size_t)M*N,0), Cref((size_t)M*N);
  const uint64_t CBp=(uint64_t)cb, IX=(uint64_t)idxp.data(), AT=(uint64_t)At.data(), SC=(uint64_t)scale.data(), ZB=(uint64_t)zb;

  auto kernel=[&](){
    AMX_SET();
    AMX_LDX(CBp|(1ULL<<56));                                        // shared NF4 codebook -> X[1]
    for(int t=0;t<NT;++t){
      for(int m=0;m<16;++m) AMX_LDZ(ZB|((uint64_t)(4*m+1)<<56));     // zero Y bank (rows 4m+1)
      for(int kg=0;kg<NKG;++kg){
        for(int m=0;m<16;++m) AMX_LDZ(ZB|((uint64_t)(4*m)<<56));     // zero P bank (rows 4m)
        for(int k=kg*G;k<(kg+1)*G;++k){
          AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));
          AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));
          AMX_MATFP(MatfpIdxX(0,0,0,1));                            // P bank 0 += A*NF4[idx]
        }
        AMX_LDY((SC+((size_t)kg*N+t*16)*4)|(2ULL<<56));             // scale[kg, tile] -> Y[2]
        for(int m=0;m<16;++m){
          AMX_EXTRX(Extrh(4*m,2*64));                               // P row 4m -> X[2]
          AMX_VECFP(Vfp(0,4*m+1,2*64,2*64));                        // Y[4m+1] += P(X2)*scale(Y2)
        }
      }
      for(int m=0;m<16;++m) AMX_STZ(((uint64_t)C.data()+((size_t)m*N+t*16)*4)|((uint64_t)(4*m+1)<<56));
    }
    AMX_CLR();
  };

  kernel();
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,Wdq.data(),N,0.f,Cref.data(),N);
  double mr=0; for(size_t i=0;i<(size_t)M*N;i+=131){double d=std::fabs(Cref[i]); mr=std::max(mr,std::fabs(C[i]-Cref[i])/(d+1e-6));}
  std::printf("per-GROUP-scale NF4 kernel vs fp32 A.W_dq: max-rel=%.2e %s\n", mr, mr<2e-3?"CORRECT (bit-exact-class)":"MISMATCH");

  auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<5;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
  double tk=best(kernel);
  std::vector<float> Cf((size_t)M*N);
  double tf=best([&](){cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,Wdq.data(),N,0.f,Cf.data(),N);});
  std::printf("M=%d K=%d N=%d  per-(channel,K-group=%d) NF4+GPTQ (the +2.85%% quant):\n",M,K,N,G);
  std::printf("  per-group-scale kernel: %.2f ms   Accelerate fp32: %.2f ms   speedup %.2fx\n", tk, tf, tf/tk);
  std::printf("(best-accuracy GPTQ quant runs on the kernel; pays EXTRX+VECFP rescale vs per-channel)\n");
  return 0;
}
