// FAIR BASELINE (the decisive limitation): AMX codebook GEMM vs a NEON int4 codebook
// GEMM -- both read the SAME 8MB of 4-bit weights + codebook (apples-to-apples low-bit),
// unlike the fp32-cblas comparison. Tells us whether the AMX matrix engine + fused gather
// actually beats hand-NEON int4, or just beats fp32. Same shape M=16 K=2048 N=8192.
// NEON kernel: N-tiled by 4 (reuse A across 4 cols), vfmaq_n with scalar codebook deq.
// CAVEAT: the NEON kernel is reasonable but NOT as tuned as llama.cpp Q4 -- a ballpark.

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
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
  const int M=16,K=2048,N=8192,NT=N/16;
  Rng r{0x2222ULL};
  std::vector<float> B((size_t)K*N); for(auto&x:B) x=(float)nrm(r);
  std::vector<float> cb((size_t)NT*16+16,0.f);
  std::vector<uint8_t> idxp((size_t)NT*K*8+64,0);          // AMX layout [tile][k]->8B
  std::vector<uint8_t> idxn((size_t)N*(K/2)+64,0);         // NEON layout [n][k] 4-bit along k
  std::vector<float> Bdq((size_t)K*N);
  for(int t=0;t<NT;++t){ for(int e=0;e<16;++e) cb[t*16+e]=(float)nrm(r)*1.2f; std::sort(&cb[t*16],&cb[t*16+16]);
    for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; float w=B[(size_t)k*N+n];
      int be=0;float bd=1e30f;for(int e=0;e<16;++e){float d=std::fabs(cb[t*16+e]-w);if(d<bd){bd=d;be=e;}}
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(be&0xF)<<((c&1)*4);
      idxn[(size_t)n*(K/2)+k/2]|=(uint8_t)(be&0xF)<<((k&1)*4);
      Bdq[(size_t)k*N+n]=cb[t*16+be]; } }
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*M+m];
  std::vector<float> Camx((size_t)M*N,0), Cneon((size_t)M*N,0), Cref((size_t)M*N,0);
  alignas(64) float zb[16]={0}; const uint64_t CB=(uint64_t)cb.data(),IX=(uint64_t)idxp.data(),ZB=(uint64_t)zb,AT=(uint64_t)At.data();

  auto amx=[&](){ AMX_SET();
    for(int t=0;t<NT;++t){ AMX_LDX((CB+(size_t)t*16*4)|(1ULL<<56));
      for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));
      for(int k=0;k<K;++k){ AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56)); AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56)); AMX_MATFP(MatfpIdxX(0,0,0,1)); }
      for(int j=0;j<16;++j) AMX_STZ(((uint64_t)Camx.data()+((size_t)j*N+t*16)*4)|((uint64_t)(4*j)<<56)); }
    AMX_CLR(); };

  auto neon=[&](){
    for(int n0=0;n0<N;n0+=4){
      const float* cbc[4]; for(int c=0;c<4;++c) cbc[c]=&cb[((n0+c)/16)*16];
      float32x4_t ac[4][4];
      for(int c=0;c<4;++c) for(int q=0;q<4;++q) ac[c][q]=vdupq_n_f32(0);
      for(int k=0;k<K;++k){
        const float* a=&At[(size_t)k*M];
        float32x4_t a0=vld1q_f32(a),a1=vld1q_f32(a+4),a2=vld1q_f32(a+8),a3=vld1q_f32(a+12);
        for(int c=0;c<4;++c){ int n=n0+c; uint8_t by=idxn[(size_t)n*(K/2)+k/2]; int e=(k&1)?(by>>4):(by&0xF); float w=cbc[c][e];
          ac[c][0]=vfmaq_n_f32(ac[c][0],a0,w); ac[c][1]=vfmaq_n_f32(ac[c][1],a1,w);
          ac[c][2]=vfmaq_n_f32(ac[c][2],a2,w); ac[c][3]=vfmaq_n_f32(ac[c][3],a3,w); }
      }
      for(int c=0;c<4;++c){ float tmp[16]; for(int q=0;q<4;++q) vst1q_f32(tmp+q*4,ac[c][q]);
        for(int m=0;m<16;++m) Cneon[(size_t)m*N+n0+c]=tmp[m]; }
    } };

  amx(); neon();
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.f,A.data(),K,Bdq.data(),N,0.f,Cref.data(),N);
  double eA=0,eN=0; for(size_t i=0;i<(size_t)M*N;i+=131){ double d=std::fabs(Cref[i]); eA=std::max(eA,std::fabs(Camx[i]-Cref[i])/(d+1e-6)); eN=std::max(eN,std::fabs(Cneon[i]-Cref[i])/(d+1e-6)); }
  std::printf("correctness: AMX max-rel=%.1e  NEON max-rel=%.1e\n",eA,eN);

  auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<5;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
  double ta=best(amx), tn=best(neon);
  std::printf("M=%d K=%d N=%d, BOTH int4 (8MB weights), single-thread:\n",M,K,N);
  std::printf("  AMX codebook (indexed MATFP): %.2f ms\n",ta);
  std::printf("  NEON int4 codebook:           %.2f ms\n",tn);
  std::printf("AMX vs NEON-int4: %.2fx  %s\n", tn/ta, tn/ta>1.05?"<-- AMX wins the fair int4 race":(tn/ta<0.95?"NEON int4 wins":"~parity"));
  return 0;
}
