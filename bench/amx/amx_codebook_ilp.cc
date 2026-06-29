// PROFILE-DRIVEN IMPROVEMENT: the M=16 codebook kernel is latency-bound (~4.76 cyc/MATFP
// vs ~1.15 throughput) -- single accumulation chain into one Z bank. At M=16 we use only
// 1 of 4 Z banks; the other 3 idle. Fix: split the K-sum across 4 independent bank
// accumulators (4-way ILP) so 4 MATFPs are in flight, then combine via EXTR+VECFP.
// Compare old (1-bank) vs ILP (4-bank); verify both vs fp32 ref. Per-channel NF4.

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
static inline uint64_t Vfp(int alu,int z,int xo,int yo){return ((uint64_t)alu<<47)|(4ULL<<42)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
static inline uint64_t Extrh(int zrow,int xoff){return ((uint64_t)zrow<<20)|(1ULL<<26)|((uint64_t)xoff&0x1FF);}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};

int main(){
  const int M=16,K=2048,N=8192,NT=N/16;
  Rng r{0x1212ULL};
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> scale(N); for(auto&x:scale) x=0.2f+0.8f*(float)r.u();
  alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e];
  alignas(64) float zb[16]={0}; alignas(64) float ones[16]; for(int i=0;i<16;++i) ones[i]=1.f;
  std::vector<uint8_t> idxp((size_t)NT*K*8+64,0); std::vector<float> Wdq((size_t)K*N);
  for(int t=0;t<NT;++t) for(int k=0;k<K;++k) for(int c=0;c<16;++c){ int n=t*16+c; int e=(int)(r.u()*16); if(e>15)e=15;
      idxp[((size_t)t*K+k)*8+c/2]|=(uint8_t)(e&0xF)<<((c&1)*4); Wdq[(size_t)k*N+n]=scale[n]*NF4[e]; }
  std::vector<float> A((size_t)M*K); for(int m=0;m<M;++m) for(int k=0;k<K;++k) A[(size_t)m*K+k]=At[(size_t)k*M+m];
  std::vector<float> Cref((size_t)M*N); { // fp32 ref via simple GEMM
    for(int m=0;m<M;++m) for(int n=0;n<N;++n){ double s=0; for(int k=0;k<K;++k) s+=(double)A[(size_t)m*K+k]*Wdq[(size_t)k*N+n]; Cref[(size_t)m*N+n]=(float)s; } }
  std::vector<float> C((size_t)M*N,0);
  const uint64_t CBp=(uint64_t)cb,IX=(uint64_t)idxp.data(),AT=(uint64_t)At.data(),ZB=(uint64_t)zb,ON=(uint64_t)ones;

  auto post=[&](){ for(int m=0;m<M;++m) for(int n=0;n<N;++n) C[(size_t)m*N+n]*=scale[n]; };

  auto old1=[&](){ AMX_SET(); AMX_LDX(CBp|(1ULL<<56));
    for(int t=0;t<NT;++t){ for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));
      for(int k=0;k<K;++k){ AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56)); AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56)); AMX_MATFP(MatfpIdxX(0,0,0,1)); }
      for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N+t*16)*4)|((uint64_t)(4*j)<<56)); }
    AMX_CLR(); post(); };

  auto ilp4=[&](){ AMX_SET(); AMX_LDX(CBp|(1ULL<<56)); AMX_LDY(ON|(2ULL<<56));   // ones->Y[2] for combine
    for(int t=0;t<NT;++t){
      for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+b)<<56));  // zero 4 banks
      for(int k=0;k<K;++k){ AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56)); AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));
        AMX_MATFP(MatfpIdxX(k&3,0,0,1)); }                                          // 4-way ILP over banks
      for(int j=0;j<16;++j){ for(int b=1;b<4;++b){ AMX_EXTRX(Extrh(4*j+b,2*64)); AMX_VECFP(Vfp(0,4*j,2*64,2*64)); } } // bank0 += bank b
      for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N+t*16)*4)|((uint64_t)(4*j)<<56)); }
    AMX_CLR(); post(); };

  // 4-N-tile blocked: load A[k] ONCE per k, reuse across 4 N-tiles (each -> its own bank).
  // cuts instr/MATFP from 3 (LDX+LDY+MATFP) to 2.25 (1 LDY + 4*(LDX+MATFP))/4, + 4-way ILP.
  auto blk4=[&](){ AMX_SET(); AMX_LDX(CBp|(1ULL<<56));
    for(int t=0;t<NT;t+=4){
      for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+b)<<56));
      for(int k=0;k<K;++k){ AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));        // A[k] once for 4 tiles
        for(int b=0;b<4;++b){ AMX_LDX((IX+((size_t)(t+b)*K+k)*8)|(0ULL<<56)); AMX_MATFP(MatfpIdxX(b,0,0,1)); } }
      for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N+(t+b)*16)*4)|((uint64_t)(4*j+b)<<56)); }
    AMX_CLR(); post(); };

  // idx-amortized: 4 tiles' indices (4*8=32B) fit in ONE X register; one LDX feeds all 4
  // MATFPs via per-tile byte offset (xo=b*8). -> 1 LDX + 1 LDY + 4 MATFP = 1.5 instr/MATFP.
  std::vector<uint8_t> idxblk((size_t)(NT/4)*K*32+64,0);
  for(int blk=0;blk<NT/4;++blk) for(int k=0;k<K;++k) for(int b=0;b<4;++b) for(int by=0;by<8;++by)
    idxblk[((size_t)(blk*K+k))*32 + b*8 + by] = idxp[((size_t)(4*blk+b)*K+k)*8 + by];
  const uint64_t IB=(uint64_t)idxblk.data();
  auto blk4i=[&](){ AMX_SET(); AMX_LDX(CBp|(1ULL<<56));
    for(int blk=0;blk<NT/4;++blk){
      for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+b)<<56));
      for(int k=0;k<K;++k){ AMX_LDX((IB+((size_t)(blk*K+k))*32)|(0ULL<<56)); AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));
        for(int b=0;b<4;++b) AMX_MATFP(MatfpIdxX(b,b*8,0,1)); }                 // xo=b*8 -> tile b's idx
      for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N+(4*blk+b)*16)*4)|((uint64_t)(4*j+b)<<56)); }
    AMX_CLR(); post(); };

  auto chk=[&](const char* nm,auto fn){ for(auto&x:C)x=0; fn();
    double num=0,den=0; for(size_t i=0;i<(size_t)M*N;++i){ double e=C[i]-Cref[i]; num+=e*e; den+=(double)Cref[i]*Cref[i]; }
    std::printf("%-18s rel-err=%.2e %s\n",nm,std::sqrt(num/den),std::sqrt(num/den)<1e-3?"OK":"BAD"); };
  chk("1-bank (old)",old1); chk("4-bank ILP",ilp4); chk("4-Ntile blocked",blk4); chk("4-Ntile+idx-amort",blk4i);

  auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<6;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
  double to=best(old1), tb=best(blk4), tbi=best(blk4i); double F=2.0*M*N*K;
  std::printf("M=%d K=%d N=%d single-thread:\n",M,K,N);
  std::printf("  1-bank (baseline)     : %.2f ms  (%.0f GFLOP/s)\n", to, F/(to/1e3)/1e9);
  std::printf("  4-Ntile blocked       : %.2f ms  (%.0f GFLOP/s)   %.2fx\n", tb, F/(tb/1e3)/1e9, to/tb);
  std::printf("  4-Ntile + idx-amort   : %.2f ms  (%.0f GFLOP/s)   %.2fx\n", tbi, F/(tbi/1e3)/1e9, to/tbi);
  double bestamx=std::min({to,tb,tbi});
  std::printf("  [fair ggml repacked M=16 = 2.02 ms]  best AMX %.2f ms vs ggml: %.2fx\n", bestamx, 2.02/bestamx);
  return 0;
}
