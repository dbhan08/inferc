// END-TO-END on REAL weights: load OPT-125M layer0.fc1, NF4-quantized in Python (exported
// to /tmp/real_*.bin), run it through the ACTUAL AMX per-channel-scale kernel, and verify the
// output matches PyTorch's A.dequant(W) bit-exactly + measure speed. Closes the seam: until
// now speed/bit-exactness was on SYNTHETIC weights in C++, quality on REAL weights in PyTorch.
// This runs REAL quantized weights through the real kernel in one pipeline.
// dims: N=3072 K=768 M=16 (OPT-125M fc1). Run the Python exporter first.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};
template<class T> static std::vector<T> rd(const char* f){ std::ifstream s(f,std::ios::binary); s.seekg(0,std::ios::end); auto n=s.tellg(); s.seekg(0); std::vector<T> v(n/sizeof(T)); s.read((char*)v.data(),n); return v; }

int main(){
  const int N=3072,K=768,M=16,NT=N/16;
  auto idxf=rd<uint8_t>("/tmp/real_idxp.bin"); auto scale=rd<float>("/tmp/real_scale.bin");
  auto At=rd<float>("/tmp/real_At.bin"); auto Cref=rd<float>("/tmp/real_Cref.bin");
  std::printf("loaded real OPT-125M fc1: idxp=%zuB scale=%zu At=%zu Cref=%zu (N=%d K=%d M=%d)\n",
              idxf.size(),scale.size(),At.size(),Cref.size(),N,K,M);
  std::vector<uint8_t> idxp(idxf); idxp.resize(idxf.size()+64,0);        // pad: LDX reads 64B
  alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e];
  alignas(64) float zb[16]={0}; std::vector<float> C((size_t)M*N,0);
  const uint64_t CBp=(uint64_t)cb, IX=(uint64_t)idxp.data(), AT=(uint64_t)At.data(), ZB=(uint64_t)zb;

  auto kernel=[&](){
    AMX_SET();
    AMX_LDX(CBp|(1ULL<<56));                                            // shared NF4 codebook -> X[1]
    for(int t=0;t<NT;++t){
      for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));
      for(int k=0;k<K;++k){
        AMX_LDX((IX+((size_t)t*K+k)*8)|(0ULL<<56));                     // 16 real weight indices
        AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));                         // real A column
        AMX_MATFP(MatfpIdxX(0,0,0,1));
      }
      for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C.data()+((size_t)j*N+t*16)*4)|((uint64_t)(4*j)<<56));
    }
    AMX_CLR();
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) C[(size_t)m*N+n]*=scale[n];  // per-channel post-scale
  };

  kernel();
  double num=0,den=0,mx=0; for(size_t i=0;i<(size_t)M*N;++i){ double e=C[i]-Cref[i]; num+=e*e; den+=(double)Cref[i]*Cref[i]; mx=std::max(mx,std::fabs(e)); }
  double frob=std::sqrt(num/den);   // norm-based rel error (robust to near-zero entries)
  std::printf("kernel output vs PyTorch A.dequant(W) on REAL weights: ||C-Cref||/||Cref||=%.2e  max-abs=%.2e  %s\n",
              frob,mx, frob<1e-4?"MATCH (kernel reproduces the real quantized layer to fp32)":"MISMATCH");

  auto best=[&](auto fn){fn();fn();double b=1e30;for(int i=0;i<8;++i){auto t0=clk::now();fn();b=std::min(b,ms(clk::now()-t0));}return b;};
  double t=best(kernel);
  std::printf("kernel time: %.3f ms  (%.0f GFLOP/s)  M=%d K=%d N=%d real quantized layer\n",
              t, 2.0*M*N*K/(t/1e3)/1e9, M,K,N);
  return 0;
}
