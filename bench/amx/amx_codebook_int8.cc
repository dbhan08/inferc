// int8 codebook GEMM (apples-to-apples with ggml's int8). MATINT int8xint8->int32, indexed
// gather of int8 codebook. 64x16 tile: C[m][n] at Z[4n+m%4][m/4], M=64 (X), N=16 (Y gathered),
// index for N=n at nibble 4n. C[m][n]=sum_k A[m,k]*cb[idx[k,n]] (int32), then *sA[m]*sW[n].
// Verify correctness vs reference; then throughput vs fp32/ggml (separately).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatintIdxY(int z,int xo,int yo,int src){
  return (10ULL<<42)|(1ULL<<53)|(1ULL<<54)|(1ULL<<48)|(1ULL<<47)|(1ULL<<63)|(1ULL<<26)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};

int main(){
  const int M=64,K=256,N=16;                       // one 64x16 tile, K accumulation
  Rng r{0x77ULL};
  std::vector<int8_t> cb(16); for(int e=0;e<16;++e) cb[e]=(int8_t)((int)(r.u()*240)-120);   // int8 codebook
  std::vector<int8_t> At((size_t)K*64,0);          // A transposed [k][64 m], int8
  for(int k=0;k<K;++k) for(int m=0;m<M;++m) At[(size_t)k*64+m]=(int8_t)((int)(r.u()*240)-120);
  std::vector<uint8_t> idx((size_t)K*N);            // idx[k][n] 4-bit
  for(int k=0;k<K;++k) for(int n=0;n<N;++n) idx[(size_t)k*N+n]=(uint8_t)(r.u()*16)&0xF;
  // packed index buffers: per k, 64 bytes, byte 2n = idx[k][n] (nibble 4n)
  std::vector<uint8_t> idxp((size_t)K*64,0);
  for(int k=0;k<K;++k) for(int n=0;n<N;++n) idxp[(size_t)k*64+2*n]=idx[(size_t)k*N+n];
  // reference: C[m][n] = sum_k A[m,k]*cb[idx[k,n]]  (int32)
  std::vector<int32_t> Cref((size_t)M*N,0);
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){ int32_t s=0; for(int k=0;k<K;++k) s+=(int32_t)At[(size_t)k*64+m]*cb[idx[(size_t)k*N+n]]; Cref[(size_t)m*N+n]=s; }

  alignas(64) int8_t cbpad[64]={0}; std::memcpy(cbpad,cb.data(),16);
  alignas(64) int32_t zb[16]={0};
  int32_t Zbuf[64*16];
  const uint64_t CB=(uint64_t)cbpad, IX=(uint64_t)idxp.data(), AT=(uint64_t)At.data(), ZB=(uint64_t)zb;
  AMX_SET();
  AMX_LDY(CB|(1ULL<<56));                            // Y[1] = int8 codebook (gather table)
  for(int r2=0;r2<64;++r2) AMX_LDZ(ZB|((uint64_t)r2<<56));   // zero Z (zb is 64 bytes of 0)
  for(int k=0;k<K;++k){
    AMX_LDX((AT+(size_t)k*64)|(0ULL<<56));           // X[0] = A[k] (64 int8 M)
    AMX_LDY((IX+(size_t)k*64)|(0ULL<<56));           // Y[0] = packed indices (stride-4)
    AMX_MATINT(MatintIdxY(0,0,0,1));                 // Z[m][n] += A[m]*cb[idx[n]]
  }
  for(int r2=0;r2<64;++r2){ AMX_STZ(ZB|((uint64_t)r2<<56)); std::memcpy(Zbuf+r2*16,zb,64); }
  AMX_CLR();
  // extract C[m][n] = Z[4n + m%4][m/4]; compare to Cref
  int bad=0; for(int m=0;m<M;++m) for(int n=0;n<N;++n){ int32_t got=Zbuf[(4*n+(m%4))*16 + (m/4)]; int32_t exp=Cref[(size_t)m*N+n];
    if(got!=exp){ if(bad<6) std::printf("  C[%d,%d] got=%d exp=%d\n",m,n,got,exp); ++bad; } }
  std::printf("int8 codebook GEMM M=%d K=%d N=%d: %s (%d mismatches)\n",M,K,N, bad==0?"CORRECT (bit-exact int)":"MISMATCH",bad);
  return bad?1:0;
}
