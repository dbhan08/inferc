// Non-uniform CODEBOOK GEMM via the verified free-fused-gather mechanism (indexed
// MATFP). C[M,N] = A[M,K] . B[K,N], where B is quantized to a per-group ARBITRARY
// 16-entry fp32 codebook (4-bit indices) -- the NF4/vector-quant case where dequant
// REQUIRES a lookup (no cheap arithmetic), so the fused free gather is the unlock.
// Per k-step: weight indices -> X operand, codebook -> table reg, A-col -> Y; indexed
// MATFP does z[m][n] += A[m,k] * codebook[idx[k,n]] with dequant FUSED. Verify vs an
// fp32 reference GEMM on the dequantized weights; time vs a plain fp32 MATFP GEMM.
// M=N=16 tile, K=64, codebook reloaded per 32-K group (LDX, cheap).

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
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
static inline uint64_t Matfp(int z,int xo,int yo){return (4ULL<<42)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}

int main(){
  const int M=16,N=16,K=64,G=32,NG=K/G;            // 2 groups of 32 K-rows
  Rng r{0x1234CABLL};
  // At[K][16] = A transposed so a K-row (= A column) is contiguous for LDY
  std::vector<float> At(K*M); for(auto&x:At) x=(float)nrm(r);
  // per-group NON-UNIFORM codebook: 16 arbitrary centroids (sorted random) per group
  std::vector<float> cb(NG*16 + 16, 0.f);
  for(int g=0;g<NG;++g){ for(int e=0;e<16;++e) cb[g*16+e]=(float)nrm(r)*1.3f; std::sort(&cb[g*16],&cb[g*16+16]); }
  // weights as 4-bit indices idx[k][n]; dequant Bdq[k][n] = cb[group(k)][idx[k][n]]
  std::vector<uint8_t> idxp(K*8 + 64, 0);          // per k: 16 nibbles in 8 bytes (+64 pad: LDX reads 64B)
  std::vector<float> Bdq(K*N);
  for(int k=0;k<K;++k){ int g=k/G; for(int n=0;n<N;++n){ int e=(int)(r.u()*16); if(e>15)e=15;
      idxp[k*8 + n/2] |= (uint8_t)(e&0xF)<<((n&1)*4); Bdq[k*N+n]=cb[g*16+e]; } }

  // fp32 reference: C_ref = A . Bdq
  std::vector<double> Cref(M*N,0);
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){ double s=0; for(int k=0;k<K;++k) s+=(double)At[k*M+m]*Bdq[k*N+n]; Cref[m*N+n]=s; }

  // AMX codebook GEMM (indexed MATFP)
  alignas(64) float zbuf[16]={0}; std::vector<float> Cout(M*N);
  const uint64_t AT=(uint64_t)At.data(), CB=(uint64_t)cb.data(), IX=(uint64_t)idxp.data(), ZB=(uint64_t)zbuf;
  AMX_SET();
  for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j)<<56));        // zero Z banks (fp32 row j at row 4*j)
  for(int k=0;k<K;++k){ int g=k/G;
    if(k%G==0) AMX_LDX((CB+(uint64_t)g*16*4)|(1ULL<<56));         // load group codebook -> X[1]
    AMX_LDX((IX+(uint64_t)k*8)|(0ULL<<56));                       // weight indices -> X[0]
    AMX_LDY((AT+(uint64_t)k*M*4)|(0ULL<<56));                     // A column k -> Y[0]
    AMX_MATFP(MatfpIdxX(0,0,0,1));                                // z[m][n] += A[m,k]*cb[idx[k][n]]
  }
  for(int j=0;j<16;++j) AMX_STZ(((uint64_t)Cout.data()+(uint64_t)j*N*4)|((uint64_t)(4*j)<<56)); // fp32: row j at Z row 4*j
  // (keep AMX enabled — timing loops below reuse it; CLR once at the very end)

  // NOTE: STZ writes Z row j (64B=16 fp32) to the given address. z[j][i]=C[m=j][n=i].
  double maxrel=0; int bad=0;
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){ double e=std::fabs(Cout[m*N+n]-Cref[m*N+n]); double d=std::fabs(Cref[m*N+n]);
    double rel=d>1e-6?e/d:e; if(rel>maxrel)maxrel=rel; if(rel>1e-4){++bad; if(bad<=4)std::printf("  C[%d,%d] amx=%.4f ref=%.4f\n",m,n,Cout[m*N+n],Cref[m*N+n]);} }
  std::printf("codebook GEMM (indexed MATFP) vs fp32 ref: max-rel=%.2e  %s\n", maxrel,
              bad==0?"CORRECT":"MISMATCH");

  // timing: codebook (indexed) vs plain fp32 MATFP GEMM, same shape
  const int REP=200000;
  auto t0=clk::now();
  for(int r2=0;r2<REP;++r2){ for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)j<<56));
    for(int k=0;k<K;++k){ if(k%G==0) AMX_LDX((CB+(uint64_t)(k/G)*16*4)|(1ULL<<56));
      AMX_LDX((IX+(uint64_t)k*8)|(0ULL<<56)); AMX_LDY((AT+(uint64_t)k*M*4)|(0ULL<<56)); AMX_MATFP(MatfpIdxX(0,0,0,1)); } }
  double tcb=sec(clk::now()-t0);
  auto t1=clk::now();
  for(int r2=0;r2<REP;++r2){ for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)j<<56));
    for(int k=0;k<K;++k){ AMX_LDX((AT+(uint64_t)k*M*4)|(0ULL<<56)); AMX_LDY((AT+(uint64_t)k*M*4)|(0ULL<<56)); AMX_MATFP(Matfp(0,0,0)); } }
  double tf=sec(clk::now()-t1);
  AMX_CLR();
  double flop=(double)REP*K*M*N*2;
  std::printf("codebook(int4+cb) GEMM: %.0f GFLOPS   plain fp32 MATFP: %.0f GFLOPS   ratio %.2fx\n",
              flop/tcb/1e9, flop/tf/1e9, tf/tcb);
  std::printf("(weights: 4-bit idx + 16-fp32 codebook/group = ~8x less weight memory than fp32, dequant fused/free)\n");
  return bad?1:0;
}
