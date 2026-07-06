// Unified real-weights codebook-GEMM e2e harness (Paper 2, "better e2e test").
// Loads a REAL NF4-quantized linear layer exported by export_real_layers.py
// (unpacked idx[N][K], per-channel scale[N], a real M=16 activation At16 + PyTorch
// Cref16), and:
//   (1) BIT-EXACT: runs it through the actual AMX indexed-MATFP kernel and diffs
//       against an ORDER-MATCHED, fused (fmaf) scalar reference -> expect max-abs = 0.
//       This is the check the old harness lacked: it only had PyTorch (5e-7, different
//       reduction order). We report BOTH here.
//   (2) FRAMEWORK: diffs the same output against PyTorch's A.dequant(W) (Cref16) -> ~5e-7.
//   (3) ENVELOPE: sweeps batch M in {1,4,16,32,64,128,256} (the kernel is M-blocked, so
//       M>=128 runs) and reports our GFLOP/s + the implied speedup vs ggml's M-linear cost.
// Usage: amx_e2e_codebook <tag>   (reads /tmp/<tag>_{idx,scale,At16,Cref16,dims}.bin)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}

// indexed MATFP: X[0]=packed 4-bit indices, codebook in X register `src`, Y=activations.
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}
template<class T> static std::vector<T> rd(const std::string&f){std::ifstream s(f,std::ios::binary);s.seekg(0,std::ios::end);auto n=s.tellg();s.seekg(0);std::vector<T> v(n/sizeof(T));s.read((char*)v.data(),n);return v;}

// Pack unpacked idx[N][K] (uint8 0..15) into the AMX layout for a given NB (col-tiles
// sharing one index load): idxbuf[(blk*K+k)*(NB*8) + nt*8 + b], 2 indices per byte.
static std::vector<uint8_t> packIdx(const std::vector<uint8_t>&idx,int N,int K,int NB){
  int NBLK=N/(NB*16); std::vector<uint8_t> out((size_t)NBLK*K*(NB*8)+64,0);
  for(int blk=0;blk<NBLK;++blk)for(int k=0;k<K;++k)for(int nt=0;nt<NB;++nt){
    int n0=(blk*NB+nt)*16; size_t base=((size_t)(blk*K+k))*(NB*8)+nt*8;
    for(int b=0;b<8;++b) out[base+b]=(uint8_t)(idx[(size_t)(n0+2*b)*K+k] | (idx[(size_t)(n0+2*b+1)*K+k]<<4));
  } return out;
}

// Templated worker: compile-time (MT,NB) (paper: runtime-arg form is ~2x slower).
// Mstride = per-column row stride of At; rowoff = first global row of this M-block.
template<int MT,int NB>
static void worker(int N,int K,int M,int Mstride,int rowoff,uint64_t CBp,uint64_t IB,uint64_t AT,uint64_t ZB,float*C){
  int NBLK=N/(NB*16);
  AMX_SET(); AMX_LDX(CBp|(1ULL<<56));
  for(int blk=0;blk<NBLK;++blk){
    for(int bk=0;bk<MT*NB;++bk) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+bk)<<56));
    for(int k=0;k<K;++k){ AMX_LDX((IB+((size_t)(blk*K+k))*(NB*8))|(0ULL<<56));
      for(int mt=0;mt<MT;++mt){ AMX_LDY((AT+((size_t)k*Mstride+rowoff+mt*16)*4)|(0ULL<<56));
        for(int nt=0;nt<NB;++nt) AMX_MATFP(MatfpIdxX(mt*NB+nt,nt*8,0,1)); } }
    for(int mt=0;mt<MT;++mt) for(int nt=0;nt<NB;++nt){ int tile=blk*NB+nt,b=mt*NB+nt;
      for(int j=0;j<16;++j){ int m=rowoff+mt*16+j; if(m<M) AMX_STZ(((uint64_t)C+((size_t)m*N+tile*16)*4)|((uint64_t)(4*j+b)<<56)); } }
  }
  AMX_CLR();
}

// Run the full GEMM for batch M by choosing (MT,NB) and M-blocking past 64.
// Returns the chosen NB (so the caller packs indices to match). Writes C[M*N] (pre-scale).
static int runKernel(int N,int K,int M,int Mstride,uint64_t CBp,uint64_t IB_NB4,uint64_t IB_NB2,uint64_t IB_NB1,uint64_t AT,uint64_t ZB,float*C){
  if(M<=16){ worker<1,4>(N,K,M,Mstride,0,CBp,IB_NB4,AT,ZB,C); return 4; }
  if(M<=32){ worker<2,2>(N,K,M,Mstride,0,CBp,IB_NB2,AT,ZB,C); return 2; }
  for(int off=0;off<M;off+=64) worker<4,1>(N,K,M,Mstride,off,CBp,IB_NB1,AT,ZB,C);
  return 1;
}

int main(int argc,char**argv){
  if(argc<2){std::printf("usage: %s <tag>\n",argv[0]);return 1;}
  std::string tag=argv[1],pre="/tmp/"+tag+"_";
  auto dims=rd<int32_t>(pre+"dims.bin"); int N=dims[0],K=dims[1];
  auto idx=rd<uint8_t>(pre+"idx.bin");        // [N][K] unpacked
  auto scale=rd<float>(pre+"scale.bin");      // [N]
  auto At16=rd<float>(pre+"At16.bin");         // [K][16] (col-major, real activation)
  auto Cref16=rd<float>(pre+"Cref16.bin");     // [16][N] PyTorch A.dequant(W)
  std::printf("\n=== %s : N=%d K=%d ===\n",tag.c_str(),N,K);

  alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e]; alignas(64) float zb[16]={0};
  const uint64_t CBp=(uint64_t)cb,ZB=(uint64_t)zb;
  // pre-pack indices for each NB regime (one-time, untimed = weight pre-pack)
  auto p4=packIdx(idx,N,K,4),p2=packIdx(idx,N,K,2),p1=packIdx(idx,N,K,1);
  const uint64_t I4=(uint64_t)p4.data(),I2=(uint64_t)p2.data(),I1=(uint64_t)p1.data();

  // ---- (1)+(2) correctness at M=16 on the REAL activation ----
  std::vector<float> C((size_t)16*N,0);
  int nb=runKernel(N,K,16,16,CBp,I4,I2,I1,(uint64_t)At16.data(),ZB,C.data()); (void)nb;
  for(int m=0;m<16;++m)for(int n=0;n<N;++n) C[(size_t)m*N+n]*=scale[n];
  // order-matched fused scalar reference (same k-order, fmaf to match AMX fused MAC)
  double bitmax=0; for(int m=0;m<16;++m)for(int n=0;n<N;++n){ float acc=0.f;
    for(int k=0;k<K;++k) acc=std::fma(At16[(size_t)k*16+m],NF4[idx[(size_t)n*K+k]],acc);
    acc*=scale[n]; bitmax=std::max(bitmax,(double)std::fabs(C[(size_t)m*N+n]-acc)); }
  double num=0,den=0,pymax=0; for(int m=0;m<16;++m)for(int n=0;n<N;++n){ double e=C[(size_t)m*N+n]-Cref16[(size_t)m*N+n];
    num+=e*e; den+=(double)Cref16[(size_t)m*N+n]*Cref16[(size_t)m*N+n]; pymax=std::max(pymax,std::fabs(e)); }
  std::printf("  bit-exact vs order-matched ref : max-abs = %.2e  %s\n",bitmax, bitmax==0.0?"BIT-EXACT":"(not 0)");
  std::printf("  framework agreement vs PyTorch : rel = %.2e  max-abs = %.2e\n",std::sqrt(num/den),pymax);

  // ---- (3) batch-size envelope: our GFLOP/s + speedup via ggml M-linear model ----
  // ggml Q4 cost is M-linear (M independent dot products); fit c from measured M=64.
  const double ggml64_ms=8.03, ggmlK=2048, ggmlN=8192;        // measured fair ggml @ K2048 N8192 M64
  double ggml_ms_per_tok = (ggml64_ms/64.0) * ((double)N*K)/(ggmlN*ggmlK);  // scale to this shape, per token
  const int Ms[]={1,4,16,32,64,128,256};
  std::printf("  %-5s %9s %9s %9s %7s\n","M","our ms","GFLOP/s","vs ggml","CV");
  for(int M:Ms){ int Mpad=((M+63)/64)*64; Rng r{0x55ULL+M};
    std::vector<float> At((size_t)K*Mpad,0.f); for(int k=0;k<K;++k)for(int m=0;m<M;++m) At[(size_t)k*Mpad+m]=(float)nrm(r);
    std::vector<float> Co((size_t)M*N,0); const uint64_t ATp=(uint64_t)At.data();
    auto run=[&](){ runKernel(N,K,M,Mpad,CBp,I4,I2,I1,ATp,ZB,Co.data()); };
    run();run(); std::vector<double> reps; const int R=(M>=128?20:40);   // median + CV over 11 reps
    for(int t=0;t<11;++t){auto t0=clk::now();for(int i=0;i<R;++i)run();reps.push_back(ms(clk::now()-t0)/R);}
    std::sort(reps.begin(),reps.end()); double med=reps[5];
    double mean=0; for(double x:reps)mean+=x; mean/=11; double var=0; for(double x:reps)var+=(x-mean)*(x-mean);
    double cv=std::sqrt(var/11)/mean*100;
    double gf=2.0*M*N*K/(med/1e3)/1e9, ggml_ms=ggml_ms_per_tok*M;
    std::printf("  %-5d %9.3f %9.0f %8.2fx %6.1f%%\n",M,med,gf,ggml_ms/med,cv);
  }
  return 0;
}
