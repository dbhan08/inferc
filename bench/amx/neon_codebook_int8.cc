// Fair NEON-SIMD codebook baseline (the Gope et al. approach: vtbl table-lookup dequant + sdot),
// to compare apples-to-apples against our AMX MATINT codebook kernel at matched int8 precision.
// Gope's actual kernel is unreleased; this is our best-effort reimplementation. We validate it
// against ggml's Q4: if it lands near Gope's reported ~3x over llama.cpp it's a faithful baseline;
// if it only ties ggml it is under-optimized and the head-to-head is inconclusive (we say so).
//
// Kernel: 16-entry int8 codebook in a NEON reg; weights are 4-bit indices (2/byte). Per output
// column n: dequant W[:,n] once via vqtbl1q_s8 (+vzip to natural k-order), reuse across all M
// rows; accumulate C[m,n] over K with vdotq_s32 (sdot). Threaded over N. M1 has FEAT_DotProd.
//
// RESULT (M1, M=64 K=2048 N=8192): ST 16.2 ms / MT(8) 4.30 ms. The dot loop runs at ~65% of
// NEON sdot peak (0.77 cyc/vdotq), so the kernel is not broken -- but at 132 G-int8op/s it is
// HALF of ggml's repacked Q4 (267) and ~6x slower than Gope et al.'s reported ~3x-over-llama.cpp
// level. A hand-written codebook microkernel cannot match Arm's production tuning in this scope.
// CONCLUSION: this is NOT a fair baseline -- comparing our AMX kernel (ST 2.70 / MT 1.56 ms) to
// it would be a STRAWMAN favoring AMX. So we do NOT use it as a paper baseline. We keep ggml's
// production Q4 as the fair NEON baseline (which we beat ~3x ST), and we reason the AMX-vs-Gope
// outcome from published numbers: Gope ~3x over ggml => ~2.7 ms ST / ~0.69 ms MT, i.e. a likely
// single-thread TIE and a multi-thread LOSS (AMX's 2-block cap vs NEON's 8 cores). Honest and
// inconclusive on a direct head-to-head; the paper says exactly this (Section 6).
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
struct Rng{uint64_t s;uint32_t u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return s>>33;}};

static int M,K,N; static int8_t* A; static uint8_t* W; static int8_t* CB; static int32_t* C;
// A: [M][K] int8 row-major. W: [N][K/2] packed 4-bit (byte b = idx[2b] | idx[2b+1]<<4). C: [N][M].
// Blocked over N (NB cols): dequant NB weight columns to L1, then per m load each A vector ONCE
// and sdot against all NB columns -> A reused NB-fold (cuts A memory traffic by NB).
static constexpr int NB=8;
static void deq(const uint8_t* wp,int8_t* wbuf,int8x16_t cb){
  for(int k=0;k<K;k+=32){
    uint8x16_t v=vld1q_u8(wp+k/2);
    uint8x16_t lo=vandq_u8(v,vdupq_n_u8(0x0F)), hi=vshrq_n_u8(v,4);
    int8x16_t wl=vqtbl1q_s8(cb,lo), wh=vqtbl1q_s8(cb,hi);
    vst1q_s8(wbuf+k,    vzip1q_s8(wl,wh));
    vst1q_s8(wbuf+k+16, vzip2q_s8(wl,wh));
  }
}
static void run(int n0,int n1){
  int8x16_t cb=vld1q_s8(CB);
  std::vector<int8_t> wbuf((size_t)NB*K);
  for(int n=n0;n<n1;n+=NB){
    int nb=std::min(NB,n1-n);
    for(int j=0;j<nb;++j) deq(W+(size_t)(n+j)*(K/2), wbuf.data()+(size_t)j*K, cb);
    for(int m=0;m<M;++m){
      const int8_t* a=A+(size_t)m*K;
      int32x4_t acc[NB]; for(int j=0;j<NB;++j) acc[j]=vdupq_n_s32(0);
      for(int k=0;k<K;k+=16){
        int8x16_t av=vld1q_s8(a+k);
        for(int j=0;j<nb;++j) acc[j]=vdotq_s32(acc[j],av,vld1q_s8(wbuf.data()+(size_t)j*K+k));
      }
      for(int j=0;j<nb;++j) C[(size_t)(n+j)*M+m]=vaddvq_s32(acc[j]);
    }
  }
}
static double best(int T){ const int R=4,per=(N+T-1)/T; double bb=1e30;
  for(int tr=0;tr<3;++tr){ auto t0=clk::now(); std::vector<std::thread> th;
    for(int i=0;i<T;++i){int a=i*per,b=std::min(N,a+per); if(a<b) th.emplace_back([=](){for(int r=0;r<R;++r) run(a,b);});}
    for(auto&x:th)x.join(); bb=std::min(bb,ms(clk::now()-t0)/R);} return bb; }

int main(){
  M=64;K=2048;N=8192; Rng r{0x5eedULL};
  std::vector<int8_t> a((size_t)M*K); for(auto&x:a)x=(int8_t)(r.u()&0xFF);
  std::vector<uint8_t> w((size_t)N*(K/2)); for(auto&x:w)x=(uint8_t)(r.u()&0xFF);
  std::vector<int8_t> cb(16); for(auto&x:cb)x=(int8_t)((int)(r.u()%240)-120);
  std::vector<int32_t> c((size_t)N*M);
  A=a.data();W=w.data();CB=cb.data();C=c.data();
  // correctness: check one (n,m) against a scalar reference (guard against a no-op kernel)
  run(0,1); { int n=0,m=3; int32_t ref=0; for(int k=0;k<K;++k){ uint8_t b=w[(size_t)n*(K/2)+k/2]; int idx=(k&1)?(b>>4):(b&0xF); ref+=(int32_t)a[(size_t)m*K+k]*cb[idx]; }
    std::printf("correctness C[%d,%d]: kernel=%d ref=%d %s\n",n,m,c[(size_t)n*M+m],ref,c[(size_t)n*M+m]==ref?"OK":"MISMATCH"); }
  double F=2.0*M*N*K;
  std::printf("NEON codebook int8 GEMM M=%d K=%d N=%d:\n",M,K,N);
  for(int T:{1,2,4,8}){ double t=best(T); std::printf("  %d-thread: %.2f ms (%.0f G-int8op/s)\n",T,t,F/(t/1e3)/1e9); }
  std::printf("[ref M=64: ggml Q4 ST 8.03 / MT 2.06 ; our AMX int8 ST 2.70 / MT 1.56 ms]\n");
  std::printf("[validation: NEON codebook ST near 8.03/3=2.7ms would match Gope's ~3x over llama.cpp]\n");
  return 0;
}
