// Decode (M=1, single-token GEMV) for the NEON codebook kernel, to answer: do we still win/lose
// decode against an optimized NEON codebook (not just ggml)? C[n] = sum_k A[k]*cb[idx[k][n]].
// Memory-bound: dominated by reading the 4-bit weights once. NEON streams them with vqtbl+sdot.
// Compare to AMX at M=1: fp32 kernel 0.48 ms (measured M-curve); int8 kernel ~2.70 ms (its 64-wide
// tile processes 64 M-rows regardless, so M=1 wastes 63/64 -> same cost as M=64).
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
struct Rng{uint64_t s;uint32_t u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return s>>33;}};
static int K=2048,N=8192; static int8_t* A; static uint8_t* W; static int8_t* CB; static int32_t* C;
static void run(int n0,int n1){            // M=1 GEMV
  int8x16_t cb=vld1q_s8(CB);
  for(int n=n0;n<n1;++n){
    const uint8_t* wp=W+(size_t)n*(K/2); const int8_t* a=A;
    int32x4_t acc=vdupq_n_s32(0);
    for(int k=0;k<K;k+=32){
      uint8x16_t v=vld1q_u8(wp+k/2);
      int8x16_t wl=vqtbl1q_s8(cb,vandq_u8(v,vdupq_n_u8(0x0F))), wh=vqtbl1q_s8(cb,vshrq_n_u8(v,4));
      acc=vdotq_s32(acc,vld1q_s8(a+k),    vzip1q_s8(wl,wh));
      acc=vdotq_s32(acc,vld1q_s8(a+k+16), vzip2q_s8(wl,wh));
    }
    C[n]=vaddvq_s32(acc);
  }
}
static double best(int T){ const int R=20,per=(N+T-1)/T; double bb=1e30;
  for(int tr=0;tr<5;++tr){ auto t0=clk::now(); std::vector<std::thread> th;
    for(int i=0;i<T;++i){int a=i*per,b=std::min(N,a+per); if(a<b) th.emplace_back([=](){for(int r=0;r<R;++r) run(a,b);});}
    for(auto&x:th)x.join(); bb=std::min(bb,ms(clk::now()-t0)/R);} return bb; }
int main(){
  Rng r{0x5eedULL};
  std::vector<int8_t> a(K); for(auto&x:a)x=(int8_t)(r.u()&0xFF);
  std::vector<uint8_t> w((size_t)N*(K/2)); for(auto&x:w)x=(uint8_t)(r.u()&0xFF);
  std::vector<int8_t> cb(16); for(auto&x:cb)x=(int8_t)((int)(r.u()%240)-120);
  std::vector<int32_t> c(N); A=a.data();W=w.data();CB=cb.data();C=c.data();
  run(0,1);{int n=0;int32_t ref=0;for(int k=0;k<K;++k){uint8_t b=w[(size_t)n*(K/2)+k/2];int idx=(k&1)?(b>>4):(b&0xF);ref+=(int32_t)a[k]*cb[idx];}std::printf("correctness C[0]: %d vs %d %s\n",c[0],ref,c[0]==ref?"OK":"BAD");}
  std::printf("NEON codebook DECODE (M=1, K=%d N=%d):\n",K,N);
  for(int T:{1,4,8}){ double t=best(T); std::printf("  %d-thread: %.3f ms\n",T,t); }
  std::printf("[AMX at M=1: fp32 kernel 0.48 ms ; int8 kernel ~2.70 ms (64-wide tile wasted)]\n");
  std::printf("[ggml Q4 at M=1: 0.25 ms]\n");
  return 0;
}
