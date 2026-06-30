// Profile the NEON codebook kernel: where does the time go, and how close to ggml (267 Gop/s)
// can it get? Decompose into (1) dequant (vqtbl) and (2) the sdot dot loop, and test a 4x4
// register-blocked microkernel (reuse each A and W load across 4 outputs) vs the 1x8 version.
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
struct Rng{uint64_t s;uint32_t u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return s>>33;}};
static int M=64,K=2048,N=8192;
static std::vector<int8_t> A,CB; static std::vector<uint8_t> W; static std::vector<int32_t> C;
static std::vector<int8_t> WD;  // fully dequantized weights [N][K] for dot-only timing

static void dequant_all(){
  int8x16_t cb=vld1q_s8(CB.data());
  for(int n=0;n<N;++n){ const uint8_t* wp=W.data()+(size_t)n*(K/2); int8_t* wd=WD.data()+(size_t)n*K;
    for(int k=0;k<K;k+=32){ uint8x16_t v=vld1q_u8(wp+k/2);
      int8x16_t wl=vqtbl1q_s8(cb,vandq_u8(v,vdupq_n_u8(0x0F))), wh=vqtbl1q_s8(cb,vshrq_n_u8(v,4));
      vst1q_s8(wd+k,vzip1q_s8(wl,wh)); vst1q_s8(wd+k+16,vzip2q_s8(wl,wh)); } }
}
// dot only, 4x4 register-blocked: 4 M-rows x 4 N-cols, reuse each A/W load across the tile
static void dot_4x4(int n0,int n1){
  for(int n=n0;n<n1;n+=4){
    for(int m=0;m<M;m+=4){
      int32x4_t ac[4][4]; for(int i=0;i<4;++i)for(int j=0;j<4;++j) ac[i][j]=vdupq_n_s32(0);
      const int8_t* a0=A.data()+(size_t)m*K; const int8_t* w0=WD.data()+(size_t)n*K;
      for(int k=0;k<K;k+=16){
        int8x16_t a[4]={vld1q_s8(a0+k),vld1q_s8(a0+K+k),vld1q_s8(a0+2*K+k),vld1q_s8(a0+3*K+k)};
        int8x16_t w[4]={vld1q_s8(w0+k),vld1q_s8(w0+K+k),vld1q_s8(w0+2*K+k),vld1q_s8(w0+3*K+k)};
        for(int i=0;i<4;++i)for(int j=0;j<4;++j) ac[i][j]=vdotq_s32(ac[i][j],a[i],w[j]);
      }
      for(int i=0;i<4;++i)for(int j=0;j<4;++j) C[(size_t)(n+j)*M+(m+i)]=vaddvq_s32(ac[i][j]);
    }
  }
}
int main(){
  Rng r{0x5eedULL};
  A.resize((size_t)M*K); for(auto&x:A)x=(int8_t)(r.u()&0xFF);
  W.resize((size_t)N*(K/2)); for(auto&x:W)x=(uint8_t)(r.u()&0xFF);
  CB.resize(16); for(auto&x:CB)x=(int8_t)((int)(r.u()%240)-120);
  WD.resize((size_t)N*K); C.resize((size_t)N*M);
  double F=2.0*M*N*K;
  auto t=[&](auto f,int R){double b=1e30;for(int tr=0;tr<3;++tr){auto t0=clk::now();for(int i=0;i<R;++i)f();b=std::min(b,ms(clk::now()-t0)/R);}return b;};
  double td=t([&](){dequant_all();},4);
  double tk=t([&](){dot_4x4(0,N);},4);
  std::printf("dequant-all (vqtbl): %.2f ms\n",td);
  std::printf("dot-only 4x4 sdot  : %.2f ms (%.0f G-int8op/s, %.0f%% of ggml's 267)\n",tk,F/(tk/1e3)/1e9,100*(F/(tk/1e3)/1e9)/267);
  std::printf("full (dequant+dot) : %.2f ms ST\n",td+tk);
  std::printf("[1x8 version was 16.2 ms ST=132 Gop/s; ggml 8.03=267; Gope-level ~2.7ms=~800]\n");
  return 0;
}
