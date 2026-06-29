// Multi-threaded optimized kernel (4-N-tile blocked + idx-amortized) vs ggml MT.
// Parallelize over N-tile blocks across threads (each: own AMX state, disjoint C cols).
// M1 has ~2 AMX blocks (P+E cluster) so AMX MT scaling is capped -- measure how far it
// goes and compare to ggml's 8-core NEON MT. Fair MT-vs-MT (our earlier "ST beats ggml MT"
// flattered us). M=16 K=2048 N=8192.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
struct Rng{uint64_t s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};
static int K,N,M,NT; static float *Atp,*Cp,*cbp; static uint8_t* ibp;
static const uint64_t Zrow=0;

static void worker(int blk0,int blk1){
  alignas(64) float zb[16]={0}; const uint64_t CB=(uint64_t)cbp,IB=(uint64_t)ibp,AT=(uint64_t)Atp,ZB=(uint64_t)zb;
  AMX_SET(); AMX_LDX(CB|(1ULL<<56));
  for(int blk=blk0;blk<blk1;++blk){
    for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_LDZ(ZB|((uint64_t)(4*j+b)<<56));
    for(int k=0;k<K;++k){ AMX_LDX((IB+((size_t)(blk*K+k))*32)|(0ULL<<56)); AMX_LDY((AT+(size_t)k*M*4)|(0ULL<<56));
      for(int b=0;b<4;++b) AMX_MATFP(MatfpIdxX(b,b*8,0,1)); }
    for(int b=0;b<4;++b) for(int j=0;j<16;++j) AMX_STZ(((uint64_t)Cp+((size_t)j*N+(4*blk+b)*16)*4)|((uint64_t)(4*j+b)<<56));
  }
  AMX_CLR();
}

int main(){
  K=2048;N=8192;M=16;NT=N/16; Rng r{0x55ULL};
  std::vector<float> At((size_t)K*M); for(auto&x:At) x=(float)nrm(r);
  std::vector<float> scale(N); for(auto&x:scale) x=0.2f+0.8f*(float)r.u();
  alignas(64) float cb[16]; for(int e=0;e<16;++e) cb[e]=NF4[e];
  std::vector<uint8_t> idxblk((size_t)(NT/4)*K*32+64,0);
  Rng rr{0x66ULL};
  for(size_t i=0;i<(size_t)(NT/4)*K*32;++i) idxblk[i]=(uint8_t)(rr.u()*256);   // random valid 4-bit-packed
  std::vector<float> C((size_t)M*N,0);
  Atp=At.data();Cp=C.data();cbp=cb;ibp=idxblk.data();

  // amortize thread-spawn: spawn T threads ONCE, each does R kernel passes, time/R.
  auto best=[&](int T){ const int R=40; int nb=NT/4, per=(nb+T-1)/T; double bb=1e30;
    for(int trial=0;trial<3;++trial){ auto t0=clk::now();
      std::vector<std::thread> th;
      for(int i=0;i<T;++i){ int a=i*per,b=std::min(nb,a+per); if(a<b) th.emplace_back([a,b,R](){ for(int r=0;r<R;++r) worker(a,b); }); }
      for(auto&x:th) x.join(); bb=std::min(bb, ms(clk::now()-t0)/R); }
    return bb; };

  std::printf("optimized kernel MT (M=%d K=%d N=%d), vs ggml [1T 2.02ms, 8T 0.68ms]:\n",M,K,N);
  double F=2.0*M*N*K;
  for(int T:{1,2,4,8}){ double t=best(T); std::printf("  %d-thread: %.3f ms  (%.0f GFLOP/s)  vs ggml-8T 0.68: %.2fx\n",T,t,F/(t/1e3)/1e9,0.68/t); }
  return 0;
}
