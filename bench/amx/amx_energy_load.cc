// Energy / perf-per-watt load generator: runs ONE engine (AMX codebook MATFP, or a NEON
// sdot codebook) in a sustained fixed-duration loop at K=2048 N=8192 M=64, printing GFLOP/s.
// Wrap the invocation with powermetrics to get average package power, then
// perf/watt = GFLOP/s / watts. Run each arm at its best thread count.
//   sudo powermetrics --samplers cpu_power -i 1000 &   # in another shell
//   ./amx_energy_load amx  8        # 8 P+E threads, 8 s
//   ./amx_energy_load neon 8
// (Apple AMX has one block per cluster, so >2 threads on AMX mostly contends; NEON scales to 8.)
#include <algorithm>
#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "amx/aarch64.h"
using clk=std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double,std::milli>(d).count()/1e3;}
static inline uint64_t MatfpIdxX(int z,int xo,int yo,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
static const float NF4[16]={-1.f,-0.6961928f,-0.52507305f,-0.39491749f,-0.28444138f,-0.18477343f,-0.09105004f,0.f,
                            0.0795803f,0.1609302f,0.2461123f,0.33791524f,0.44070983f,0.562617f,0.72295684f,1.f};
static const int K=2048,N=8192,M=64;

// one AMX codebook-GEMM pass (MT=4,NB=1 tiling, indices pre-packed), N split across threads
static void amx_pass(int n0,int n1,const uint8_t* idx,const float* At,const float* cb,float* C){
  alignas(64) float zb[16]={0};
  AMX_SET(); AMX_LDX(((uint64_t)cb)|(1ULL<<56));
  for(int t=n0;t<n1;++t){
    for(int bk=0;bk<4;++bk) for(int j=0;j<16;++j) AMX_LDZ(((uint64_t)zb)|((uint64_t)(4*j+bk)<<56));
    for(int k=0;k<K;++k){ AMX_LDX(((uint64_t)idx+((size_t)(t*K+k))*8)|(0ULL<<56));
      for(int mt=0;mt<4;++mt){ AMX_LDY(((uint64_t)At+((size_t)k*64+mt*16)*4)|(0ULL<<56));
        AMX_MATFP(MatfpIdxX(mt,0,0,1)); } }
    for(int mt=0;mt<4;++mt) for(int j=0;j<16;++j) AMX_STZ(((uint64_t)C+((size_t)(mt*16+j)*N+t*16)*4)|((uint64_t)(4*j+mt)<<56));
  }
  AMX_CLR();
}
// representative NEON int8 codebook microkernel: vqtbl gather hoisted across a 4-row block,
// sdot accumulate (matches the Section 5.4 4x4-blocked baseline). Does the full M*N*K work.
static void neon_pass(int n0,int n1,const uint8_t* idx,const int8_t* Aq,const int8_t* cb,int32_t* C){
  int8x16_t cbv=vld1q_s8(cb);
  for(int t=n0;t<n1;++t) for(int nn=0;nn<16;++nn){ int n=t*16+nn;     // 16 cols/tile, matches AMX coverage
    for(int m0=0;m0<M;m0+=4){ int32x4_t a0=vdupq_n_s32(0),a1=a0,a2=a0,a3=a0;
      for(int k=0;k<K;k+=16){ int8x16_t w=vqtbl1q_s8(cbv,vld1q_u8(idx+(size_t)n*K+k));  // gather once / 4 rows
        a0=vdotq_s32(a0,w,vld1q_s8(Aq+(size_t)(m0+0)*K+k));
        a1=vdotq_s32(a1,w,vld1q_s8(Aq+(size_t)(m0+1)*K+k));
        a2=vdotq_s32(a2,w,vld1q_s8(Aq+(size_t)(m0+2)*K+k));
        a3=vdotq_s32(a3,w,vld1q_s8(Aq+(size_t)(m0+3)*K+k)); }
      C[(size_t)(m0+0)*N+n]=vaddvq_s32(a0); C[(size_t)(m0+1)*N+n]=vaddvq_s32(a1);
      C[(size_t)(m0+2)*N+n]=vaddvq_s32(a2); C[(size_t)(m0+3)*N+n]=vaddvq_s32(a3);
    }
  }
}

int main(int argc,char**argv){
  std::string mode=argc>1?argv[1]:"amx"; int T=argc>2?atoi(argv[2]):8; double dur=argc>3?atof(argv[3]):8.0;
  int NT=N/16;
  std::vector<uint8_t> idx((size_t)N*K+64,3); for(size_t i=0;i<idx.size();++i) idx[i]=(uint8_t)(i*7); // fits AMX-packed + NEON-unpacked layouts
  std::vector<float> At((size_t)K*64,0.1f), C((size_t)M*N,0), cb(16); for(int e=0;e<16;++e) cb[e]=NF4[e];
  std::vector<int8_t> Aq((size_t)M*K,1), cbi(16); for(int e=0;e<16;++e) cbi[e]=(int8_t)(e*8-64);
  std::vector<int32_t> Ci((size_t)M*N,0);
  std::atomic<long> passes{0}; std::atomic<bool> stop{false};
  auto t0=clk::now();
  std::vector<std::thread> th;
  for(int i=0;i<T;++i) th.emplace_back([&,i](){
    int per=(NT+T-1)/T, a=i*per, b=std::min(NT,a+per);
    while(!stop.load()){
      if(mode=="amx") amx_pass(a,b,idx.data(),At.data(),cb.data(),C.data());
      else            neon_pass(a,b,idx.data(),Aq.data(),cbi.data(),Ci.data());
      passes.fetch_add(1,std::memory_order_relaxed);
    }
  });
  std::this_thread::sleep_for(std::chrono::duration<double>(dur));
  stop.store(true); for(auto&x:th)x.join();
  double el=sec(clk::now()-t0); long p=passes.load();
  double flop=(double)p*2.0*M*N*K/T;   // each pass does N/T cols x M x K MACs
  std::printf("mode=%s threads=%d  %.2fs  %ld passes  %.0f GFLOP/s  (wrap with powermetrics for watts)\n",
              mode.c_str(),T,el,p,flop/el/1e9);
  return 0;
}
