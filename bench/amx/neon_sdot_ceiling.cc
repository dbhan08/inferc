// What is the M1's single-core sdot throughput ceiling, and how close is our NEON codebook
// kernel to it? If we're near the ceiling, NO single-thread SIMD kernel (Gope's included) can
// beat us by much ST -> our AMX 2.4x ST win is robust vs ANY NEON kernel, and Gope's 3x must be
// a multi-thread (core-count) effect, not a better single-thread kernel.
// Pure-issue ceiling: many INDEPENDENT vdotq on register-resident operands (no loads, no deps).
#include <arm_neon.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
using clk=std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
int main(){
  const double GHZ=3.2;
  int8x16_t a=vdupq_n_s8(3), b=vdupq_n_s8(2);
  // 12 independent accumulators to saturate issue and hide sdot latency
  int32x4_t z0=vdupq_n_s32(0),z1=z0,z2=z0,z3=z0,z4=z0,z5=z0,z6=z0,z7=z0,z8=z0,z9=z0,zA=z0,zB=z0;
  const int64_t IT=50'000'000;
  auto t0=clk::now();
  for(int64_t i=0;i<IT;++i){
    #define BLK z0=vdotq_s32(z0,a,b); z1=vdotq_s32(z1,a,b); z2=vdotq_s32(z2,a,b); z3=vdotq_s32(z3,a,b); \
                z4=vdotq_s32(z4,a,b); z5=vdotq_s32(z5,a,b); z6=vdotq_s32(z6,a,b); z7=vdotq_s32(z7,a,b); \
                z8=vdotq_s32(z8,a,b); z9=vdotq_s32(z9,a,b); zA=vdotq_s32(zA,a,b); zB=vdotq_s32(zB,a,b);
    BLK BLK BLK BLK   // 48 vdotq/iter -> branch overhead amortized 4x
  }
  double s=sec(clk::now()-t0);
  // sink so nothing is optimized away
  int32x4_t acc=vaddq_s32(vaddq_s32(vaddq_s32(z0,z1),vaddq_s32(z2,z3)),
               vaddq_s32(vaddq_s32(z4,z5),vaddq_s32(z6,z7)));
  acc=vaddq_s32(acc,vaddq_s32(vaddq_s32(z8,z9),vaddq_s32(zA,zB)));
  volatile int sink=vaddvq_s32(acc); (void)sink;
  double nvdot=(double)IT*48;
  double vpc=nvdot/(s*GHZ*1e9);        // vdotq per cycle
  double gops=nvdot*32.0/s/1e9;        // 16 MAC = 32 ops per vdotq
  std::printf("pure sdot ceiling: %.2f vdotq/cycle, %.0f G-int8op/s (single-thread)\n",vpc,gops);
  std::printf("our NEON codebook dot-only: ~347 Gop/s  -> %.0f%% of ceiling\n",100*347/gops);
  std::printf("our AMX int8 codebook ST: 2.70 ms = ~796 Gop/s-equiv (matrix engine, not sdot-bound)\n");
  std::printf("=> if we're near ceiling, no ST SIMD kernel beats us much; Gope's 3x must be multi-thread.\n");
  return 0;
}
