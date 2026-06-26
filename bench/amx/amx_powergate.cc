// NEW-ANOMALY HUNT: does the Apple M1 AMX block power-gate, paying a cold-start
// wake-up penalty after idle? Intel AMX pays ~20,000 cyc (GATEBLEED, MICRO'25);
// the Apple-AMX wake-up curve is UNMEASURED. Method: warm AMX, then busy-spin on
// NON-AMX work (so the AMX block idles) for a swept duration, then time the FIRST
// AMX op. If first-op latency grows with idle duration -> power-gating anomaly
// (perf: keep-warm heuristic; security: a timing side channel).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static inline double ns(clk::duration d){return std::chrono::duration<double,std::nano>(d).count();}
static inline uint64_t Fma32(int z){return (uint64_t(z)<<20);}

// non-AMX busy work the compiler won't elide; returns a value to consume.
static volatile double sink = 0;
static double spin(int64_t iters){ double a=1.0; for(int64_t i=0;i<iters;++i) a=a*1.0000001+1e-9; return a; }

int main(){
  AMX_SET();
  for(int i=0;i<5000;++i) AMX_FMA32(Fma32(0));            // warm

  std::printf("%-16s %-14s %s\n","idle spin","~idle (us)","first-AMX-op latency (ns), median of 200");
  // calibrate spin->time
  { auto t0=clk::now(); sink=spin(10'000'000); double us=ns(clk::now()-t0)/1e3; std::printf("(calib: 10M spin = %.1f us)\n", us); }

  for(int64_t S : {0LL, 1000LL, 10'000LL, 100'000LL, 1'000'000LL, 5'000'000LL, 20'000'000LL}){
    double best=1e30; // min over trials = cleanest single-op latency after idle
    double med[200]; int n=0;
    for(int trial=0; trial<200; ++trial){
      if(S) sink=spin(S);                                  // AMX idles during this
      auto t0=clk::now();
      AMX_FMA32(Fma32(0));                                 // the FIRST AMX op after idle
      auto t1=clk::now();
      double l=ns(t1-t0); med[n++]=l; if(l<best)best=l;
      // touch AMX a bit so next trial's idle is a real cold->warm transition
      for(int i=0;i<3;++i) AMX_FMA32(Fma32(0));
    }
    // median
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) if(med[j]<med[i]){double t=med[i];med[i]=med[j];med[j]=t;}
    double mil = (n? med[n/2]:0);
    double us=0; { auto t0=clk::now(); if(S) sink=spin(S); us=ns(clk::now()-t0)/1e3; }
    std::printf("%-16lld %-14.1f median=%.1f  min=%.1f\n",(long long)S, us, mil, best);
  }
  AMX_CLR();
  std::printf("\nIf median first-op latency RISES with idle duration -> AMX power-gates\n"
              "(cold-start wake-up anomaly). Flat -> no gating / always warm.\n");
  return 0;
}
