// fp16-ACCUMULATE accuracy vs K -- the decisive Paper-2 experiment. The fast AMX
// mode (FMA16 with fp16 accumulate, bit62=0) hit a 3,027 GFLOPS ceiling, ~2x the
// fp32-accumulate mode -- the ONLY place the 2x headline survives. But accumulating
// K products in fp16 loses bits as the running sum grows. This sweeps K and reports
// the accuracy of BOTH accumulate modes (fp16 vs fp32) against a fp64 reference,
// with identical fp16 INPUTS, so the only variable is accumulator precision.
//
// If fp16-acc rms-rel degrades sharply with K -> the 2x mode is unusable for deep
// prefill GEMMs; the safe ~1.3-1.5x fp32-acc kernel is the paper. If it holds ->
// the 2x mode is in play and the accuracy/throughput frontier is the contribution.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct Rng { uint64_t s; double u(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return ((s>>11)&((1ULL<<53)-1))/9007199254740992.0; } };
static double nrm(Rng& r){ double u1=r.u(),u2=r.u(); if(u1<1e-12)u1=1e-12; return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2); }

int main() {
  const int M = 128, N = 128;
  const int Ks[] = {512, 2048, 4096, 8192, 11008};
  std::printf("%-8s %-16s %-16s %s\n", "K", "fp32-acc rms-rel", "fp16-acc rms-rel", "fp16-acc / fp32-acc");
  for (int K : Ks) {
    Rng r{0x9E3779B97F4A7C15ULL ^ (uint64_t)K};
    std::vector<__fp16> Ah(size_t(M)*K), Bh(size_t(K)*N);
    for (auto& x : Ah) x = (__fp16)nrm(r);
    for (auto& x : Bh) x = (__fp16)nrm(r);

    double sse32 = 0, sse16 = 0, sref = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc64 = 0;          // fp64 ground truth (same fp16 inputs)
        float  acc32 = 0.f;        // fp32 accumulate
        __fp16 acc16 = (__fp16)0;  // fp16 accumulate (forced each step)
        for (int k = 0; k < K; ++k) {
          float ah = (float)Ah[size_t(m)*K + k], bh = (float)Bh[size_t(k)*N + n];
          acc64 += (double)ah * bh;
          acc32 += ah * bh;
          acc16 = (__fp16)(acc16 + (__fp16)(ah * bh));   // round product AND sum to fp16
        }
        double d32 = (double)acc32 - acc64, d16 = (double)acc16 - acc64;
        sse32 += d32*d32; sse16 += d16*d16; sref += acc64*acc64;
      }
    }
    double r32 = std::sqrt(sse32/sref), r16 = std::sqrt(sse16/sref);
    std::printf("%-8d %-16.3e %-16.3e %.1fx\n", K, r32, r16, r16/r32);
  }
  std::printf("\nfp32-acc should stay ~3e-4 (input rounding only, K-independent).\n"
              "fp16-acc grows with K (running sum dominates small products -> swamping).\n"
              "Verdict: if fp16-acc rms-rel exceeds ~1e-2 at prefill K (2048-11008),\n"
              "the fast 2x mode is too lossy -> the safe fp32-acc ~1.4x kernel is Paper 2.\n");
  return 0;
}
