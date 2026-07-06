// BLOCKED fp16-accumulate accuracy -- validates the novel-method premise.
// Full-K fp16 accumulate is lossy (error ~= 1.45e-4 * sqrt(K): 6.6e-3 at K=2048,
// 1.5e-2 at K=11008). The FABsum idea (Blanchard-Higham-Mary), mapped onto the
// AMX Z-register drain: accumulate fp16 within K-BLOCKS of size b, spill each
// block's partial to a fp32 accumulator, sum blocks in fp32. Predicted residual
// error ~= 1.45e-4 * sqrt(b), INDEPENDENT of total K. This measures it directly
// (block-fp16 accumulate, fp32 cross-block) vs a fp64 reference, sweeping b at
// the deepest prefill K, to confirm the block size that recovers ~fp32 accuracy.
// Nobody has published this on Apple AMX; existing recovery work (Ootomo, SGEMM-
// cube) fixes INPUT precision with fp32 accumulate -- a different problem.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct Rng { uint64_t s; double u(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return ((s>>11)&((1ULL<<53)-1))/9007199254740992.0; } };
static double nrm(Rng& r){ double u1=r.u(),u2=r.u(); if(u1<1e-12)u1=1e-12; return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2); }

int main() {
  const int M = 96, N = 96;                       // enough output samples
  const int Ks[] = {2048, 11008};                 // shallow + deepest prefill K
  const int Bs[] = {2, 4, 8, 16, 32, 64, 128};    // block size sweep (b)
  for (int K : Ks) {
    std::printf("=== K = %d ===\n", K);
    std::printf("%-10s %-14s %-14s %s\n", "block b", "rms-rel", "predicted 1.45e-4*sqrt(b)", "vs fp32 floor 3e-4");
    Rng rg{0x9E3779B97F4A7C15ULL ^ (uint64_t)K};
    std::vector<__fp16> Ah(size_t(M)*K), Bh(size_t(K)*N);
    for (auto& x : Ah) x = (__fp16)nrm(rg);
    for (auto& x : Bh) x = (__fp16)nrm(rg);

    // fp64 ground truth (same fp16 inputs) once.
    std::vector<double> ref(size_t(M)*N);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double a = 0;
        for (int k = 0; k < K; ++k) a += (double)(float)Ah[size_t(m)*K+k] * (float)Bh[size_t(k)*N+n];
        ref[size_t(m)*N+n] = a;
      }

    for (int b : Bs) {
      double sse = 0, sref = 0;
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          float spill = 0.f;                       // fp32 cross-block accumulator
          for (int k0 = 0; k0 < K; k0 += b) {
            __fp16 blk = (__fp16)0;                // fp16 within-block accumulate
            int ke = std::min(k0 + b, K);
            for (int k = k0; k < ke; ++k)
              blk = (__fp16)(blk + (__fp16)((float)Ah[size_t(m)*K+k] * (float)Bh[size_t(k)*N+n]));
            spill += (float)blk;                   // promote block partial to fp32
          }
          double d = (double)spill - ref[size_t(m)*N+n];
          sse += d*d; sref += ref[size_t(m)*N+n]*ref[size_t(m)*N+n];
        }
      double rms = std::sqrt(sse/sref), pred = 1.45e-4*std::sqrt((double)b);
      std::printf("%-10d %-14.3e %-14.3e %s\n", b, rms, pred,
                  rms <= 5e-4 ? "<= ~fp32 floor" : (rms <= 1e-3 ? "~ within 3x" : "> too coarse"));
    }
    std::printf("\n");
  }
  std::printf("If rms-rel is K-INDEPENDENT and tracks 1.45e-4*sqrt(b), the FABsum\n"
              "mapping holds: pick the largest b at acceptable accuracy (fewest fp32\n"
              "drains = highest throughput). b<=8 should land at/below the ~3e-4 floor.\n");
  return 0;
}
