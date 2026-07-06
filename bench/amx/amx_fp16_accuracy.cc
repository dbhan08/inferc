// fp16-in/fp32-accumulate accuracy map -- resolves the accuracy caveat of the
// Paper-2 gate. The kernel will feed fp16 inputs to FMA16 with fp32 accumulate;
// this quantifies how far that lands from bit-exact Accelerate fp32, by shape and
// by K (accumulation depth). Method: round A,B through fp16 then run cblas_sgemm
// (fp32 accumulate) -- byte-identical semantics to fp16-in/fp32-acc -- and diff
// against fp32 sgemm on the original inputs. The gap is purely fp16 input rounding.
//
// Reference point from Paper 1: BNNS Graph computes the rectangular prefill GEMMs
// at reduced precision with max-abs-diff up to 1.4e-3 vs cblas. If fp16-in/fp32-acc
// is at or below that, the precision cost is no worse than a path Accelerate
// already ships -- and the kernel is ~2x faster (the gate result).

#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// deterministic LCG -> N(0,1)-ish via Box-Muller, so runs are reproducible.
struct Rng { uint64_t s; double u(){ s = s*6364136223846793005ULL+1442695040888963407ULL; return ((s>>11)&((1ULL<<53)-1))/9007199254740992.0; } };
static double nrm(Rng& r){ double u1=r.u(), u2=r.u(); if(u1<1e-12)u1=1e-12; return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2); }

int main() {
  struct S { int M, N, K; const char* t; };
  const S sh[] = {
    {128, 2048, 2048,  "QKV-2048      (K=2048)"},
    {128, 8192, 2048,  "FFN1-8192     (K=2048)"},
    {128, 2048, 8192,  "FFN2-8192     (K=8192)"},
    {128, 32000, 2048, "LMhead-32000  (K=2048)"},
    {128, 4096, 4096,  "QKV-4096      (K=4096)"},
    {128, 4096, 11008, "FFN2-Llama    (K=11008)"},
  };
  std::printf("%-24s %-12s %-12s %-12s %s\n",
              "shape", "max-abs", "max-rel", "rms-rel", "vs BNNSGraph 1.4e-3");
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    Rng r{0x9E3779B97F4A7C15ULL ^ (uint64_t)K};
    std::vector<float> A(size_t(M)*K), B(size_t(K)*N), Cf(size_t(M)*N), Ch(size_t(M)*N);
    std::vector<float> A16(A.size()), B16(B.size());
    for (auto& x : A) x = (float)nrm(r);
    for (auto& x : B) x = (float)nrm(r);
    // fp16 round-trip of the inputs (fp16-precision values held in fp32 storage).
    for (size_t i = 0; i < A.size(); ++i) A16[i] = (float)(__fp16)A[i];
    for (size_t i = 0; i < B.size(); ++i) B16[i] = (float)(__fp16)B[i];

    auto gemm = [&](const float* a, const float* b, float* c) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                  1.0f, a, K, b, N, 0.0f, c, N);
    };
    gemm(A.data(),   B.data(),   Cf.data());   // fp32 reference (bit-exact bar)
    gemm(A16.data(), B16.data(), Ch.data());   // fp16-in / fp32-acc

    double maxabs = 0, maxrel = 0, sse = 0, sref = 0;
    for (size_t i = 0; i < Cf.size(); ++i) {
      double d = std::fabs((double)Ch[i] - Cf[i]);
      maxabs = std::max(maxabs, d);
      double den = std::fabs((double)Cf[i]);
      if (den > 1e-6) maxrel = std::max(maxrel, d / den);
      sse += d * d; sref += (double)Cf[i] * Cf[i];
    }
    double rmsrel = std::sqrt(sse / sref);
    std::printf("%-24s %-12.2e %-12.2e %-12.2e %s\n", s.t, maxabs, maxrel, rmsrel,
                maxabs <= 1.4e-3 ? "<= (no worse)" : "> (worse, by shape)");
  }
  std::printf("\nfp16-in/fp32-acc relative error is ~K-independent for zero-mean inputs\n"
              "(sqrt(K) error growth cancels sqrt(K) signal growth). Compare max-abs to\n"
              "Paper 1's BNNS Graph reduced-precision path (up to 1.4e-3). N(0,1) inputs\n"
              "are a neutral stress; real activation/weight scales shift abs but not rel.\n");
  return 0;
}
