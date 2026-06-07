// FFN1 (N>K) tuning: can a better (Nc, Kc) for the N>K regime beat Accelerate
// at the two FFN1 shapes we currently lose (TinyLlama 5632/2048, Llama
// 11008/4096)? Sweeps panel sizes for the finer-panel MT kernel vs Accelerate
// (cold, default threading). GPT-2-style FFN1 (already a win) included as a ref.

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}
static void panel(const float* At, const float* B, float* C, int64_t M, int64_t N,
                  int64_t K, int Kc, int64_t jc, int64_t Ncm, std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Ncm);
  AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k)
      std::memcpy(&packB[k * Ncm], &B[(pc + k) * N + jc], Ncm * sizeof(float));
    const float* pB = packB.data();
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
            AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool f = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + kk * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, f)); AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f)); AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
  }
  AMX_CLR();
}
static void tail_cols(const float* At, const float* B, float* C, int64_t M,
                      int64_t N, int64_t K, int64_t j0) {
  AMX_SET();
  for (; j0 + 16 <= N; j0 += 16)
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) | (uint64_t(4 * j) << 56));
    }
  AMX_CLR();
}
static void mt_gemm(const float* A, const float* B, float* C, int64_t M, int64_t N,
                    int64_t K, int Nc, int Kc, std::vector<float>& At,
                    std::vector<std::vector<float>>& packP) {
  At.resize(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
  const float* atp = At.data();
  std::vector<int64_t> jcs;
  for (int64_t jc = 0; jc < N; jc += Nc) jcs.push_back(jc);
  const int nP = (int)jcs.size();
  if ((int)packP.size() < nP) packP.resize(nP);
  int64_t covered = 0;
  for (int64_t jc : jcs) covered = std::max<int64_t>(covered, jc + (std::min<int64_t>(Nc, N - jc) / 64) * 64);
  const int64_t* jcp = jcs.data();
  auto* pp = &packP;
  dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
    int64_t jc = jcp[w], Ncm = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
    if (Ncm > 0) panel(atp, B, C, M, N, K, Kc, jc, Ncm, (*pp)[w]);
  });
  if (covered < N) tail_cols(atp, B, C, M, N, K, covered);
}

int main() {
  struct S { int M, N, K; const char* t; };
  const S sh[] = {
    {128, 11008, 4096, "Llama FFN1   [11008,4096]"},
    {128, 5632, 2048, "TinyLlama FFN1 [5632,2048]"},
    {128, 8192, 2048, "GPT-2-style FFN1 [8192,2048] (ref-win)"},
  };
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f), Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float *ap = A.data(), *bp = B.data(); float* cp = C.data(); float* crefp = Cref.data();

    auto accel = ^{ cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, ap, K, bp, N, 0.0f, crefp, N); };
    accel(); accel();
    double tA = 1e30;
    for (int i = 0; i < 7; ++i) { auto t0 = clk::now(); accel(); tA = std::min(tA, ms(clk::now() - t0)); }
    double gA = flops / (tA / 1e3) / 1e9;

    std::printf("\n=== %s ===  Accelerate %.0f GFLOPS\n", s.t, gA);
    std::vector<float> At; std::vector<std::vector<float>> packP;
    double bestg = 0; int bNc = 0, bKc = 0;
    for (int Nc : {128, 256, 512, 1024}) {
      for (int Kc : {256, 512, 1024, 2048, 4096}) {
        if (Kc > K) continue;
        auto run = [&] { mt_gemm(ap, bp, cp, M, N, K, Nc, Kc, At, packP); };
        run(); run();
        double t = 1e30;
        for (int i = 0; i < 5; ++i) { auto t0 = clk::now(); run(); t = std::min(t, ms(clk::now() - t0)); }
        double g = flops / (t / 1e3) / 1e9;
        float maxd = 0.f;
        for (size_t i = 0; i < C.size(); i += 1023) maxd = std::max(maxd, std::fabs(cp[i] - crefp[i]));
        bool ok = maxd < 1e-3f;
        if (ok && g > bestg) { bestg = g; bNc = Nc; bKc = Kc; }
        std::printf("  Nc=%-5d Kc=%-5d  %5.0f GFLOPS  %.2fx %s\n", Nc, Kc, g, g / gA, ok ? "" : "BUG");
      }
    }
    std::printf("  -> best Nc=%d Kc=%d : %.0f GFLOPS = %.2fx Accelerate %s\n",
                bNc, bKc, bestg, bestg / gA, bestg > gA ? "<- BEATS" : "(loses)");
  }
  return 0;
}
