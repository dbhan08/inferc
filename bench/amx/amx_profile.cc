// Profiling, not copying. Two budgets per shape, single-thread (clean per-AMX
// budget, no GCD mix):
//   (1) per-call vs pre-packed: the weight-pack penalty (why per-call is only
//       parity with BNNS). Measured as pack-B wall-clock fraction.
//   (2) pre-packed vs the ~924 GFLOPS large-GEMM AMX ceiling: the headroom left.
// Then a Kc sweep on the PRE-PACKED compute. With B already packed, Kc only sets
// the Z-accumulate blocking: Kc<K forces K/Kc passes that each LDZ-reload and
// STZ-store the 16x64 accumulator. Larger Kc => fewer Z reloads. This directly
// targets the remaining Llama LM-head loss (K=4096 => 16 passes at Kc=256).

#include <Accelerate/Accelerate.h>
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
static double best(int n, void (^f)()) { f(); f(); double b = 1e30; for (int i = 0; i < n; ++i) { auto t0 = clk::now(); f(); b = std::min(b, ms(clk::now() - t0)); } return b; }

// compute one pre-packed [K x Ncm] panel, Kc-blocked. Z reloaded each pc>0 pass.
static void compute_kc(const float* At, const float* pB, float* C, int64_t M,
                       int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
            AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool f = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + (pc + kk) * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, f)); AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f)); AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
  }
}

int main() {
  struct S { int M, N, K; const char* t; };
  const S sh[] = {
    {128, 2048, 2048, "GPT2 QKV "}, {128, 8192, 2048, "GPT2 FFN1"},
    {128, 2048, 8192, "GPT2 FFN2"}, {128, 60000, 2048, "GPT2 LMh "},
    {128, 4096, 4096, "Llama QKV"}, {128, 11008, 4096, "Llama FN1"},
    {128, 4096, 11008, "LlamaFN2"}, {128, 32000, 4096, "Llama LMh"},
  };
  const double CEIL = 924.0;
  std::printf("=== budget: wall-clock split (single panel, single thread) + util vs %.0f ===\n", CEIL);
  std::printf("%-9s %5s %5s | xpose  pack   cmp  | util\n", "shape", "g(pre)", "shape");
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float* bp = B.data(); float* cp = C.data();
    const int Nc = (N > K) ? 256 : 512, Kc = (N > K) ? 256 : 512;
    const int64_t Ncm = (std::min<int64_t>(Nc, N) / 64) * 64;
    const double panels = std::ceil((double)N / Nc);

    std::vector<float> At(size_t(K) * M);
    const float* ap = A.data(); float* atw = At.data();
    double tT = best(7, ^{ for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) atw[k * M + i] = ap[i * K + k]; });
    const float* atp = At.data();

    std::vector<float> pB(size_t(K) * Ncm); float* pbp = pB.data();
    double tPk = best(7, ^{ for (int64_t k = 0; k < K; ++k) std::memcpy(&pbp[k * Ncm], &bp[k * N], Ncm * sizeof(float)); }) * panels;
    double tC = best(7, ^{ AMX_SET(); compute_kc(atp, pbp, cp, M, N, K, Kc, 0, Ncm); AMX_CLR(); }) * panels;
    double gP = flops / (tC / 1e3) / 1e9;
    double total_pc = tT + tPk + tC;
    std::printf("%-9s %5.0f %5s | %4.0f%% %4.0f%% %4.0f%% | %3.0f%%\n",
                s.t, gP, K > N ? "K>N" : (N > K ? "N>K" : "K=N"),
                100 * tT / total_pc, 100 * tPk / total_pc, 100 * tC / total_pc, 100 * gP / CEIL);
  }

  std::printf("\n=== Kc sweep on pre-packed compute (Z-reload redundancy), full N, MT off ===\n");
  std::printf("%-9s  %s\n", "shape", "Kc=128  256   512   1024  2048  =K (passes shrink ->)");
  const int kcs[] = {128, 256, 512, 1024, 2048, 1 << 30};
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float* bp = B.data(); float* cp = C.data();
    const int Nc = (N > K) ? 256 : 512;
    const int64_t Ncm = (std::min<int64_t>(Nc, N) / 64) * 64;
    const double panels = std::ceil((double)N / Nc);
    std::vector<float> At(size_t(K) * M); const float* ap = A.data();
    for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) At[k * M + i] = ap[i * K + k];
    const float* atp = At.data();
    std::vector<float> pB(size_t(K) * Ncm); float* pbp = pB.data();
    for (int64_t k = 0; k < K; ++k) std::memcpy(&pbp[k * Ncm], &bp[k * N], Ncm * sizeof(float));
    std::printf("%-9s ", s.t);
    for (int kc : kcs) {
      int kce = std::min<int64_t>(kc, K);
      double t = best(7, ^{ AMX_SET(); compute_kc(atp, pbp, cp, M, N, K, kce, 0, Ncm); AMX_CLR(); }) * panels;
      std::printf("%5.0f ", flops / (t / 1e3) / 1e9);
    }
    std::printf("\n");
  }
  std::printf("\n(util%% is compute-only vs the %.0f ceiling; pack/xpose vanish when B is pre-packed at load.\n"
              " Kc sweep: if a shape peaks at Kc>=K, the Z-reload passes were the redundancy.)\n", CEIL);
  return 0;
}
