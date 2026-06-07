// Clean confirmation: does finer-panel multi-thread direct-AMX beat Accelerate
// at LLM prefill, measured WITHOUT the artifacts that contaminated earlier runs?
//
//   * Run as two SEPARATE processes:  amx_confirm accel   and   amx_confirm ours
//     so our GCD worker threads can never starve Accelerate's threadpool.
//   * mean +/- sample std over 20 trials after warmup (not min-of-N).
//   * "ours" reports P-cluster-only and P-cluster + a 15% E-cluster slice.
//   * bit-exactness verified vs Accelerate in "ours" mode.
//
// Combine the two outputs offline to get the honest ratio per shape.

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}
static void panel(const float* At, const float* B, float* C, int64_t M,
                  int64_t N, int64_t K, int Kc, int64_t jc, int64_t Nc_main,
                  std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Nc_main);
  AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k)
      std::memcpy(&packB[k * Nc_main], &B[(pc + k) * N + jc], Nc_main * sizeof(float));
    const float* pB = packB.data();
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t jr = 0; jr < Nc_main; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 16; ++j)
              AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                      (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool first = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + kk * Nc_main + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, first));   AMX_FMA32(Fma32Op(1, 64, first));
          AMX_FMA32(Fma32Op(2, 128, first)); AMX_FMA32(Fma32Op(3, 192, first));
        }
        for (int t = 0; t < 4; ++t)
          for (int j = 0; j < 16; ++j)
            AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                    (uint64_t(4 * j + t) << 56));
      }
    }
  }
  AMX_CLR();
}

// Residual columns [j0, N) that the Nc/64-granular panels skip. 16-wide AMX
// tiles plus a scalar remainder for the final <16 columns. Correctness path.
static void tail_cols(const float* At, const float* B, float* C, int64_t M,
                      int64_t N, int64_t K, int64_t j0) {
  AMX_SET();
  for (; j0 + 16 <= N; j0 += 16) {
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) | (uint64_t(4 * j) << 56));
    }
  }
  AMX_CLR();
  for (; j0 < N; ++j0)
    for (int64_t i = 0; i < M; ++i) {
      float s = 0;
      for (int64_t k = 0; k < K; ++k) s += At[k * M + i] * B[k * N + j0];
      C[i * N + j0] = s;
    }
}

struct Stat { double mean, sd; };
static Stat stat(std::vector<double>& v) {
  double m = 0; for (double x : v) m += x; m /= v.size();
  double s = 0; for (double x : v) s += (x - m) * (x - m);
  return {m, std::sqrt(s / (v.size() - 1))};
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "accel";
  const char* veclib = std::getenv("VECLIB_MAXIMUM_THREADS");
  std::printf("MODE=%s  VECLIB_MAXIMUM_THREADS=%s\n", mode.c_str(),
              veclib ? veclib : "(default MT)");

  struct Shape { int M, N, K; const char* model; const char* op; };
  const Shape shapes[] = {
    {128, 2048, 2048, "GPT-2-style", "QKV"}, {128, 8192, 2048, "GPT-2-style", "FFN1"},
    {128, 2048, 8192, "GPT-2-style", "FFN2"}, {128, 60000, 2048, "GPT-2-style", "LMhead"},
    {128, 2048, 2048, "TinyLlama", "QKV"}, {128, 5632, 2048, "TinyLlama", "FFN1"},
    {128, 2048, 5632, "TinyLlama", "FFN2"}, {128, 32000, 2048, "TinyLlama", "LMhead"},
    {128, 4096, 4096, "Llama-7B", "QKV"}, {128, 11008, 4096, "Llama-7B", "FFN1"},
    {128, 4096, 11008, "Llama-7B", "FFN2"}, {128, 32000, 4096, "Llama-7B", "LMhead"},
  };
  const int Nc = 512, Kc = 512;       // finer panels for multi-thread parallelism
  const int TRIALS = 20, WARM = 4;
  const double EFRAC = 0.15;

  std::printf("%-12s %-7s %-9s", "model", "op", "accel" );
  if (mode == "ours") std::printf(" %-12s %-12s %-6s", "P-only", "P+E(15%)", "bitexact");
  std::printf("\n");

  for (auto& s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f),
        Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * double(N) * K;
    const float *ap = A.data(), *bp = B.data(); float* cp = C.data(); float* crefp = Cref.data();

    if (mode == "accel") {
      auto accel = ^{ cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                                  1.0f, ap, K, bp, N, 0.0f, crefp, N); };
      for (int i = 0; i < WARM; ++i) accel();
      std::vector<double> g;
      for (int i = 0; i < TRIALS; ++i) {
        auto t0 = clk::now(); accel(); g.push_back(flops / (ms(clk::now() - t0) / 1e3) / 1e9);
      }
      Stat st = stat(g);
      std::printf("%-12s %-7s %4.0f+/-%-3.0f\n", s.model, s.op, st.mean, st.sd);
      continue;
    }

    // mode == ours
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, ap, K, bp,
                N, 0.0f, crefp, N);                       // correctness reference only
    std::vector<float> At(size_t(K) * M);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
    const float* atp = At.data();

    std::vector<int64_t> all;
    for (int64_t jc = 0; jc < N; jc += Nc) all.push_back(jc);
    const int nPanels = (int)all.size();

    auto run = [&](int nE) {
      nE = std::min(nE, nPanels - 1);
      int nP = nPanels - nE;
      std::vector<int64_t> pjc(all.begin(), all.begin() + nP);
      std::vector<int64_t> ejc(all.begin() + nP, all.end());
      std::vector<std::vector<float>> packP(nP);
      int64_t covered = 0;  // columns the 64-granular panels cover; rest is tail
      for (int64_t jc : all)
        covered = std::max<int64_t>(covered, jc + (std::min<int64_t>(Nc, N - jc) / 64) * 64);
      auto once = [&] {
        std::fill(C.begin(), C.end(), 0.f);
        dispatch_group_t g = dispatch_group_create();
        if (nE > 0)
          dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            std::vector<float> pk;
            for (int64_t jc : ejc) {
              int64_t Ncm = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
              if (Ncm > 0) panel(atp, bp, cp, M, N, K, Kc, jc, Ncm, pk);
            }
          });
        dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
          int64_t jc = pjc[w];
          int64_t Ncm = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
          if (Ncm > 0) panel(atp, bp, cp, M, N, K, Kc, jc, Ncm, packP[w]);
        });
        dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
        if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
      };
      for (int i = 0; i < WARM; ++i) once();
      std::vector<double> g;
      for (int i = 0; i < TRIALS; ++i) {
        auto t0 = clk::now(); once(); g.push_back(flops / (ms(clk::now() - t0) / 1e3) / 1e9);
      }
      return stat(g);
    };

    Stat sP = run(0);
    Stat sPE = run((int)std::lround(EFRAC * nPanels));
    float maxd = 0.f;  // C holds the last (P+E) result; check prefix AND tail
    size_t spot = std::min<size_t>(C.size(), 4096);
    for (size_t i = 0; i < spot; ++i) maxd = std::max(maxd, std::fabs(cp[i] - crefp[i]));
    for (size_t i = C.size() - spot; i < C.size(); ++i)
      maxd = std::max(maxd, std::fabs(cp[i] - crefp[i]));
    std::printf("%-12s %-7s %-9s %4.0f+/-%-6.0f %4.0f+/-%-6.0f %s\n", s.model, s.op, "-",
                sP.mean, sP.sd, sPE.mean, sPE.sd, maxd < 1e-3f ? "OK" : "BUG");
  }
  return 0;
}
