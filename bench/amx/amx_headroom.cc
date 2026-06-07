// AMX headroom + dual-cluster probe.
//
// Goal: decide whether there is a path to BEAT Accelerate-multithread on M1.
// Three measurements:
//
//   Part 1 — Headroom. Per-shape Accelerate (default multi-thread) GFLOPS as a
//            % of the measured AMX fp32 peak. Where Accelerate is far below
//            peak, the shared AMX block is underfed and there is room to win.
//
//   Part 2 — Second AMX block? M1 has one AMX unit per cluster (P and E).
//            Run the SAME single-thread AMX GEMM on a P-core (QoS user-initiated)
//            and on an E-core (QoS background) and compare GFLOPS. A non-trivial
//            E-core number means a second matrix engine exists.
//
//   Part 3 — Dual-cluster aggregate (the decisive test). Run two independent
//            QKV GEMMs concurrently, (a) both dispatched to P, (b) one to P and
//            one to E. If P+E finishes faster than P+P, the E-cluster AMX adds
//            real aggregate throughput that Accelerate (P-pinned) leaves on the
//            table — the mechanism for a single-machine win.
//
// macOS gives no hard core affinity; QoS class is the standard proxy
// (USER_INITIATED -> P-cluster, BACKGROUND -> E-cluster). Results are
// indicative, not guaranteed placement; we report enough runs to see the trend.

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
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}
static inline uint64_t Fma32Op(int zbase, int x_off_bytes, bool first) {
  return (uint64_t(zbase) << 20) | (uint64_t(x_off_bytes) << 10) |
         (first ? (1ULL << 27) : 0);
}

// One jc panel of the BLIS+Kc kernel (copied from multithread_bench.cc, frozen).
static void blis_kc_panel(const float* At, const float* B, float* C,
                          int64_t M, int64_t N, int64_t K, int Kc,
                          int64_t jc, int64_t Nc_main,
                          std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Nc_main);
  AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k)
      std::memcpy(&packB[k * Nc_main], &B[(pc + k) * N + jc],
                  Nc_main * sizeof(float));
    const float* pB = packB.data();
    const bool is_first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t jr = 0; jr < Nc_main; jr += 64) {
        if (!is_first_pc)
          for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 16; ++j)
              AMX_LDZ(reinterpret_cast<uint64_t>(
                          C + (i0 + j) * N + jc + jr + 16 * t) |
                      (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool first = (is_first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + kk * Nc_main + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0,   0, first));
          AMX_FMA32(Fma32Op(1,  64, first));
          AMX_FMA32(Fma32Op(2, 128, first));
          AMX_FMA32(Fma32Op(3, 192, first));
        }
        for (int t = 0; t < 4; ++t)
          for (int j = 0; j < 16; ++j)
            AMX_STZ(reinterpret_cast<uint64_t>(
                        C + (i0 + j) * N + jc + jr + 16 * t) |
                    (uint64_t(4 * j + t) << 56));
      }
    }
  }
  AMX_CLR();
}

// Self-contained single-thread GEMM (allocates its own scratch so it is safe to
// run inside a GCD block on any core). Nc must be a multiple of 64.
static void amx_sgemm_st(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K, int Nc, int Kc) {
  std::vector<float> At(size_t(K) * M), packB;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
  for (int64_t jc = 0; jc < N; jc += Nc) {
    int64_t Nc_eff = std::min<int64_t>(Nc, N - jc);
    int64_t Nc_main = (Nc_eff / 64) * 64;
    if (Nc_main > 0)
      blis_kc_panel(At.data(), B, C, M, N, K, Kc, jc, Nc_main, packB);
  }
}

// Run a block synchronously on a global queue of the given QoS, return ms.
static double run_on(dispatch_qos_class_t qos, void (^work)(void)) {
  dispatch_queue_t q = dispatch_get_global_queue(qos, 0);
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  __block double t = 0;
  dispatch_async(q, ^{
    auto t0 = clk::now();
    work();
    t = ms(clk::now() - t0);
    dispatch_semaphore_signal(sem);
  });
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  return t;
}

static double best_ms(int iters, void (^run)(void)) {
  run(); run();  // warmup
  double best = 1e30;
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now(); run(); best = std::min(best, ms(clk::now() - t0));
  }
  return best;
}

int main() {
  const char* veclib = std::getenv("VECLIB_MAXIMUM_THREADS");
  std::printf("VECLIB_MAXIMUM_THREADS = %s\n\n",
              veclib ? veclib : "(unset, Accelerate default multi-thread)");

  // ---- measure AMX fp32 peak: a large square GEMM via Accelerate (default) ----
  double peak = 0;
  {
    int n = 2048;
    std::vector<float> A(size_t(n) * n, 1.0f), B(size_t(n) * n, 1.0f),
        C(size_t(n) * n, 0.f);
    const float *a = A.data(), *b = B.data(); float* c = C.data();
    double flops = 2.0 * n * double(n) * n;
    double t = best_ms(5, ^{
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
                  a, n, b, n, 0.0f, c, n);
    });
    peak = flops / (t / 1e3) / 1e9;
    std::printf("Measured AMX fp32 peak (Accelerate %dx%dx%d): %.0f GFLOPS\n\n",
                n, n, n, peak);
  }

  struct Shape { int M, N, K; const char* tag; };
  const Shape shapes[] = {
    {128, 2048, 2048, "QKV    (H=2048)"}, {128, 8192, 2048, "FFN1   (H=2048)"},
    {128, 2048, 8192, "FFN2   (H=2048)"}, {128, 60000, 2048, "LM-head(H=2048)"},
    {128, 4096, 4096, "QKV    (H=4096)"}, {128, 11008, 4096, "FFN1   (H=4096)"},
    {128, 4096, 11008, "FFN2  (H=4096)"}, {128, 32000, 4096, "LM-head(H=4096)"},
  };

  // ---- Part 1: headroom — Accelerate (default) % of peak per shape ----
  std::printf("=== Part 1: Accelerate (default multi-thread) headroom ===\n");
  std::printf("  %-16s %-12s %-10s\n", "shape", "GFLOPS", "%% of peak");
  for (auto& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K, 1.0f), B(size_t(s.K) * s.N, 1.0f),
        C(size_t(s.M) * s.N, 0.f);
    const float *a = A.data(), *b = B.data(); float* c = C.data();
    int MM = s.M, NN = s.N, KK = s.K;
    double flops = 2.0 * s.M * double(s.N) * s.K;
    double t = best_ms(5, ^{
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MM, NN, KK, 1.0f,
                  a, KK, b, NN, 0.0f, c, NN);
    });
    double gf = flops / (t / 1e3) / 1e9;
    std::printf("  %-16s %-12.0f %5.0f%%   %s\n", s.tag, gf, 100.0 * gf / peak,
                gf < 0.6 * peak ? "<- underfed (headroom)" : "");
  }

  // ---- Part 2: single-block throughput, P-core vs E-core ----
  std::printf("\n=== Part 2: single AMX GEMM, P-core vs E-core (QKV 128x2048x2048) ===\n");
  {
    int M = 128, N = 2048, K = 2048, Nc = 2048, Kc = 512;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const float *a = A.data(), *b = B.data(); float* c = C.data();
    double flops = 2.0 * M * double(N) * K;
    auto gemm = ^{ amx_sgemm_st(a, b, c, M, N, K, Nc, Kc); };

    double tP = 1e30, tE = 1e30;
    for (int i = 0; i < 7; ++i) {
      tP = std::min(tP, run_on(QOS_CLASS_USER_INITIATED, gemm));
      tE = std::min(tE, run_on(QOS_CLASS_BACKGROUND, gemm));
    }
    std::printf("  P-core (USER_INITIATED): %6.0f GFLOPS\n", flops / (tP / 1e3) / 1e9);
    std::printf("  E-core (BACKGROUND):     %6.0f GFLOPS\n", flops / (tE / 1e3) / 1e9);
    std::printf("  -> %s\n", flops / (tE / 1e3) / 1e9 > 50
                    ? "E-cluster has a usable AMX block (second matrix engine)."
                    : "E-cluster AMX not engaging via QoS proxy.");
  }

  // ---- Part 3: dual-cluster aggregate — two QKV GEMMs, P+P vs P+E ----
  std::printf("\n=== Part 3: two concurrent QKV GEMMs, P+P vs P+E ===\n");
  {
    int M = 128, N = 2048, K = 2048, Nc = 2048, Kc = 512;
    auto mk = [&](std::vector<float>& A, std::vector<float>& B,
                  std::vector<float>& C) {
      A.assign(size_t(M) * K, 0.f); B.assign(size_t(K) * N, 0.f);
      C.assign(size_t(M) * N, 0.f);
      for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
      for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    };
    std::vector<float> A1, B1, C1, A2, B2, C2;
    mk(A1, B1, C1); mk(A2, B2, C2);
    const float *a1 = A1.data(), *b1 = B1.data(); float* c1 = C1.data();
    const float *a2 = A2.data(), *b2 = B2.data(); float* c2 = C2.data();
    double flops2 = 2.0 * (2.0 * M * double(N) * K);  // two GEMMs

    auto run2 = [&](dispatch_qos_class_t q1, dispatch_qos_class_t q2) -> double {
      double best = 1e30;
      for (int i = 0; i < 9; ++i) {
        dispatch_group_t g = dispatch_group_create();
        auto t0 = clk::now();
        dispatch_group_async(g, dispatch_get_global_queue(q1, 0),
                             ^{ amx_sgemm_st(a1, b1, c1, M, N, K, Nc, Kc); });
        dispatch_group_async(g, dispatch_get_global_queue(q2, 0),
                             ^{ amx_sgemm_st(a2, b2, c2, M, N, K, Nc, Kc); });
        dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
        best = std::min(best, ms(clk::now() - t0));
      }
      return best;
    };

    double tPP = run2(QOS_CLASS_USER_INITIATED, QOS_CLASS_USER_INITIATED);
    double tPE = run2(QOS_CLASS_USER_INITIATED, QOS_CLASS_BACKGROUND);
    double gPP = flops2 / (tPP / 1e3) / 1e9, gPE = flops2 / (tPE / 1e3) / 1e9;
    std::printf("  both on P (P+P):  %6.0f GFLOPS aggregate  (%.2f ms)\n", gPP, tPP);
    std::printf("  split P + E:      %6.0f GFLOPS aggregate  (%.2f ms)\n", gPE, tPE);
    std::printf("  -> P+E / P+P = %.2fx. %s\n", gPE / gPP,
                gPE > 1.10 * gPP
                    ? "Second AMX block adds throughput -> dual-cluster is a real lever."
                    : "No aggregate gain from the E-cluster on this shape.");
  }
  return 0;
}
