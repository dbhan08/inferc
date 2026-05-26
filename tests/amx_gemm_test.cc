#include <gtest/gtest.h>

#include <Accelerate/Accelerate.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "kernels/amx_gemm.h"

namespace {

using clk = std::chrono::steady_clock;

double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

// Correctness on a small shape vs a naive reference.
TEST(AmxGemm, MatchesReferenceSmall) {
  const int64_t M = 32, N = 48, K = 20;  // M,N multiples of 16
  std::vector<float> A(M * K), B(K * N), C(M * N, 0), R(M * N, 0);
  for (int64_t i = 0; i < M * K; ++i) A[i] = std::sin(0.1 * i);
  for (int64_t i = 0; i < K * N; ++i) B[i] = std::cos(0.07 * i);
  inferc::rt::AmxSgemmF32(A.data(), B.data(), C.data(), M, N, K);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0;
      for (int64_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      R[i * N + j] = acc;
    }
  float max_diff = 0;
  for (int64_t i = 0; i < M * N; ++i)
    max_diff = std::max(max_diff, std::fabs(C[i] - R[i]));
  EXPECT_LT(max_diff, 1e-3) << "AMX GEMM diverges from reference";
}

// PoC benchmark on the DistilBERT FFN shape: AMX kernel vs Accelerate sgemm.
TEST(AmxGemm, VsAccelerateFFNShape) {
  const int64_t M = 128, N = 3072, K = 768;  // FFN: [128,768]x[768,3072]
  std::vector<float> A(M * K), B(K * N), C(M * N, 0), Cref(M * N, 0);
  for (int64_t i = 0; i < M * K; ++i) A[i] = std::sin(0.001 * i);
  for (int64_t i = 0; i < K * N; ++i) B[i] = std::cos(0.001 * i);
  const double flops = 2.0 * M * N * K;
  const int iters = 20, warmup = 5;

  auto bench = [&](auto fn) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
      auto t0 = clk::now();
      fn();
      best = std::min(best, ms(clk::now() - t0));
    }
    return best;
  };

  double amx_ms = bench([&] { inferc::rt::AmxSgemmF32(A.data(), B.data(), C.data(), M, N, K); });
  double acc_ms = bench([&] {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
                1.0f, A.data(), (int)K, B.data(), (int)N, 0.0f, Cref.data(), (int)N);
  });

  float max_diff = 0;
  for (int64_t i = 0; i < M * N; ++i)
    max_diff = std::max(max_diff, std::fabs(C[i] - Cref[i]));

  std::printf(
      "\n[AMX PoC] FFN shape 128x3072x768:\n"
      "    custom AMX : %.3f ms  (%.0f GFLOPs)\n"
      "    Accelerate : %.3f ms  (%.0f GFLOPs)\n"
      "    AMX/Accel  : %.2fx   max-abs-diff %.2e\n",
      amx_ms, flops / (amx_ms / 1e3) / 1e9,
      acc_ms, flops / (acc_ms / 1e3) / 1e9,
      acc_ms / amx_ms, max_diff);
  EXPECT_LT(max_diff, 1e-2) << "AMX FFN result diverges from Accelerate";
}

}  // namespace
