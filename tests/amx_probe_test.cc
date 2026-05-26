#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/amx_probe.h"

namespace {

using inferc::amx::Kernel;
using inferc::amx::ProbeConfig;
using inferc::amx::ProbeResult;

// A single sgemm probe should report the correct FLOP count and a positive,
// finite GFLOPs figure.
TEST(AmxProbe, SgemmFlopsAndGflops) {
  ProbeResult r = inferc::amx::ProbeSgemm(/*M=*/64, /*N=*/64, /*K=*/64,
                                          /*iters=*/3, /*warmup=*/1);
  EXPECT_EQ(r.kernel, Kernel::kSgemm);
  EXPECT_EQ(r.M, 64);
  EXPECT_EQ(r.N, 64);
  EXPECT_EQ(r.K, 64);
  EXPECT_DOUBLE_EQ(r.flops, 2.0 * 64 * 64 * 64);
  EXPECT_GT(r.gflops, 0.0);
  EXPECT_TRUE(std::isfinite(r.gflops));
  EXPECT_GT(r.min_ms, 0.0);
  // min is the best (smallest) time, so min <= mean.
  EXPECT_LE(r.min_ms, r.mean_ms + 1e-9);
}

// GEMV is modeled as GEMM with N=1; FLOPs = 2*M*K.
TEST(AmxProbe, SgemvIsNEqualsOne) {
  ProbeResult r = inferc::amx::ProbeSgemv(/*M=*/256, /*K=*/256,
                                          /*iters=*/3, /*warmup=*/1);
  EXPECT_EQ(r.kernel, Kernel::kSgemv);
  EXPECT_EQ(r.N, 1);
  EXPECT_EQ(r.M, 256);
  EXPECT_EQ(r.K, 256);
  EXPECT_DOUBLE_EQ(r.flops, 2.0 * 256 * 256);
  EXPECT_GT(r.gflops, 0.0);
}

// BNNS fp16 GEMM should run and report a positive, finite GFLOPs figure.
TEST(AmxProbe, BnnsF16RunsAndReportsGflops) {
  ProbeResult r = inferc::amx::ProbeBnnsF16(/*M=*/128, /*N=*/128, /*K=*/128,
                                            /*iters=*/3, /*warmup=*/1);
  EXPECT_EQ(r.kernel, Kernel::kBnnsF16);
  EXPECT_DOUBLE_EQ(r.flops, 2.0 * 128 * 128 * 128);
  EXPECT_GT(r.gflops, 0.0);
  EXPECT_TRUE(std::isfinite(r.gflops));
  EXPECT_GT(r.min_ms, 0.0);
}

// A small sweep should produce one result per grid cell plus the gemv sweep,
// and a positive empirical peak.
TEST(AmxProbe, SweepShapeAndPeak) {
  ProbeConfig cfg;
  cfg.m_sweep = {1, 8, 64};
  cfg.nk_sweep = {64, 256};
  cfg.iters = 2;
  cfg.warmup = 1;
  cfg.include_gemm = true;
  cfg.include_gemv = true;
  cfg.include_bnns_f16 = false;

  std::vector<ProbeResult> results = inferc::amx::RunProbe(cfg);
  // 3*2 gemm cells + 2 gemv sweep points.
  EXPECT_EQ(results.size(), 3u * 2u + 2u);

  const double peak = inferc::amx::EmpiricalPeakGflops(results);
  EXPECT_GT(peak, 0.0);
  for (const auto& r : results) EXPECT_LE(r.gflops, peak + 1e-9);
}

// Toggling include flags drops the corresponding kernel from the sweep.
TEST(AmxProbe, GemmOnlyAndGemvOnly) {
  ProbeConfig base;
  base.m_sweep = {1, 16};
  base.nk_sweep = {128};
  base.iters = 2;
  base.warmup = 0;
  base.include_bnns_f16 = false;

  ProbeConfig gemm_only = base;
  gemm_only.include_gemv = false;
  auto a = inferc::amx::RunProbe(gemm_only);
  EXPECT_EQ(a.size(), 2u);  // 2 m-values x 1 nk, no gemv
  for (const auto& r : a) EXPECT_EQ(r.kernel, Kernel::kSgemm);

  ProbeConfig gemv_only = base;
  gemv_only.include_gemm = false;
  auto b = inferc::amx::RunProbe(gemv_only);
  EXPECT_EQ(b.size(), 1u);  // 1 nk gemv point
  for (const auto& r : b) EXPECT_EQ(r.kernel, Kernel::kSgemv);
}

}  // namespace
