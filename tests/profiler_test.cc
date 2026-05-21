#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "json.hpp"
#include "profiler/profiler.h"

using inferc::prof::Percentile;
using inferc::prof::Profiler;
using inferc::prof::Stats;
using inferc::prof::StatsFrom;

TEST(ProfilerStats, PercentileMatchesNumpyLinearInterp) {
  // numpy's default percentile: linear interp between order-statistics.
  std::vector<double> v = {1, 2, 3, 4, 5};
  EXPECT_DOUBLE_EQ(Percentile(v, 0), 1.0);
  EXPECT_DOUBLE_EQ(Percentile(v, 100), 5.0);
  EXPECT_DOUBLE_EQ(Percentile(v, 50), 3.0);
  // rank = 0.25 * 4 = 1.0 → exactly v[1] = 2
  EXPECT_DOUBLE_EQ(Percentile(v, 25), 2.0);
  // rank = 0.75 * 4 = 3.0 → exactly v[3] = 4
  EXPECT_DOUBLE_EQ(Percentile(v, 75), 4.0);
}

TEST(ProfilerStats, PercentileInterpolates) {
  std::vector<double> v = {10, 20, 30, 40};
  // rank = 0.5 * 3 = 1.5 → 20 + 0.5*(30-20) = 25
  EXPECT_DOUBLE_EQ(Percentile(v, 50), 25.0);
}

TEST(ProfilerStats, StatsFromBasic) {
  Stats s = StatsFrom({2.0, 4.0, 6.0, 8.0});
  EXPECT_DOUBLE_EQ(s.mean, 5.0);
  EXPECT_DOUBLE_EQ(s.min, 2.0);
  EXPECT_DOUBLE_EQ(s.max, 8.0);
  EXPECT_DOUBLE_EQ(s.p50, 5.0);  // rank=1.5 → 4 + 0.5*(6-4)
}

TEST(Profiler, RecordsIterationsAndOps) {
  Profiler p;
  for (int i = 0; i < 5; ++i) {
    p.BeginIteration();
    p.BeginOp("MatMul", "n0");
    std::this_thread::sleep_for(std::chrono::microseconds(200));
    p.EndOp(/*activation_bytes_after=*/1024);
    p.BeginOp("Add", "n1");
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    p.EndOp(/*activation_bytes_after=*/2048);
    p.EndIteration();
  }
  ASSERT_EQ(p.iterations().size(), 5u);
  for (const auto& it : p.iterations()) {
    ASSERT_EQ(it.ops.size(), 2u);
    EXPECT_EQ(it.ops[0].op_type, "MatMul");
    EXPECT_EQ(it.ops[1].op_type, "Add");
    EXPECT_EQ(it.activation_peak, 2048);
    EXPECT_GT(it.total_ms, 0.0);
  }
  auto op_counts = p.OpCountsPerIter();
  EXPECT_EQ(op_counts["MatMul"], 1);
  EXPECT_EQ(op_counts["Add"], 1);
  EXPECT_EQ(p.ActivationPeakBytes(), 2048);
}

TEST(Profiler, BeginOpTwiceThrows) {
  Profiler p;
  p.BeginIteration();
  p.BeginOp("A", "a");
  EXPECT_THROW(p.BeginOp("B", "b"), std::runtime_error);
}

TEST(Profiler, EndOpWithoutBeginThrows) {
  Profiler p;
  p.BeginIteration();
  EXPECT_THROW(p.EndOp(0), std::runtime_error);
}

TEST(Profiler, PerOpTypeAggregatesAcrossIters) {
  Profiler p;
  for (int i = 0; i < 3; ++i) {
    p.BeginIteration();
    for (int k = 0; k < 4; ++k) {
      p.BeginOp("MatMul", "m" + std::to_string(k));
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      p.EndOp(0);
    }
    p.EndIteration();
  }
  auto stats = p.PerOpTypeStats();
  ASSERT_TRUE(stats.count("MatMul"));
  // 4 calls per iter, each ~50us → sum per iter > 0
  EXPECT_GT(stats["MatMul"].mean, 0.0);
  EXPECT_EQ(p.OpCountsPerIter()["MatMul"], 4);
}

TEST(Profiler, JsonRoundTripsExpectedKeys) {
  Profiler p;
  for (int i = 0; i < 3; ++i) {
    p.BeginIteration();
    p.BeginOp("MatMul", "m0");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    p.EndOp(512);
    p.EndIteration();
  }
  p.SnapshotPeakRss();
  std::string s = p.ToJson("inferc-baseline", "models/distilbert.onnx");
  auto j = nlohmann::json::parse(s);
  EXPECT_EQ(j["backend"], "inferc-baseline");
  EXPECT_EQ(j["model"], "models/distilbert.onnx");
  EXPECT_EQ(j["iterations"], 3);
  EXPECT_TRUE(j.contains("total"));
  EXPECT_TRUE(j["total"].contains("mean_ms"));
  EXPECT_TRUE(j["total"].contains("p50_ms"));
  EXPECT_TRUE(j["total"].contains("p95_ms"));
  EXPECT_TRUE(j["per_op_type"].contains("MatMul"));
  EXPECT_EQ(j["per_op_type"]["MatMul"]["calls_per_iter"], 1);
  EXPECT_EQ(j["op_counts"]["MatMul"], 1);
  EXPECT_GT(j["peak_rss_bytes"].get<int64_t>(), 0);
  EXPECT_EQ(j["activation_bytes_peak"], 512);
}

TEST(Profiler, PeakRssIsPositive) {
  Profiler p;
  p.SnapshotPeakRss();
  EXPECT_GT(p.peak_rss_bytes(), 0);
}
