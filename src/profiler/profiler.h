#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace inferc {
namespace prof {

// One op execution within one inference pass.
struct OpRecord {
  std::string op_type;
  std::string node_name;
  double ms = 0.0;
  int64_t activation_bytes_after = 0;  // live activation bytes after this op
};

// All op records + wall-clock total for one full inference pass.
struct IterRecord {
  double total_ms = 0.0;
  std::vector<OpRecord> ops;
  int64_t activation_peak = 0;
};

struct Stats {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

// Profiler records timings across a sequence of inference passes ("iterations").
// Designed to be wired into Executor::Run() via a nullable hook pointer — when
// null, the executor pays zero cost.
class Profiler {
 public:
  void BeginIteration();
  void EndIteration();

  // Start/stop timing one op within the current iteration. No nesting.
  void BeginOp(std::string op_type, std::string node_name);
  void EndOp(int64_t activation_bytes_after);

  // Sample peak resident set size via mach task_info. Call once after all runs.
  void SnapshotPeakRss();
  int64_t peak_rss_bytes() const { return peak_rss_bytes_; }

  const std::vector<IterRecord>& iterations() const { return iters_; }

  // ---- Aggregations ----
  Stats TotalStats() const;
  // Stats over per-iteration sums of ms for each op_type.
  std::map<std::string, Stats> PerOpTypeStats() const;
  // Static op counts per iteration (taken from the first iteration).
  std::map<std::string, int64_t> OpCountsPerIter() const;
  // Max activation peak across all iterations (bytes).
  int64_t ActivationPeakBytes() const;

  // ---- JSON ----
  // Serialize the report. backend_name is e.g. "inferc-baseline" or "ort-cpu";
  // model_path is informational metadata.
  std::string ToJson(const std::string& backend_name,
                     const std::string& model_path) const;

 private:
  std::vector<IterRecord> iters_;
  IterRecord cur_;
  std::chrono::steady_clock::time_point iter_t0_{};
  std::optional<std::pair<std::string, std::string>> in_flight_op_;
  std::chrono::steady_clock::time_point op_t0_{};
  int64_t peak_rss_bytes_ = 0;
};

// Percentile p in [0,100] of values v (v is copied + sorted internally).
double Percentile(std::vector<double> v, double p);
Stats StatsFrom(std::vector<double> v);

}  // namespace prof
}  // namespace inferc
