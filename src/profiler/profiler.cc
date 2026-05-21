#include "profiler/profiler.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <mach/mach.h>
#include <mach/task.h>
#include <mach/task_info.h>

#include "json.hpp"

namespace inferc {
namespace prof {

using clock_t_ = std::chrono::steady_clock;

namespace {
double ToMs(clock_t_::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}
}  // namespace

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  if (v.size() == 1) return v[0];
  // Linear interpolation between closest ranks ("type 7" / numpy default).
  const double rank = (p / 100.0) * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(rank));
  const size_t hi = static_cast<size_t>(std::ceil(rank));
  if (lo == hi) return v[lo];
  const double frac = rank - static_cast<double>(lo);
  return v[lo] + frac * (v[hi] - v[lo]);
}

Stats StatsFrom(std::vector<double> v) {
  Stats s;
  if (v.empty()) return s;
  double sum = 0;
  s.min = v[0]; s.max = v[0];
  for (double x : v) { sum += x; if (x < s.min) s.min = x; if (x > s.max) s.max = x; }
  s.mean = sum / static_cast<double>(v.size());
  s.p50 = Percentile(v, 50.0);
  s.p95 = Percentile(v, 95.0);
  return s;
}

void Profiler::BeginIteration() {
  cur_ = IterRecord{};
  iter_t0_ = clock_t_::now();
  in_flight_op_.reset();
}

void Profiler::EndIteration() {
  const auto t1 = clock_t_::now();
  cur_.total_ms = ToMs(t1 - iter_t0_);
  // Activation peak from the recorded ops.
  for (const auto& r : cur_.ops) {
    if (r.activation_bytes_after > cur_.activation_peak) {
      cur_.activation_peak = r.activation_bytes_after;
    }
  }
  iters_.push_back(std::move(cur_));
  cur_ = IterRecord{};
}

void Profiler::BeginOp(std::string op_type, std::string node_name) {
  if (in_flight_op_) {
    throw std::runtime_error("Profiler: BeginOp called while another op is in flight");
  }
  in_flight_op_.emplace(std::move(op_type), std::move(node_name));
  op_t0_ = clock_t_::now();
}

void Profiler::EndOp(int64_t activation_bytes_after) {
  if (!in_flight_op_) {
    throw std::runtime_error("Profiler: EndOp called without BeginOp");
  }
  const auto t1 = clock_t_::now();
  OpRecord r;
  r.op_type = std::move(in_flight_op_->first);
  r.node_name = std::move(in_flight_op_->second);
  r.ms = ToMs(t1 - op_t0_);
  r.activation_bytes_after = activation_bytes_after;
  cur_.ops.push_back(std::move(r));
  in_flight_op_.reset();
}

void Profiler::SnapshotPeakRss() {
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    peak_rss_bytes_ = static_cast<int64_t>(info.resident_size_max);
  }
}

Stats Profiler::TotalStats() const {
  std::vector<double> v;
  v.reserve(iters_.size());
  for (const auto& it : iters_) v.push_back(it.total_ms);
  return StatsFrom(std::move(v));
}

std::map<std::string, Stats> Profiler::PerOpTypeStats() const {
  // For each iter, sum ms by op_type. Then aggregate across iters.
  std::map<std::string, std::vector<double>> by_op;
  for (const auto& it : iters_) {
    std::map<std::string, double> iter_sums;
    for (const auto& r : it.ops) iter_sums[r.op_type] += r.ms;
    for (auto& [k, v] : iter_sums) by_op[k].push_back(v);
    // Ensure op_types absent in this iter still get a 0 — only if seen
    // elsewhere. (Skip — DistilBERT runs the same graph every iter.)
  }
  std::map<std::string, Stats> out;
  for (auto& [k, v] : by_op) out[k] = StatsFrom(std::move(v));
  return out;
}

std::map<std::string, int64_t> Profiler::OpCountsPerIter() const {
  std::map<std::string, int64_t> out;
  if (iters_.empty()) return out;
  for (const auto& r : iters_.front().ops) ++out[r.op_type];
  return out;
}

int64_t Profiler::ActivationPeakBytes() const {
  int64_t peak = 0;
  for (const auto& it : iters_) {
    if (it.activation_peak > peak) peak = it.activation_peak;
  }
  return peak;
}

std::string Profiler::ToJson(const std::string& backend_name,
                             const std::string& model_path) const {
  using nlohmann::json;
  json j;
  j["backend"] = backend_name;
  j["model"] = model_path;
  j["iterations"] = static_cast<int64_t>(iters_.size());

  auto stats_to_json = [](const Stats& s) {
    json o;
    o["mean_ms"] = s.mean;
    o["p50_ms"] = s.p50;
    o["p95_ms"] = s.p95;
    o["min_ms"] = s.min;
    o["max_ms"] = s.max;
    return o;
  };

  j["total"] = stats_to_json(TotalStats());

  json per_op = json::object();
  const auto stats = PerOpTypeStats();
  const auto counts = OpCountsPerIter();
  for (const auto& [op, s] : stats) {
    json entry;
    entry["calls_per_iter"] = counts.count(op) ? counts.at(op) : 0;
    entry["total_ms"] = stats_to_json(s);
    per_op[op] = std::move(entry);
  }
  j["per_op_type"] = std::move(per_op);

  json op_counts = json::object();
  for (const auto& [op, c] : counts) op_counts[op] = c;
  j["op_counts"] = std::move(op_counts);

  j["peak_rss_bytes"] = peak_rss_bytes_;
  j["activation_bytes_peak"] = ActivationPeakBytes();
  return j.dump(2);
}

}  // namespace prof
}  // namespace inferc
