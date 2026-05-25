#include "kernels/amx_probe.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "profiler/profiler.h"  // Percentile / StatsFrom

namespace inferc {
namespace amx {

namespace {

using clock_t_ = std::chrono::steady_clock;

double ToMs(clock_t_::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

double Gflops(double flops, double ms) {
  if (ms <= 0.0) return 0.0;
  return flops / (ms / 1000.0) / 1e9;
}

}  // namespace

const char* KernelName(Kernel k) {
  return k == Kernel::kSgemm ? "sgemm" : "sgemv";
}

ProbeResult ProbeSgemm(int64_t M, int64_t N, int64_t K, int iters, int warmup) {
  // Row-major fp32 buffers. Constant fill is fine: BLAS throughput is
  // data-independent, and a constant output element keeps the result observable
  // so the optimizer can't elide the call.
  std::vector<float> A(static_cast<size_t>(M * K), 0.5f);
  std::vector<float> B(static_cast<size_t>(K * N), 0.5f);
  std::vector<float> C(static_cast<size_t>(M * N), 0.0f);

  const int mi = static_cast<int>(M), ni = static_cast<int>(N), ki = static_cast<int>(K);
  auto call = [&] {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mi, ni, ki, 1.0f,
                A.data(), ki, B.data(), ni, 0.0f, C.data(), ni);
  };

  for (int i = 0; i < warmup; ++i) call();

  std::vector<double> times;
  times.reserve(static_cast<size_t>(iters));
  volatile float sink = 0.0f;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = clock_t_::now();
    call();
    const auto t1 = clock_t_::now();
    times.push_back(ToMs(t1 - t0));
    sink += C[0];  // observe output; defeats dead-store elimination
  }
  (void)sink;

  ProbeResult r;
  r.kernel = Kernel::kSgemm;
  r.M = M; r.N = N; r.K = K;
  r.flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
            static_cast<double>(K);
  const prof::Stats s = prof::StatsFrom(times);
  r.mean_ms = s.mean;
  r.p50_ms = s.p50;
  r.min_ms = s.min;
  r.gflops = Gflops(r.flops, r.min_ms);
  return r;
}

ProbeResult ProbeSgemv(int64_t M, int64_t K, int iters, int warmup) {
  std::vector<float> A(static_cast<size_t>(M * K), 0.5f);
  std::vector<float> x(static_cast<size_t>(K), 0.5f);
  std::vector<float> y(static_cast<size_t>(M), 0.0f);

  const int mi = static_cast<int>(M), ki = static_cast<int>(K);
  auto call = [&] {
    // y = 1.0 * A[M,K] * x[K] + 0.0 * y
    cblas_sgemv(CblasRowMajor, CblasNoTrans, mi, ki, 1.0f, A.data(), ki,
                x.data(), 1, 0.0f, y.data(), 1);
  };

  for (int i = 0; i < warmup; ++i) call();

  std::vector<double> times;
  times.reserve(static_cast<size_t>(iters));
  volatile float sink = 0.0f;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = clock_t_::now();
    call();
    const auto t1 = clock_t_::now();
    times.push_back(ToMs(t1 - t0));
    sink += y[0];
  }
  (void)sink;

  ProbeResult r;
  r.kernel = Kernel::kSgemv;
  r.M = M; r.N = 1; r.K = K;  // GEMV is GEMM with N = 1
  r.flops = 2.0 * static_cast<double>(M) * static_cast<double>(K);
  const prof::Stats s = prof::StatsFrom(times);
  r.mean_ms = s.mean;
  r.p50_ms = s.p50;
  r.min_ms = s.min;
  r.gflops = Gflops(r.flops, r.min_ms);
  return r;
}

ProbeConfig DefaultConfig() {
  ProbeConfig c;
  c.m_sweep = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512};
  c.nk_sweep = {64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
  c.iters = 30;
  c.warmup = 5;
  c.include_gemm = true;
  c.include_gemv = true;
  return c;
}

std::vector<ProbeResult> RunProbe(const ProbeConfig& cfg) {
  std::vector<ProbeResult> out;
  if (cfg.include_gemm) {
    for (int64_t m : cfg.m_sweep) {
      for (int64_t nk : cfg.nk_sweep) {
        out.push_back(ProbeSgemm(m, nk, nk, cfg.iters, cfg.warmup));
      }
    }
  }
  if (cfg.include_gemv) {
    for (int64_t nk : cfg.nk_sweep) {
      out.push_back(ProbeSgemv(nk, nk, cfg.iters, cfg.warmup));
    }
  }
  return out;
}

double EmpiricalPeakGflops(const std::vector<ProbeResult>& results) {
  double peak = 0.0;
  for (const auto& r : results) peak = std::max(peak, r.gflops);
  return peak;
}

}  // namespace amx
}  // namespace inferc
