#pragma once

#include <cstdint>
#include <string>
#include <vector>

// AMX engagement microbenchmark.
//
// Apple's Accelerate framework dispatches `cblas_sgemm` / `cblas_sgemv` either
// to the AMX matrix coprocessor (high throughput) or to a NEON SIMD fallback
// (lower throughput), depending on undocumented shape thresholds. This probe
// sweeps a grid of (M, N, K) shapes, times each BLAS call, and reports achieved
// GFLOPs. The throughput cliff between the two regimes *is* the AMX engagement
// threshold — Figure 1 of the v2 paper.
//
// We treat GEMV as the N=1 special case of GEMM, so every measurement carries a
// uniform (M, N, K) shape and FLOP count 2*M*N*K.
namespace inferc {
namespace amx {

enum class Kernel { kSgemm, kSgemv };

const char* KernelName(Kernel k);

struct ProbeResult {
  Kernel kernel;
  int64_t M = 0;
  int64_t N = 0;  // GEMV always reports N = 1
  int64_t K = 0;
  double flops = 0.0;     // 2 * M * N * K
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double min_ms = 0.0;    // best-observed time; basis for `gflops`
  double gflops = 0.0;    // flops / (min_ms/1000) / 1e9
};

struct ProbeConfig {
  // GEMM grid is the cartesian product m_sweep x nk_sweep, evaluated as
  // sgemm(M = m, N = nk, K = nk). m=1 is the autoregressive-decode row.
  std::vector<int64_t> m_sweep;
  // Feature dims (N = K). Also the square sizes for the GEMV sweep:
  // sgemv(M = nk, K = nk), the matrix-vector equivalent of the m=1 GEMM row.
  std::vector<int64_t> nk_sweep;
  int iters = 30;
  int warmup = 5;
  bool include_gemm = true;
  bool include_gemv = true;
};

// Default sweep producing the paper's Figure 1 (n=30, warmup=5). m_sweep spans
// 1..512 rows; nk_sweep spans 64..2048 feature dims (768 = GPT-2-small hidden).
ProbeConfig DefaultConfig();

// Time a single sgemm: C[M,N] = A[M,K] * B[K,N].
ProbeResult ProbeSgemm(int64_t M, int64_t N, int64_t K, int iters, int warmup);

// Time a single sgemv: y[M] = A[M,K] * x[K]. Reported with N = 1.
ProbeResult ProbeSgemv(int64_t M, int64_t K, int iters, int warmup);

// Run the full sweep described by `cfg`. GEMM grid first, then GEMV sweep.
std::vector<ProbeResult> RunProbe(const ProbeConfig& cfg);

// Max achieved GFLOPs across results — the empirical peak used as the
// reference for "% of peak" (honest alternative to a contested theoretical
// fp32 AMX peak figure for M1).
double EmpiricalPeakGflops(const std::vector<ProbeResult>& results);

}  // namespace amx
}  // namespace inferc
