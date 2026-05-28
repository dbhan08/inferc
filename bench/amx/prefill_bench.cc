// Prefill regime baseline: measure Accelerate sgemm at typical LLM prefill shapes.
// Accelerate uses AMX internally for fp32 GEMM, so its number is the
// "AMX-fp32 ceiling" any custom AMX prefill kernel would have to beat.
// If Accelerate is far above NEON DOTPROD throughput here, AMX has real headroom
// in the prefill regime (batch>1, compute-bound) that it doesn't have for decode.
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

// Naive AMX fp32 GEMM (4-way ILP across N — banks 0..3), copied inline from
// src/kernels/amx_gemm.cc so this microbench stays standalone.  Requires
// M % 16 == 0 and N handled in 64-blocks with a 16-wide tail.
static inline uint64_t Fma32Op(int zbase, int x_off_bytes, bool first) {
  return (uint64_t(zbase) << 20) | (uint64_t(x_off_bytes) << 10) |
         (first ? (1ULL << 27) : 0);
}
// Cache-blocked AMX fp32 GEMM: outer loop over j (N), inner over i (M), so that
// each B[K, j0:j0+64] panel (K*64*4 bytes ≈ 512 KB at K=2048) stays L2-resident
// while we sweep all M tiles, instead of B being re-streamed M/16 times. A is
// pre-transposed once so A[i0:i0+16, k] is contiguous → no per-i0 packing.
static void amx_sgemm_blocked(const float* A, const float* B, float* C,
                              int64_t M, int64_t N, int64_t K) {
  std::vector<float> At(size_t(K) * M);          // K x M, contiguous columns of A
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];

  AMX_SET();
  int64_t j0 = 0;
  for (; j0 + 64 <= N; j0 += 64) {                 // outer: B panel reused across i0
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        const bool first = (k == 0);
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));   // 16 contiguous A vals
        const float* brow = B + k * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 16) | (1ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 48) | (3ULL << 56));
        AMX_FMA32(Fma32Op(0,   0, first));
        AMX_FMA32(Fma32Op(1,  64, first));
        AMX_FMA32(Fma32Op(2, 128, first));
        AMX_FMA32(Fma32Op(3, 192, first));
      }
      for (int t = 0; t < 4; ++t)
        for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0 + 16 * t) |
                  (uint64_t(4 * j + t) << 56));
    }
  }
  for (; j0 < N; j0 += 16) {                        // 16-wide N tail
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) |
                (uint64_t(4 * j) << 56));
    }
  }
  AMX_CLR();
}

static void amx_sgemm_naive(const float* A, const float* B, float* C,
                            int64_t M, int64_t N, int64_t K) {
  std::vector<float> packA(size_t(K) * 16);
  AMX_SET();
  for (int64_t i0 = 0; i0 < M; i0 += 16) {
    for (int64_t k = 0; k < K; ++k) {
      float* dst = packA.data() + k * 16;
      const float* col = A + i0 * K + k;
      for (int i = 0; i < 16; ++i) dst[i] = col[i * K];
    }
    const float* pa = packA.data();
    int64_t j0 = 0;
    for (; j0 + 64 <= N; j0 += 64) {
      for (int64_t k = 0; k < K; ++k) {
        const bool first = (k == 0);
        AMX_LDY(reinterpret_cast<uint64_t>(pa + k * 16));
        const float* brow = B + k * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 16) | (1ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 48) | (3ULL << 56));
        AMX_FMA32(Fma32Op(0,   0, first));
        AMX_FMA32(Fma32Op(1,  64, first));
        AMX_FMA32(Fma32Op(2, 128, first));
        AMX_FMA32(Fma32Op(3, 192, first));
      }
      for (int t = 0; t < 4; ++t)
        for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0 + 16 * t) |
                  (uint64_t(4 * j + t) << 56));
    }
    for (; j0 < N; j0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_LDY(reinterpret_cast<uint64_t>(pa + k * 16));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) |
                (uint64_t(4 * j) << 56));
    }
  }
  AMX_CLR();
}

int main() {
  struct Shape { int M, N, K; const char* name; };
  // Shapes representative of LLM prefill at GPT-2-small/TinyLlama scale, batch S=128.
  // M = prefill batch (sequence length), B[K,N] = weights, A[M,K] = activations.
  Shape shapes[] = {
    { 128, 2048,  2048, "Q/K/V-proj-like" },
    { 128, 8192,  2048, "FFN1 (up-proj)" },
    { 128, 2048,  8192, "FFN2 (down-proj)" },
    { 128, 60000, 2048, "LM-head" },
  };
  std::printf("AMX-fp32 ceiling (Accelerate sgemm) at LLM prefill shapes, M1 1 thread:\n");
  std::printf("(M=batch S=128; B[K,N]=weights; flops = 2*M*N*K)\n\n");

  for (auto& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K);
    std::vector<float> B(size_t(s.K) * s.N);
    std::vector<float> C(size_t(s.M) * s.N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * s.M * double(s.N) * s.K;
    const double mem_b = 4.0 * (size_t(s.M) * s.K + size_t(s.K) * s.N + size_t(s.M) * s.N);

    auto bench = [&](const char* name, auto fn) {
      fn(); fn();
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        auto t0 = clk::now();
        fn();
        best = std::min(best, ms(clk::now() - t0));
      }
      double gflops = flops / (best / 1e3) / 1e9;
      double arith_intensity = flops / mem_b;
      std::printf("    %-22s  %7.2f ms  %6.0f GFLOPS  (AI=%.1f flops/byte)\n",
                  name, best, gflops, arith_intensity);
    };

    std::printf("  %-17s [M=%d N=%d K=%d, %.1f GFLOPS work]\n",
                s.name, s.M, s.N, s.K, flops / 1e9);
    bench("Accelerate sgemm", [&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  s.M, s.N, s.K, 1.0f, A.data(), s.K, B.data(), s.N, 0.0f, C.data(), s.N);
    });
    // The same shape through our naive direct-AMX sgemm (4-way ILP, no cache blocking).
    std::vector<float> C_amx(size_t(s.M) * s.N, 0.f);
    bench("AMX naive sgemm", [&] {
      amx_sgemm_naive(A.data(), B.data(), C_amx.data(), s.M, s.N, s.K);
    });
    // Cache-blocked variant (j0 outer, i0 inner, A pre-transposed).
    std::vector<float> C_blk(size_t(s.M) * s.N, 0.f);
    bench("AMX cache-blocked", [&] {
      amx_sgemm_blocked(A.data(), B.data(), C_blk.data(), s.M, s.N, s.K);
    });
    // Cheap correctness spot-check (output should match within fp32 rounding).
    float maxd_n = 0.f, maxd_b = 0.f;
    size_t spot = std::min<size_t>(C.size(), 4096);
    for (size_t i = 0; i < spot; ++i) {
      maxd_n = std::max(maxd_n, std::fabs(C[i] - C_amx[i]));
      maxd_b = std::max(maxd_b, std::fabs(C[i] - C_blk[i]));
    }
    std::printf("    [correctness] naive max-diff %.2e  blocked max-diff %.2e\n\n",
                maxd_n, maxd_b);
  }
  return 0;
}
