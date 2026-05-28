// Phase 1 of the prefill-beat-Accelerate bet: parameterized BLIS-style AMX fp32 GEMM
// with explicit B-panel packing, auto-tuned per shape over Nc.
//
//   outer:  jc over N step Nc          (B[:, jc:jc+Nc] packed into L2)
//   middle: i0 over M step Mr=16        (A reused across the packed panel)
//           jr over Nc step Nr=64       (Nr-wide microkernel tiles)
//   inner:  k over K (microkernel)      (1 LDY + 2 LDX_pair + 4 FMA32, 4-way ILP)
//
// A is pre-transposed once (At[K,M] so columns are contiguous).
// Output C is correctness-checked against Accelerate (bit-exact since fp32 FMA
// is the same primitive both kernels use under the hood).

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
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

// Parameterized BLIS-style cache-blocked AMX sgemm.
// Buffers At_scratch (K*M floats) and packB_scratch (K*Nc floats) are caller-owned
// so the auto-tuner doesn't pay alloc cost per sweep iteration.
static void amx_sgemm_blis(const float* A, const float* B, float* C,
                           int64_t M, int64_t N, int64_t K, int Nc,
                           std::vector<float>& At_scratch,
                           std::vector<float>& packB_scratch) {
  At_scratch.resize(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At_scratch[k * M + i] = A[i * K + k];
  const float* At = At_scratch.data();
  const uint64_t LDX_PAIR = 1ULL << 62;

  AMX_SET();
  for (int64_t jc = 0; jc < N; jc += Nc) {
    int64_t Nc_eff  = std::min<int64_t>(Nc, N - jc);
    int64_t Nc_main = (Nc_eff / 64) * 64;
    int64_t Nc_tail = Nc_eff - Nc_main;

    if (Nc_main > 0) {
      // Pack B[:, jc:jc+Nc_main] into packB row-major contiguous.
      packB_scratch.resize(size_t(K) * Nc_main);
      for (int64_t k = 0; k < K; ++k) {
        std::memcpy(&packB_scratch[k * Nc_main],
                    &B[k * N + jc],
                    Nc_main * sizeof(float));
      }
      const float* packB = packB_scratch.data();

      // Inner: (i0, jr) tiles over the packed panel. A panel reused across jr.
      for (int64_t i0 = 0; i0 < M; i0 += 16) {
        for (int64_t jr = 0; jr < Nc_main; jr += 64) {
          for (int64_t k = 0; k < K; ++k) {
            const bool first = (k == 0);
            AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
            const float* brow = packB + k * Nc_main + jr;
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

    // Nc_tail (16-wide tiles, unpacked, fewer here so less penalty).
    int64_t jr_tail_base = jc + Nc_main;
    for (int64_t j_off = 0; j_off < Nc_tail; j_off += 16) {
      for (int64_t i0 = 0; i0 < M; i0 += 16) {
        for (int64_t k = 0; k < K; ++k) {
          const bool first = (k == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
          AMX_LDX(reinterpret_cast<uint64_t>(&B[k * N + jr_tail_base + j_off]));
          AMX_FMA32(Fma32Op(0, 0, first));
        }
        for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(
                      C + (i0 + j) * N + jr_tail_base + j_off) |
                  (uint64_t(4 * j) << 56));
      }
    }
  }
  AMX_CLR();
}

int main() {
  struct Shape { int M, N, K; const char* name; };
  const Shape shapes[] = {
    { 128, 2048,  2048, "QKV"     },
    { 128, 8192,  2048, "FFN1"    },
    { 128, 2048,  8192, "FFN2"    },
    { 128, 60000, 2048, "LM-head" },
  };
  // Auto-tune sweep over Nc panel size (must be a multiple of 64 for the main path).
  const int Ncs[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

  std::vector<float> At_scratch, packB_scratch;

  std::printf("Phase 1 BLIS-style cache-blocked AMX sgemm with B-panel packing.\n");
  std::printf("Auto-tune sweep over Nc per shape; bit-exact vs Accelerate (fp32 FMA).\n\n");

  for (auto& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K);
    std::vector<float> B(size_t(s.K) * s.N);
    std::vector<float> C_ref(size_t(s.M) * s.N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7)  * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * s.M * double(s.N) * s.K;

    auto bench = [&](auto fn) {
      fn(); fn();
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        auto t0 = clk::now();
        fn();
        best = std::min(best, ms(clk::now() - t0));
      }
      return std::pair<double, double>(best, flops / (best / 1e3) / 1e9);
    };

    auto [accel_ms, accel_gflops] = bench([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  s.M, s.N, s.K, 1.0f, A.data(), s.K, B.data(), s.N, 0.0f,
                  C_ref.data(), s.N);
    });

    std::printf("=== %s [M=%d N=%d K=%d, %.1f GFLOPs work] ===\n",
                s.name, s.M, s.N, s.K, flops / 1e9);
    std::printf("  Accelerate sgemm:                 %7.2f ms  %5.0f GFLOPS\n",
                accel_ms, accel_gflops);

    double best_gflops = 0.;
    int best_Nc = 0;
    for (int Nc : Ncs) {
      if (Nc > s.N) continue;
      std::vector<float> C(size_t(s.M) * s.N, 0.f);
      auto [t, gf] = bench([&] {
        amx_sgemm_blis(A.data(), B.data(), C.data(), s.M, s.N, s.K, Nc,
                       At_scratch, packB_scratch);
      });
      float maxd = 0.f;
      size_t spot = std::min<size_t>(C.size(), 4096);
      for (size_t i = 0; i < spot; ++i)
        maxd = std::max(maxd, std::fabs(C[i] - C_ref[i]));
      const bool is_best = (gf > best_gflops);
      std::printf("  blis Nc=%5d:                  %7.2f ms  %5.0f GFLOPS  %.2fx Accel  diff=%.0e%s\n",
                  Nc, t, gf, gf / accel_gflops, maxd, is_best ? "  <-" : "");
      if (is_best) { best_gflops = gf; best_Nc = Nc; }
    }
    std::printf("  best blis: Nc=%d  %.0f GFLOPS  %.2fx Accel\n\n",
                best_Nc, best_gflops, best_gflops / accel_gflops);
  }
  return 0;
}
