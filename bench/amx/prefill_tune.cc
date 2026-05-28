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

// BLIS with Kc blocking. Loop order: jc > pc > i0 > jr, with Z carried across
// pc iterations VIA C (STZ at end of each (i0, jr, pc), LDZ at start of pc>0).
// This is the standard BLIS pattern: A[pc:pc+Kc, i0:i0+Mr] stays L1-hot across
// all jr iterations within a single (i0, pc); pc moves on once we've swept all
// (i0, jr) for that pc chunk.  Kc=K disables Kc blocking (single pc, no LDZ).
static void amx_sgemm_blis_kc(const float* A, const float* B, float* C,
                              int64_t M, int64_t N, int64_t K, int Nc, int Kc,
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
      packB_scratch.resize(size_t(Kc) * Nc_main);
      for (int64_t pc = 0; pc < K; pc += Kc) {
        int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
        for (int64_t k = 0; k < Kc_eff; ++k) {
          std::memcpy(&packB_scratch[k * Nc_main],
                      &B[(pc + k) * N + jc],
                      Nc_main * sizeof(float));
        }
        const float* packB = packB_scratch.data();
        const bool is_first_pc = (pc == 0);
        const bool is_last_pc  = (pc + Kc_eff >= K);

        for (int64_t i0 = 0; i0 < M; i0 += 16) {
          for (int64_t jr = 0; jr < Nc_main; jr += 64) {
            // Load partial Z from C for pc > 0 (carry accumulation across pcs).
            if (!is_first_pc) {
              for (int t = 0; t < 4; ++t)
                for (int j = 0; j < 16; ++j)
                  AMX_LDZ(reinterpret_cast<uint64_t>(
                              C + (i0 + j) * N + jc + jr + 16 * t) |
                          (uint64_t(4 * j + t) << 56));
            }
            for (int64_t kk = 0; kk < Kc_eff; ++kk) {
              // skip-Z (first=true) only on pc=0 kk=0; pc>0 always accumulates onto loaded Z.
              const bool first = (is_first_pc && kk == 0);
              AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
              const float* brow = packB + kk * Nc_main + jr;
              AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
              AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
              AMX_FMA32(Fma32Op(0,   0, first));
              AMX_FMA32(Fma32Op(1,  64, first));
              AMX_FMA32(Fma32Op(2, 128, first));
              AMX_FMA32(Fma32Op(3, 192, first));
            }
            // Always STZ — pc>0 has the running sum, pc=last writes the final value.
            // Intermediate pcs save state so next pc can LDZ.
            (void)is_last_pc;
            for (int t = 0; t < 4; ++t)
              for (int j = 0; j < 16; ++j)
                AMX_STZ(reinterpret_cast<uint64_t>(
                            C + (i0 + j) * N + jc + jr + 16 * t) |
                        (uint64_t(4 * j + t) << 56));
          }
        }
      }
    }

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

// Software-pipelined unpacked variant — wins at huge N (LM-head) where packing
// would just add memory traffic.  Same body as bench/amx/prefill_bench.cc's
// amx_sgemm_pipelined, kept here for the shape-adaptive dispatch.
static void amx_sgemm_pipelined_unpacked(const float* A, const float* B, float* C,
                                         int64_t M, int64_t N, int64_t K,
                                         std::vector<float>& At_scratch) {
  At_scratch.resize(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At_scratch[k * M + i] = A[i * K + k];
  const uint64_t LDX_PAIR = 1ULL << 62;
  static const uint64_t fma_A[4] = {
    (0ULL << 20) | (  0ULL << 10) | 0ULL,
    (1ULL << 20) | ( 64ULL << 10) | 0ULL,
    (2ULL << 20) | (128ULL << 10) | 0ULL,
    (3ULL << 20) | (192ULL << 10) | 0ULL,
  };
  static const uint64_t fma_B[4] = {
    (0ULL << 20) | (256ULL << 10) | 64ULL,
    (1ULL << 20) | (320ULL << 10) | 64ULL,
    (2ULL << 20) | (384ULL << 10) | 64ULL,
    (3ULL << 20) | (448ULL << 10) | 64ULL,
  };

  AMX_SET();
  int64_t j0 = 0;
  for (; j0 + 64 <= N; j0 += 64) {
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      AMX_LDY(reinterpret_cast<uint64_t>(&At_scratch[0 * M + i0]) | (0ULL << 56));
      const float* b0 = B + 0 * N + j0;
      AMX_LDX(reinterpret_cast<uint64_t>(b0)      | (0ULL << 56) | LDX_PAIR);
      AMX_LDX(reinterpret_cast<uint64_t>(b0 + 32) | (2ULL << 56) | LDX_PAIR);
      const uint64_t first_mask = (1ULL << 27);
      int64_t k = 0;
      for (; k + 2 <= K - 1; k += 2) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At_scratch[(k + 1) * M + i0]) | (1ULL << 56));
        const float* b1 = B + (k + 1) * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(b1)      | (4ULL << 56) | LDX_PAIR);
        AMX_LDX(reinterpret_cast<uint64_t>(b1 + 32) | (6ULL << 56) | LDX_PAIR);
        uint64_t fm0 = (k == 0) ? first_mask : 0;
        AMX_FMA32(fma_A[0] | fm0);
        AMX_FMA32(fma_A[1] | fm0);
        AMX_FMA32(fma_A[2] | fm0);
        AMX_FMA32(fma_A[3] | fm0);
        AMX_LDY(reinterpret_cast<uint64_t>(&At_scratch[(k + 2) * M + i0]) | (0ULL << 56));
        const float* b2 = B + (k + 2) * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(b2)      | (0ULL << 56) | LDX_PAIR);
        AMX_LDX(reinterpret_cast<uint64_t>(b2 + 32) | (2ULL << 56) | LDX_PAIR);
        AMX_FMA32(fma_B[0]);
        AMX_FMA32(fma_B[1]);
        AMX_FMA32(fma_B[2]);
        AMX_FMA32(fma_B[3]);
      }
      while (k < K) {
        bool cur_is_B = (k & 1);
        const uint64_t* f = cur_is_B ? fma_B : fma_A;
        uint64_t fm0 = (k == 0) ? first_mask : 0;
        if (k + 1 < K) {
          bool nxt_is_B = !cur_is_B;
          int xreg = nxt_is_B ? 4 : 0;
          AMX_LDY(reinterpret_cast<uint64_t>(&At_scratch[(k + 1) * M + i0]) | (uint64_t(nxt_is_B) << 56));
          const float* bn = B + (k + 1) * N + j0;
          AMX_LDX(reinterpret_cast<uint64_t>(bn)      | (uint64_t(xreg) << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(bn + 32) | (uint64_t(xreg + 2) << 56) | LDX_PAIR);
        }
        AMX_FMA32(f[0] | fm0);
        AMX_FMA32(f[1] | fm0);
        AMX_FMA32(f[2] | fm0);
        AMX_FMA32(f[3] | fm0);
        ++k;
      }
      for (int t = 0; t < 4; ++t)
        for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0 + 16 * t) |
                  (uint64_t(4 * j + t) << 56));
    }
  }
  for (; j0 < N; j0 += 16) {
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At_scratch[k * M + i0]));
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

// Shape-adaptive dispatch — picks the Nc/Kc that won at each regime in the
// Phase 1.5 auto-tune sweep:
//   - N > 32K (LM-head class):  unpacked sw-pipelined (B too big to pack, 0.88x)
//   - N >= K and N <= 32K (FFN1 class):  Nc=min(N/2,4096), Kc=256  (~1.43x)
//   - K > N (FFN2 class):                Nc=min(N,2048), Kc=512   (~1.32x)
//   - small/square (QKV class):          Nc=min(N,2048), Kc=512   (~1.46x)
static void amx_sgemm_auto(const float* A, const float* B, float* C,
                           int64_t M, int64_t N, int64_t K,
                           std::vector<float>& At_scratch,
                           std::vector<float>& packB_scratch) {
  if (N > 32768) {
    amx_sgemm_pipelined_unpacked(A, B, C, M, N, K, At_scratch);
    return;
  }
  int Nc, Kc;
  if (N >= K * 2) {                                  // FFN1-class (N >> K)
    Nc = int(std::min<int64_t>(N / 2, 4096));
    Kc = 256;
  } else {                                           // QKV / FFN2 class
    Nc = int(std::min<int64_t>(N, 2048));
    Kc = 512;
  }
  amx_sgemm_blis_kc(A, B, C, M, N, K, Nc, Kc, At_scratch, packB_scratch);
}

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
    int best_Nc = 0, best_Kc = 0;
    // Phase 1.5: sweep (Nc, Kc) jointly. Reject buggy results (diff > 1e-3).
    const int Kcs[] = { 256, 512, 1024, 2048, 4096, 8192 };
    for (int Nc : Ncs) {
      if (Nc > s.N) continue;
      for (int Kc : Kcs) {
        if (Kc > s.K) continue;
        std::vector<float> C(size_t(s.M) * s.N, 0.f);
        auto [t, gf] = bench([&] {
          amx_sgemm_blis_kc(A.data(), B.data(), C.data(), s.M, s.N, s.K, Nc, Kc,
                            At_scratch, packB_scratch);
        });
        float maxd = 0.f;
        size_t spot = std::min<size_t>(C.size(), 4096);
        for (size_t i = 0; i < spot; ++i)
          maxd = std::max(maxd, std::fabs(C[i] - C_ref[i]));
        const bool valid = (maxd < 1e-3f);
        const bool is_best = (valid && gf > best_gflops);
        if (is_best || !valid) {
          std::printf("  blis Nc=%5d Kc=%5d:       %7.2f ms  %5.0f GFLOPS  %.2fx Accel  diff=%.0e%s\n",
                      Nc, Kc, t, gf, gf / accel_gflops, maxd,
                      valid ? (is_best ? "  <-" : "") : "  *BUG*");
          if (is_best) { best_gflops = gf; best_Nc = Nc; best_Kc = Kc; }
        }
      }
    }
    std::printf("  best valid blis_kc: Nc=%d Kc=%d  %.0f GFLOPS  %.2fx Accel\n",
                best_Nc, best_Kc, best_gflops, best_gflops / accel_gflops);

    // Phase 1.5: shape-adaptive dispatch.
    {
      std::vector<float> C(size_t(s.M) * s.N, 0.f);
      auto [t, gf] = bench([&] {
        amx_sgemm_auto(A.data(), B.data(), C.data(), s.M, s.N, s.K,
                       At_scratch, packB_scratch);
      });
      float maxd = 0.f;
      size_t spot = std::min<size_t>(C.size(), 4096);
      for (size_t i = 0; i < spot; ++i)
        maxd = std::max(maxd, std::fabs(C[i] - C_ref[i]));
      std::printf("  ADAPTIVE dispatch:               %7.2f ms  %5.0f GFLOPS  %.2fx Accel  diff=%.0e\n\n",
                  t, gf, gf / accel_gflops, maxd);
    }
  }
  return 0;
}
