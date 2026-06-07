// Llama-shape generalization: measure our adaptive BLIS+Kc kernel against
// Accelerate at the GEMM shapes that dominate prefill for TinyLlama-1.1B
// (the model llama.cpp's Q4_0 SOTA was measured on) and Llama-7B (the
// canonical open-weights LLM). Validates that the QKV/FFN1/FFN2 SOTA-beat
// generalizes across model sizes from 124M to 7B params.
//
// Shapes covered (all M=S=128 prefill batch):
//   GPT-2-small (124M)   H=2048 V=60000  -- existing in prefill_tune.cc
//   TinyLlama-1.1B (1.1B) H=2048 FFN=5632 V=32000
//   Llama-7B (7B)         H=4096 FFN=11008 V=32000
//
// Reuses amx_sgemm_auto from prefill_tune.cc by direct #include of the
// non-main portion. To keep the bench standalone, the kernel is duplicated
// inline (the kernel code is small and frozen at v1.5).

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// === The kernels (copied verbatim from prefill_tune.cc post-Phase-1.5) ===

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
                      &B[(pc + k) * N + jc], Nc_main * sizeof(float));
        }
        const float* packB = packB_scratch.data();
        const bool is_first_pc = (pc == 0);
        for (int64_t i0 = 0; i0 < M; i0 += 16) {
          for (int64_t jr = 0; jr < Nc_main; jr += 64) {
            if (!is_first_pc) {
              for (int t = 0; t < 4; ++t)
                for (int j = 0; j < 16; ++j)
                  AMX_LDZ(reinterpret_cast<uint64_t>(
                              C + (i0 + j) * N + jc + jr + 16 * t) |
                          (uint64_t(4 * j + t) << 56));
            }
            for (int64_t kk = 0; kk < Kc_eff; ++kk) {
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
          AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
          AMX_LDX(reinterpret_cast<uint64_t>(&B[k * N + jr_tail_base + j_off]));
          AMX_FMA32(Fma32Op(0, 0, k == 0));
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

static void amx_sgemm_pipelined_unpacked(const float* A, const float* B, float* C,
                                         int64_t M, int64_t N, int64_t K,
                                         std::vector<float>& At_scratch) {
  At_scratch.resize(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At_scratch[k * M + i] = A[i * K + k];
  const uint64_t LDX_PAIR = 1ULL << 62;
  static const uint64_t fma_A[4] = {
    (0ULL << 20) | (  0ULL << 10) | 0ULL, (1ULL << 20) | ( 64ULL << 10) | 0ULL,
    (2ULL << 20) | (128ULL << 10) | 0ULL, (3ULL << 20) | (192ULL << 10) | 0ULL,
  };
  static const uint64_t fma_B[4] = {
    (0ULL << 20) | (256ULL << 10) | 64ULL, (1ULL << 20) | (320ULL << 10) | 64ULL,
    (2ULL << 20) | (384ULL << 10) | 64ULL, (3ULL << 20) | (448ULL << 10) | 64ULL,
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
        AMX_FMA32(fma_A[0] | fm0); AMX_FMA32(fma_A[1] | fm0);
        AMX_FMA32(fma_A[2] | fm0); AMX_FMA32(fma_A[3] | fm0);
        AMX_LDY(reinterpret_cast<uint64_t>(&At_scratch[(k + 2) * M + i0]) | (0ULL << 56));
        const float* b2 = B + (k + 2) * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(b2)      | (0ULL << 56) | LDX_PAIR);
        AMX_LDX(reinterpret_cast<uint64_t>(b2 + 32) | (2ULL << 56) | LDX_PAIR);
        AMX_FMA32(fma_B[0]); AMX_FMA32(fma_B[1]);
        AMX_FMA32(fma_B[2]); AMX_FMA32(fma_B[3]);
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
        AMX_FMA32(f[0] | fm0); AMX_FMA32(f[1] | fm0);
        AMX_FMA32(f[2] | fm0); AMX_FMA32(f[3] | fm0);
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

static void amx_sgemm_auto(const float* A, const float* B, float* C,
                           int64_t M, int64_t N, int64_t K,
                           std::vector<float>& At_scratch,
                           std::vector<float>& packB_scratch) {
  if (N > 32768) {
    amx_sgemm_pipelined_unpacked(A, B, C, M, N, K, At_scratch);
    return;
  }
  int Nc, Kc;
  if (N >= K * 2) { Nc = int(std::min<int64_t>(N / 2, 4096)); Kc = 256; }
  else            { Nc = int(std::min<int64_t>(N, 2048));     Kc = 512; }
  amx_sgemm_blis_kc(A, B, C, M, N, K, Nc, Kc, At_scratch, packB_scratch);
}

int main() {
  struct Shape { int M, N, K; const char* model; const char* op; };
  // All M=128 batch (prefill). H = hidden dim, V = vocab, FFN = mid dim.
  const Shape shapes[] = {
    // GPT-2-small (124M) — existing baseline, included for comparison
    { 128,  2048, 2048, "GPT-2-small",    "QKV  (H=2048)"        },
    { 128,  8192, 2048, "GPT-2-small",    "FFN1 (4H=8192)"       },
    { 128,  2048, 8192, "GPT-2-small",    "FFN2 (4H=8192)"       },
    { 128, 60000, 2048, "GPT-2-small",    "LM-head (V=60000)"    },
    // TinyLlama-1.1B — what llama.cpp Q4_0 SOTA was measured on
    { 128,  2048, 2048, "TinyLlama-1.1B", "QKV  (H=2048)"        },
    { 128,  5632, 2048, "TinyLlama-1.1B", "FFN1 (FFN=5632)"      },
    { 128,  2048, 5632, "TinyLlama-1.1B", "FFN2 (FFN=5632)"      },
    { 128, 32000, 2048, "TinyLlama-1.1B", "LM-head (V=32000)"    },
    // Llama-7B — canonical open-weights LLM
    { 128,  4096, 4096, "Llama-7B",       "QKV  (H=4096)"        },
    { 128, 11008, 4096, "Llama-7B",       "FFN1 (FFN=11008)"     },
    { 128,  4096, 11008,"Llama-7B",       "FFN2 (FFN=11008)"     },
    { 128, 32000, 4096, "Llama-7B",       "LM-head (V=32000)"    },
  };

  // Honest measurement methodology (replaces the earlier best-of-3 x
  // best-of-sweep reporting, which paired our favorable tail against
  // Accelerate's and inflated the ratios). Per trial we time Accelerate and
  // the DEPLOYABLE adaptive kernel (amx_sgemm_auto) back-to-back so both see
  // the same thermal/scheduler state, then take the per-trial ratio. We report
  // mean +/- sample std over N trials -- not best-of-N, and NOT a per-shape
  // best-of-sweep pick. N via PAIRED_TRIALS env (default 30).
  const char* nenv = std::getenv("PAIRED_TRIALS");
  const int N = nenv ? std::max(5, std::atoi(nenv)) : 30;
  const int WARMUP = 3;
  const char* veclib = std::getenv("VECLIB_MAXIMUM_THREADS");
  std::printf("Llama-shape generalization: adaptive BLIS+Kc vs Accelerate sgemm\n");
  std::printf("M1, PAIRED mean +/- sample-std over %d trials (deployable amx_sgemm_auto,\n"
              "not best-of-sweep). VECLIB_MAXIMUM_THREADS = %s\n\n",
              N, veclib ? veclib : "(unset, Accelerate default)");

  std::vector<float> At_scratch, packB_scratch;

  // Geomean accumulators (log-space) over per-shape mean ratios.
  struct Geo {
    double logsum = 0.0; int n = 0;
    void add(double r) { logsum += std::log(r); ++n; }
    double val() const { return n ? std::exp(logsum / n) : 0.0; }
  };
  Geo g_all, g_qkv, g_ffn1, g_ffn2, g_lm;

  std::printf("  %-15s %-8s %-15s %-15s %-13s\n",
              "model", "op", "Accel GFLOPS", "ours GFLOPS", "ratio");
  std::printf("  %-15s %-8s %-15s %-15s %-13s\n",
              "-----", "--", "------------", "-----------", "-----");

  const char* last_model = "";
  for (auto& s : shapes) {
    if (std::strcmp(s.model, last_model) != 0) { std::printf("\n"); last_model = s.model; }

    std::vector<float> A(size_t(s.M) * s.K), B(size_t(s.K) * s.N);
    std::vector<float> C_ref(size_t(s.M) * s.N, 0.f), C(size_t(s.M) * s.N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7)  * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * s.M * double(s.N) * s.K;

    auto time_gflops = [&](auto fn) {
      auto t0 = clk::now(); fn();
      return flops / (ms(clk::now() - t0) / 1e3) / 1e9;
    };
    auto accel = [&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.M, s.N, s.K,
                  1.0f, A.data(), s.K, B.data(), s.N, 0.0f, C_ref.data(), s.N);
    };
    auto ours = [&] {
      amx_sgemm_auto(A.data(), B.data(), C.data(), s.M, s.N, s.K,
                     At_scratch, packB_scratch);
    };

    for (int w = 0; w < WARMUP; ++w) { accel(); ours(); }

    std::vector<double> ag, og, rg;
    ag.reserve(N); og.reserve(N); rg.reserve(N);
    for (int t = 0; t < N; ++t) {
      double a = time_gflops(accel);   // paired, back-to-back
      double o = time_gflops(ours);
      ag.push_back(a); og.push_back(o); rg.push_back(o / a);
    }

    auto stats = [](const std::vector<double>& v) {
      double m = 0; for (double x : v) m += x; m /= v.size();
      double s2 = 0; for (double x : v) s2 += (x - m) * (x - m);
      double sd = v.size() > 1 ? std::sqrt(s2 / (v.size() - 1)) : 0.0;
      return std::pair<double, double>(m, sd);
    };
    auto [am, asd] = stats(ag);
    auto [om, osd] = stats(og);
    auto [rm, rsd] = stats(rg);

    float maxd = 0.f;
    size_t spot = std::min<size_t>(C.size(), 4096);
    for (size_t i = 0; i < spot; ++i) maxd = std::max(maxd, std::fabs(C[i] - C_ref[i]));

    g_all.add(rm);
    if      (std::strncmp(s.op, "QKV", 3) == 0)     g_qkv.add(rm);
    else if (std::strncmp(s.op, "FFN1", 4) == 0)    g_ffn1.add(rm);
    else if (std::strncmp(s.op, "FFN2", 4) == 0)    g_ffn2.add(rm);
    else if (std::strncmp(s.op, "LM-head", 7) == 0) g_lm.add(rm);

    char opb[8]; std::snprintf(opb, sizeof(opb), "%.4s", s.op);
    std::printf("  %-15s %-8s %4.0f +/- %-7.0f %4.0f +/- %-7.0f %.2f +/- %.2fx%s\n",
                s.model, opb, am, asd, om, osd, rm, rsd,
                maxd > 1e-3f ? "  BUG" : "");
  }

  std::printf("\n=== summary: paired mean ratios, geomean across shapes ===\n");
  std::printf("  QKV     geomean:  %.3fx  (n=%d)\n", g_qkv.val(),  g_qkv.n);
  std::printf("  FFN1    geomean:  %.3fx  (n=%d)\n", g_ffn1.val(), g_ffn1.n);
  std::printf("  FFN2    geomean:  %.3fx  (n=%d)\n", g_ffn2.val(), g_ffn2.n);
  std::printf("  LM-head geomean:  %.3fx  (n=%d)\n", g_lm.val(),   g_lm.n);
  std::printf("  ----\n");
  std::printf("  OVERALL geomean:  %.3fx  (n=%d)\n", g_all.val(),  g_all.n);
  return 0;
}
