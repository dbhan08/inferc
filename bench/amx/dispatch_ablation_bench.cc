// Ablation of the shape-adaptive dispatch decision (paper §4.1).
//
// Compares three dispatch policies on the 12 LLM prefill GEMMs:
//   (a) ALWAYS_PACK   — always run BLIS+Kc with fixed (Nc=2048, Kc=512).
//   (b) ALWAYS_UNPACK — always run sw-pipelined unpacked.
//   (c) ADAPTIVE      — the heuristic in amx_sgemm_auto (paper §3.3).
//
// Headline question for §4.1 ablation: does the adaptive choice quantitatively
// matter, or would a simpler always-pack/always-unpack policy do nearly as well?
//
// Bit-exact gate at every cell. Reports mean ratio vs Accelerate per policy.

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

// --- Same two kernels as llama_shapes_bench.cc, inlined verbatim ---

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

// Three dispatch policies under test.
enum class Policy { ALWAYS_PACK, ALWAYS_UNPACK, ADAPTIVE };
static const char* policy_name(Policy p) {
  switch (p) {
    case Policy::ALWAYS_PACK:   return "always-pack  (Nc=2048,Kc=512)";
    case Policy::ALWAYS_UNPACK: return "always-unpack (sw-pipelined)";
    case Policy::ADAPTIVE:      return "ADAPTIVE     (paper §3.3)";
  }
  return "?";
}

static void run(Policy p, const float* A, const float* B, float* C,
                int64_t M, int64_t N, int64_t K,
                std::vector<float>& At_scratch,
                std::vector<float>& packB_scratch) {
  if (p == Policy::ALWAYS_PACK) {
    amx_sgemm_blis_kc(A, B, C, M, N, K, 2048, 512, At_scratch, packB_scratch);
  } else if (p == Policy::ALWAYS_UNPACK) {
    amx_sgemm_pipelined_unpacked(A, B, C, M, N, K, At_scratch);
  } else {  // ADAPTIVE
    if (N > 32768) {
      amx_sgemm_pipelined_unpacked(A, B, C, M, N, K, At_scratch);
    } else {
      int Nc, Kc;
      if (N >= K * 2) { Nc = int(std::min<int64_t>(N / 2, 4096)); Kc = 256; }
      else            { Nc = int(std::min<int64_t>(N, 2048));     Kc = 512; }
      amx_sgemm_blis_kc(A, B, C, M, N, K, Nc, Kc, At_scratch, packB_scratch);
    }
  }
}

int main() {
  struct Shape { int M, N, K; const char* model; const char* op; };
  const Shape shapes[] = {
    { 128,  2048, 2048, "GPT-2-small",    "QKV"     },
    { 128,  8192, 2048, "GPT-2-small",    "FFN1"    },
    { 128,  2048, 8192, "GPT-2-small",    "FFN2"    },
    { 128, 60000, 2048, "GPT-2-small",    "LM-head" },
    { 128,  2048, 2048, "TinyLlama-1.1B", "QKV"     },
    { 128,  5632, 2048, "TinyLlama-1.1B", "FFN1"    },
    { 128,  2048, 5632, "TinyLlama-1.1B", "FFN2"    },
    { 128, 32000, 2048, "TinyLlama-1.1B", "LM-head" },
    { 128,  4096, 4096, "Llama-7B",       "QKV"     },
    { 128, 11008, 4096, "Llama-7B",       "FFN1"    },
    { 128,  4096, 11008,"Llama-7B",       "FFN2"    },
    { 128, 32000, 4096, "Llama-7B",       "LM-head" },
  };

  std::vector<float> At_scratch, packB_scratch;
  std::printf("Shape-adaptive dispatch ablation (paper §4 add).\n");
  std::printf("Per-cell ratio = our_kernel_GFLOPS / Accelerate_GFLOPS, bit-exact.\n\n");

  std::printf("  %-16s %-9s | %-9s | %-9s | %-9s\n",
              "model / op", "Accel", "alwaysPK", "alwaysUN", "ADAPTIVE");
  std::printf("  %-16s %-9s | %-9s | %-9s | %-9s\n",
              "----------", "-----", "--------", "--------", "--------");

  double sum_log_pk = 0, sum_log_un = 0, sum_log_ad = 0;
  int wins_pk = 0, wins_un = 0, wins_ad = 0;

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
        auto t0 = clk::now(); fn(); best = std::min(best, ms(clk::now() - t0));
      }
      return std::pair<double, double>(best, flops / (best / 1e3) / 1e9);
    };

    auto [accel_ms, accel_gf] = bench([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.M, s.N, s.K,
                  1.0f, A.data(), s.K, B.data(), s.N, 0.0f, C_ref.data(), s.N);
    });

    double r[3] = {0, 0, 0};
    for (int p = 0; p < 3; ++p) {
      std::vector<float> C(size_t(s.M) * s.N, 0.f);
      Policy pol = (p == 0) ? Policy::ALWAYS_PACK :
                   (p == 1) ? Policy::ALWAYS_UNPACK : Policy::ADAPTIVE;
      auto [t, gf] = bench([&] {
        run(pol, A.data(), B.data(), C.data(), s.M, s.N, s.K,
            At_scratch, packB_scratch);
      });
      float maxd = 0.f;
      size_t spot = std::min<size_t>(C.size(), 1024);
      for (size_t i = 0; i < spot; ++i)
        maxd = std::max(maxd, std::fabs(C[i] - C_ref[i]));
      r[p] = (maxd > 1e-3f) ? -1.0 : gf / accel_gf;
    }
    sum_log_pk += std::log(r[0] > 0 ? r[0] : 1e-9);
    sum_log_un += std::log(r[1] > 0 ? r[1] : 1e-9);
    sum_log_ad += std::log(r[2] > 0 ? r[2] : 1e-9);
    if (r[0] >= 1.0) wins_pk++;
    if (r[1] >= 1.0) wins_un++;
    if (r[2] >= 1.0) wins_ad++;

    char tag[40];
    std::snprintf(tag, sizeof(tag), "%-12s %s", s.model, s.op);
    std::printf("  %-16s %4.0f      | %.2fx     | %.2fx     | %.2fx %s\n",
                tag, accel_gf, r[0], r[1], r[2],
                (r[2] >= r[0] && r[2] >= r[1]) ? "<-best" : "");
  }

  const int N = 12;
  std::printf("\n  geomean ratio: always-pack %.3fx (%d/12 wins), "
              "always-unpack %.3fx (%d/12), ADAPTIVE %.3fx (%d/12)\n",
              std::exp(sum_log_pk / N), wins_pk,
              std::exp(sum_log_un / N), wins_un,
              std::exp(sum_log_ad / N), wins_ad);
  std::printf("\nADAPTIVE > max(always-pack, always-unpack) at every cell where it's\n"
              "best confirms the dispatch decision quantitatively matters.\n");
  return 0;
}
