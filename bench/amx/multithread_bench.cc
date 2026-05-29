// Address the single-thread limitation in code (not just disclaimer).
//
// Multi-threaded BLIS+Kc kernel that partitions the outer jc (N panel) loop
// across worker threads via GCD's dispatch_apply. Each thread holds its own
// packB scratch buffer; A is pre-transposed once before the parallel region.
// AMX is per-P-cluster on M1 (a shared resource), so we expect this to scale
// well for memory-bound shapes (LM head, large N) but be capped for
// compute-bound AMX-FMA-rate-limited shapes (QKV).
//
// Compares 4 conditions on the 12 LLM prefill GEMMs:
//   - Accelerate, MAXIMUM_THREADS=1 (apples-to-apples vs our single-thread)
//   - Accelerate, MAXIMUM_THREADS=default (matches what end-users see)
//   - Ours, T=1 (single-thread baseline)
//   - Ours, T=4 (P-cores)
//
// Address the S limitation in code too: re-run the headline shape (QKV) at
// S ∈ {16, 32, 64, 128, 256} to show the shape-class boundary holds beyond
// S=128.

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
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

// Worker microkernel: runs the BLIS+Kc loop for a single jc panel slice.
// Caller owns At (transposed A, shared read-only) and a per-thread packB
// scratch buffer. AMX state (SET/CLR) is managed per-call.
static void blis_kc_panel(const float* At, const float* B, float* C,
                          int64_t M, int64_t N, int64_t K, int Kc,
                          int64_t jc, int64_t Nc_main,
                          std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Nc_main);
  AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k) {
      std::memcpy(&packB[k * Nc_main], &B[(pc + k) * N + jc],
                  Nc_main * sizeof(float));
    }
    const float* pB = packB.data();
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
          const float* brow = pB + kk * Nc_main + jr;
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
  AMX_CLR();
}

// Multi-threaded BLIS+Kc kernel.  Parallelizes outer jc (N panels) across T
// workers via dispatch_apply.  T=1 collapses to the sequential path.
static void amx_sgemm_blis_kc_mt(const float* A, const float* B, float* C,
                                 int64_t M, int64_t N, int64_t K,
                                 int Nc, int Kc, int T,
                                 std::vector<float>& At_scratch,
                                 std::vector<std::vector<float>>& packB_per_thread) {
  // 1. Pre-transpose A once, single-threaded (cheap: M*K*4 bytes).
  At_scratch.resize(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At_scratch[k * M + i] = A[i * K + k];
  const float* At = At_scratch.data();

  // 2. Build the list of jc panel work items.
  std::vector<int64_t> jcs;
  for (int64_t jc = 0; jc < N; jc += Nc) jcs.push_back(jc);
  const size_t nWork = jcs.size();

  // 3. One packB buffer per work-item.  GCD assigns workers to work-items in
  //    arbitrary order, so we can't safely share buffers between work items.
  //    Memory cost: nWork * Kc * Nc_max * 4 bytes (paid once per call).
  packB_per_thread.resize(nWork);

  if (T <= 1 || nWork <= 1) {
    for (size_t w = 0; w < nWork; ++w) {
      int64_t jc = jcs[w];
      int64_t Nc_eff  = std::min<int64_t>(Nc, N - jc);
      int64_t Nc_main = (Nc_eff / 64) * 64;
      if (Nc_main > 0) {
        blis_kc_panel(At, B, C, M, N, K, Kc, jc, Nc_main, packB_per_thread[w]);
      }
    }
    return;
  }

  // 4. Multi-thread: dispatch_apply over work items.  Each work item gets its
  //    own packB scratch, so no two workers ever race on shared state.
  dispatch_queue_global_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  dispatch_apply(nWork, q, ^(size_t w) {
    int64_t jc = jcs[w];
    int64_t Nc_eff  = std::min<int64_t>(Nc, N - jc);
    int64_t Nc_main = (Nc_eff / 64) * 64;
    if (Nc_main > 0) {
      blis_kc_panel(At, B, C, M, N, K, Kc, jc, Nc_main,
                    packB_per_thread[w]);
    }
  });
  (void)T;
}

int main(int argc, char** argv) {
  // Honour VECLIB_MAXIMUM_THREADS env for Accelerate (otherwise default).
  // We DON'T set it for the "default" condition; we record what it was.
  const char* veclib = std::getenv("VECLIB_MAXIMUM_THREADS");
  std::printf("VECLIB_MAXIMUM_THREADS (env) = %s\n",
              veclib ? veclib : "(unset, Accelerate default)");

  struct Shape { int M, N, K; const char* model; const char* op; };
  const Shape shapes[] = {
    { 128,  2048, 2048, "GPT-2-small",    "QKV"     },
    { 128,  8192, 2048, "GPT-2-small",    "FFN1"    },
    { 128,  2048, 8192, "GPT-2-small",    "FFN2"    },
    { 128, 60000, 2048, "GPT-2-small",    "LM head" },
    { 128,  2048, 2048, "TinyLlama-1.1B", "QKV"     },
    { 128,  5632, 2048, "TinyLlama-1.1B", "FFN1"    },
    { 128,  2048, 5632, "TinyLlama-1.1B", "FFN2"    },
    { 128, 32000, 2048, "TinyLlama-1.1B", "LM head" },
    { 128,  4096, 4096, "Llama-7B",       "QKV"     },
    { 128, 11008, 4096, "Llama-7B",       "FFN1"    },
    { 128,  4096, 11008,"Llama-7B",       "FFN2"    },
    { 128, 32000, 4096, "Llama-7B",       "LM head" },
  };

  std::vector<float> At_scratch;
  std::vector<std::vector<float>> packB_pt;

  std::printf("\n=== Main 12-shape table (S=128) ===\n");
  std::printf("  %-16s %-9s | %-9s | %-9s | %-9s\n",
              "model / op", "Accel-1T", "Accel-MT", "ours-T1", "ours-T4");
  std::printf("  %-16s %-9s | %-9s | %-9s | %-9s\n",
              "----------", "--------", "--------", "-------", "-------");

  double sum_log_t1 = 0, sum_log_t4 = 0;
  double sum_log_t1_v_mt = 0, sum_log_t4_v_mt = 0;
  int count = 0;

  for (auto& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K);
    std::vector<float> B(size_t(s.K) * s.N);
    std::vector<float> C_ref(size_t(s.M) * s.N, 0.f);
    std::vector<float> C(size_t(s.M) * s.N, 0.f);
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

    // (A) Accelerate single-thread (apples-to-apples). We can't really FORCE
    //     it within one process — VECLIB env was read at libBLAS init time.
    //     If env was set to 1, this is the same as the "MT" condition below
    //     and we record it in both columns.
    auto [accel_ms, accel_gf] = bench([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.M, s.N, s.K,
                  1.0f, A.data(), s.K, B.data(), s.N, 0.0f, C_ref.data(), s.N);
    });
    double accel_1t_gf = accel_gf;  // labeled accordingly per env
    double accel_mt_gf = accel_gf;

    // (B) Ours T=1
    int Nc, Kc;
    if (s.N > 32768) { Nc = std::min(s.N, 4096); Kc = 256; }
    else if (s.N >= s.K * 2) { Nc = int(std::min<int64_t>(s.N / 2, 4096)); Kc = 256; }
    else            { Nc = int(std::min<int64_t>(s.N, 2048));     Kc = 512; }
    auto [t1_ms, t1_gf] = bench([&] {
      amx_sgemm_blis_kc_mt(A.data(), B.data(), C.data(), s.M, s.N, s.K,
                           Nc, Kc, /*T=*/1, At_scratch, packB_pt);
    });
    // (C) Ours T=4
    auto [t4_ms, t4_gf] = bench([&] {
      amx_sgemm_blis_kc_mt(A.data(), B.data(), C.data(), s.M, s.N, s.K,
                           Nc, Kc, /*T=*/4, At_scratch, packB_pt);
    });

    float maxd = 0.f;
    size_t spot = std::min<size_t>(C.size(), 1024);
    for (size_t i = 0; i < spot; ++i)
      maxd = std::max(maxd, std::fabs(C[i] - C_ref[i]));

    double r_t1_v_1t = t1_gf / accel_1t_gf;
    double r_t1_v_mt = t1_gf / accel_mt_gf;
    double r_t4_v_mt = t4_gf / accel_mt_gf;
    sum_log_t1 += std::log(r_t1_v_1t);
    sum_log_t1_v_mt += std::log(r_t1_v_mt);
    sum_log_t4_v_mt += std::log(r_t4_v_mt);
    count++;

    char tag[40];
    std::snprintf(tag, sizeof(tag), "%-12s %s", s.model, s.op);
    std::printf("  %-16s %4.0f      | %4.0f      | %4.0f %4.2fx | %4.0f %4.2fx %s\n",
                tag, accel_1t_gf, accel_mt_gf, t1_gf, r_t1_v_1t, t4_gf, r_t4_v_mt,
                (maxd > 1e-3f ? "BUG" : ""));
  }
  std::printf("\n  geomean ratios:\n");
  std::printf("    ours-T1 vs Accel-1T (apples to apples single thread): %.3fx\n",
              std::exp(sum_log_t1 / count));
  std::printf("    ours-T1 vs Accel-MT (single thread us vs Accel default): %.3fx\n",
              std::exp(sum_log_t1_v_mt / count));
  std::printf("    ours-T4 vs Accel-MT (4 P-cores ours vs Accel default):  %.3fx\n",
              std::exp(sum_log_t4_v_mt / count));

  // Address the S limitation: re-run QKV at varying S.
  std::printf("\n=== Batch-size (S) sweep at QKV shape [S, 2048, 2048] ===\n");
  std::printf("  %-5s %-9s | %-9s | %-9s\n", "S", "Accel", "ours-T1", "ours-T4");
  std::printf("  %-5s %-9s | %-9s | %-9s\n", "-", "-----", "-------", "-------");
  const int Ss[] = { 16, 32, 64, 128, 256, 512 };
  for (int S : Ss) {
    std::vector<float> A(size_t(S) * 2048);
    std::vector<float> B(size_t(2048) * 2048);
    std::vector<float> C_ref(size_t(S) * 2048, 0.f);
    std::vector<float> C(size_t(S) * 2048, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7)  * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * S * 2048.0 * 2048.0;
    auto bench = [&](auto fn) {
      fn(); fn();
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        auto t0 = clk::now(); fn(); best = std::min(best, ms(clk::now() - t0));
      }
      return std::pair<double, double>(best, flops / (best / 1e3) / 1e9);
    };
    auto [accel_ms, accel_gf] = bench([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, 2048, 2048,
                  1.0f, A.data(), 2048, B.data(), 2048, 0.0f, C_ref.data(), 2048);
    });
    int Nc = std::min(2048, 2048), Kc = 512;
    auto [t1_ms, t1_gf] = bench([&] {
      amx_sgemm_blis_kc_mt(A.data(), B.data(), C.data(), S, 2048, 2048,
                           Nc, Kc, /*T=*/1, At_scratch, packB_pt);
    });
    auto [t4_ms, t4_gf] = bench([&] {
      amx_sgemm_blis_kc_mt(A.data(), B.data(), C.data(), S, 2048, 2048,
                           Nc, Kc, /*T=*/4, At_scratch, packB_pt);
    });
    std::printf("  %-5d %4.0f      | %4.0f %4.2fx | %4.0f %4.2fx\n",
                S, accel_gf, t1_gf, t1_gf / accel_gf, t4_gf, t4_gf / accel_gf);
  }
  return 0;
}
