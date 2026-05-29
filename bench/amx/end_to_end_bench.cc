// Measure end-to-end prefill speedup (Q5.1 Q4.3 follow-up).
//
// Simulates the GEMM work of a full transformer block at prefill: per layer
// the 4 GEMMs that dominate compute (Q-proj, K-proj, V-proj, attention-output
// projection, FFN1 up, FFN2 down), iterated over L layers, then a single LM
// head at the end of the model. Attention SDPA (Q@K^T, softmax @ V) is not a
// GEMM in our shape sense and we don't claim it here, so it's omitted; we
// report end-to-end prefill *GEMM* time only.
//
// Two policies measured per model:
//   (a) ALL_ACCELERATE  — every GEMM goes to cblas_sgemm
//   (b) PER_GEMM_DISPATCH — our kernel for QKV/attn-out/FFN-down,
//                          Accelerate for FFN1/LM head (matches paper §1)
//
// Built single-thread Accelerate (VECLIB_MAXIMUM_THREADS=1) so the ratio is
// the apples-to-apples speedup a serving stack would actually see if it
// dispatched per-GEMM at the BLAS layer.
//
// Reports for each model: total prefill GEMM time under each policy, the
// per-layer breakdown, and the end-to-end speedup ratio. Compare against
// the paper's derived 1.29× = 29% claim from per-GEMM ratios.

#include <Accelerate/Accelerate.h>

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

// === BLIS+Kc kernel (verbatim from llama_shapes_bench.cc) ===
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

// Convenience wrappers
enum class Backend { ACCEL, OURS };
static void gemm(Backend b, const float* A, const float* B, float* C,
                 int M, int N, int K,
                 std::vector<float>& At_scratch,
                 std::vector<float>& packB_scratch) {
  if (b == Backend::ACCEL) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                A, K, B, N, 0.0f, C, N);
  } else {
    // Same auto Nc/Kc as paper §3.3
    int Nc, Kc;
    if (N >= K * 2) { Nc = int(std::min<int64_t>(N / 2, 4096)); Kc = 256; }
    else            { Nc = int(std::min<int64_t>(N, 2048));     Kc = 512; }
    amx_sgemm_blis_kc(A, B, C, M, N, K, Nc, Kc, At_scratch, packB_scratch);
  }
}

// Model dimensions
struct Model {
  const char* name;
  int H;       // hidden dim
  int F;       // FFN intermediate dim
  int V;       // vocab size
  int L;       // num layers
};

// Three real LLM scales.
static const Model models[] = {
  { "GPT-2-small",    768,  3072, 50257, 12 },
  { "TinyLlama-1.1B", 2048, 5632, 32000, 22 },
  { "Llama-7B",       4096, 11008, 32000, 32 },
};

// PER_GEMM_DISPATCH heuristic: which backend handles each shape class.
//   QKV / attention-output (square H×H @ H×H): OURS (paper says we beat 1.51×)
//   FFN1 up-proj (H×4F):                       ACCEL (paper says we lose 0.91×)
//   FFN2 down-proj (4F×H):                     OURS (paper says we beat 1.45×)
//   LM head (H×V):                             ACCEL (paper says we lose 0.62×)
static Backend dispatch_for(const char* op) {
  if (std::strcmp(op, "Q-proj")   == 0) return Backend::OURS;
  if (std::strcmp(op, "K-proj")   == 0) return Backend::OURS;
  if (std::strcmp(op, "V-proj")   == 0) return Backend::OURS;
  if (std::strcmp(op, "attn-out") == 0) return Backend::OURS;
  if (std::strcmp(op, "FFN1")     == 0) return Backend::ACCEL;
  if (std::strcmp(op, "FFN2")     == 0) return Backend::OURS;
  if (std::strcmp(op, "LM-head")  == 0) return Backend::ACCEL;
  return Backend::ACCEL;
}

int main() {
  const int S = 128;
  std::vector<float> At_scratch;
  std::vector<float> packB_scratch;

  std::printf("End-to-end prefill GEMM bench (S=%d single thread).\n", S);
  std::printf("Per-layer GEMMs: Q-proj, K-proj, V-proj, attn-out, FFN1, FFN2.\n");
  std::printf("Plus one LM head at the end. Repeated for L layers per model.\n\n");

  for (const auto& m : models) {
    std::printf("=== %s (H=%d, F=%d, V=%d, L=%d) ===\n",
                m.name, m.H, m.F, m.V, m.L);

    // Persistent activation buffers (so we don't re-allocate per layer)
    std::vector<float> X (size_t(S) * m.H);              // current layer input
    std::vector<float> Q (size_t(S) * m.H);
    std::vector<float> K_(size_t(S) * m.H);
    std::vector<float> V (size_t(S) * m.H);
    std::vector<float> attn_out(size_t(S) * m.H);
    std::vector<float> mid(size_t(S) * m.F);             // FFN intermediate
    std::vector<float> out(size_t(S) * m.H);
    std::vector<float> logits(size_t(S) * m.V);

    // Weights — one set per kind (we reuse across layers; doesn't affect timing)
    std::vector<float> Wq(size_t(m.H) * m.H);
    std::vector<float> Wk(size_t(m.H) * m.H);
    std::vector<float> Wv(size_t(m.H) * m.H);
    std::vector<float> Wo(size_t(m.H) * m.H);
    std::vector<float> W1(size_t(m.H) * m.F);
    std::vector<float> W2(size_t(m.F) * m.H);
    std::vector<float> Wlm(size_t(m.H) * m.V);
    for (size_t i = 0; i < X.size();   ++i) X[i]   = float(i % 7)  * 0.01f;
    for (size_t i = 0; i < Wq.size();  ++i) Wq[i]  = float(i % 11) * 0.01f;
    for (size_t i = 0; i < Wk.size();  ++i) Wk[i]  = float(i % 11) * 0.01f;
    for (size_t i = 0; i < Wv.size();  ++i) Wv[i]  = float(i % 11) * 0.01f;
    for (size_t i = 0; i < Wo.size();  ++i) Wo[i]  = float(i % 11) * 0.01f;
    for (size_t i = 0; i < W1.size();  ++i) W1[i]  = float(i % 11) * 0.01f;
    for (size_t i = 0; i < W2.size();  ++i) W2[i]  = float(i % 11) * 0.01f;
    for (size_t i = 0; i < Wlm.size(); ++i) Wlm[i] = float(i % 11) * 0.01f;

    auto one_pass = [&](Backend qkv_b, Backend attn_b, Backend ffn1_b,
                        Backend ffn2_b, Backend lm_b) {
      for (int layer = 0; layer < m.L; ++layer) {
        gemm(qkv_b , X.data(),        Wq.data(), Q .data(), S, m.H, m.H, At_scratch, packB_scratch);
        gemm(qkv_b , X.data(),        Wk.data(), K_.data(), S, m.H, m.H, At_scratch, packB_scratch);
        gemm(qkv_b , X.data(),        Wv.data(), V .data(), S, m.H, m.H, At_scratch, packB_scratch);
        // (attention SDPA omitted — not a GEMM in our shape sense)
        gemm(attn_b, V.data(),        Wo.data(), attn_out.data(), S, m.H, m.H, At_scratch, packB_scratch);
        gemm(ffn1_b, attn_out.data(), W1.data(), mid.data(),     S, m.F, m.H, At_scratch, packB_scratch);
        gemm(ffn2_b, mid.data(),      W2.data(), out.data(),     S, m.H, m.F, At_scratch, packB_scratch);
        std::swap(X, out);
      }
      gemm(lm_b, X.data(), Wlm.data(), logits.data(), S, m.V, m.H, At_scratch, packB_scratch);
    };

    auto bench = [&](auto fn) {
      fn(); fn();
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        auto t0 = clk::now(); fn(); best = std::min(best, ms(clk::now() - t0));
      }
      return best;
    };

    double t_all_accel = bench([&] {
      one_pass(Backend::ACCEL, Backend::ACCEL, Backend::ACCEL,
               Backend::ACCEL, Backend::ACCEL);
    });

    double t_dispatch = bench([&] {
      one_pass(dispatch_for("Q-proj"),
               dispatch_for("attn-out"),
               dispatch_for("FFN1"),
               dispatch_for("FFN2"),
               dispatch_for("LM-head"));
    });

    double ratio = t_all_accel / t_dispatch;
    std::printf("  all-Accelerate :      %7.2f ms\n", t_all_accel);
    std::printf("  per-GEMM dispatch:    %7.2f ms\n", t_dispatch);
    std::printf("  end-to-end speedup:   %.3f×   (%.1f%% faster)\n",
                ratio, (ratio - 1) * 100);
    std::printf("\n");
  }

  std::printf("Paper §1 derives 1.29×. Numbers above are the MEASURED end-to-end\n"
              "speedup from per-GEMM dispatch, single thread, S=128 prefill.\n");
  return 0;
}
