// Measured end-to-end prefill GEMM time (replaces the FLOP-weighted
// composition of the paper's Table 6 with a wall-clock measurement).
//
// Runs the full prefill GEMM sequence of a transformer in layer order, S=128,
// fp32, and times it under three backends:
//   cblas   cblas_sgemm, re-packs the weight on every call
//   BNNS    BNNSMatMul,   descriptors+workspace reused, re-packs each call
//   ours    the pre-packed direct-AMX kernel (Nc=64/Kc=2048): every weight is
//           packed ONCE at "model load" (untimed), and each call runs only the
//           A-transpose + compute loop against the resident packed panels.
//
// Unlike the per-shape Table 3 harness, the A-transpose (the activation, which
// changes every call) is INSIDE the timed region here -- a real forward pays
// it per GEMM, shared only across the Q/K/V trio that consume the same input.
// So this measurement is strictly more conservative than composing Table 3.
//
// Per layer: Q,K,V projections, attention-output projection, FFN up, FFN down;
// then a single LM head. Attention SDPA (Q@K^T, softmax, value sum) is not a
// GEMM in this paper's sense and is omitted -- the reported quantity is prefill
// GEMM time, matching Table 6. Weights are reused across layers (does not
// affect timing); the chained activations are irrelevant to AMX timing, which
// is data-independent.
//
// Within a binary invocation each policy is timed over 15 passes after 2 warmup
// passes and the median taken; run the binary 11x (run_e2e_campaign.sh) and
// take the median of medians for the paper protocol.
//
// Build:
//   clang++ -O3 -std=c++17 -I third_party -framework Accelerate \
//     bench/amx/amx_e2e_measured.cc -o /tmp/amx_e2e

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}
static BNNSNDArrayDescriptor desc(void* data, BNNSDataType dt, int64_t rows, int64_t cols) {
  BNNSNDArrayDescriptor d; std::memset(&d, 0, sizeof(d));
  d.layout = BNNSDataLayoutRowMajorMatrix; d.size[0] = (size_t)cols; d.size[1] = (size_t)rows;
  d.data = data; d.data_type = dt; return d;
}

// === canonical pre-packed kernel pieces (verbatim from amx_prepack.cc) ======
static void pack_full(const float* B, float* dst, int64_t N, int64_t K, int64_t jc, int64_t Ncm) {
  for (int64_t k = 0; k < K; ++k) std::memcpy(&dst[k * Ncm], &B[k * N + jc], Ncm * sizeof(float));
}
static void compute_kc(const float* At, const float* pB, float* C, int64_t M,
                       int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
            AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool f = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + (pc + kk) * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, f)); AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f)); AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
  }
}
static void tail_cols(const float* At, const float* B, float* C, int64_t M, int64_t N, int64_t K, int64_t j0) {
  AMX_SET();
  for (; j0 + 16 <= N; j0 += 16)
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) | (uint64_t(4 * j) << 56));
    }
  AMX_CLR();
  for (; j0 < N; ++j0)
    for (int64_t i = 0; i < M; ++i) {
      float s = 0; for (int64_t k = 0; k < K; ++k) s += At[k * M + i] * B[k * N + j0];
      C[i * N + j0] = s;
    }
}

// A weight pre-packed once at load into Nc-wide [K x Ncm] panels.
struct Packed {
  int Nc = 64, Kc = 2048;
  int64_t N = 0, K = 0, covered = 0;
  std::vector<int64_t> jcs;
  std::vector<std::vector<float>> panels;  // panels[p] = [K x Ncm_p]
  const float* B = nullptr;                // raw weight, for the tail columns
};
static Packed prepack(const float* B, int64_t N, int64_t K, int Nc, int Kc) {
  Packed P; P.Nc = Nc; P.Kc = Kc; P.N = N; P.K = K; P.B = B;
  for (int64_t jc = 0; jc < N; jc += Nc) P.jcs.push_back(jc);
  const int nP = (int)P.jcs.size();
  P.panels.resize(nP);
  auto ncm = [&](int p) { return (std::min<int64_t>(Nc, N - P.jcs[p]) / 64) * 64; };
  for (int p = 0; p < nP; ++p) {
    int64_t Ncm = ncm(p);
    P.covered = std::max<int64_t>(P.covered, P.jcs[p] + Ncm);
    if (Ncm <= 0) continue;
    P.panels[p].resize(size_t(K) * Ncm);
    pack_full(B, P.panels[p].data(), N, K, P.jcs[p], Ncm);
  }
  return P;
}
// Compute C[M,N] = At^T . B from the pre-packed weight. At is the already-
// transposed activation [K x M]. (Transpose timing is the caller's, so it can
// be shared across the Q/K/V trio.)
static void ours_compute(const Packed& P, const float* At, float* C, int64_t M) {
  const int Nc = P.Nc, Kc = P.Kc; const int64_t N = P.N, K = P.K;
  const int nP = (int)P.jcs.size();
  const int64_t* jcp = P.jcs.data();
  const std::vector<std::vector<float>>* pan = &P.panels;
  dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
    int64_t Ncm = (std::min<int64_t>(Nc, N - jcp[w]) / 64) * 64;
    if (Ncm <= 0) return;
    AMX_SET(); compute_kc(At, (*pan)[w].data(), C, M, N, K, Kc, jcp[w], Ncm); AMX_CLR();
  });
  if (P.covered < N) tail_cols(At, P.B, C, M, N, K, P.covered);
}
static void transpose_A(const float* A, float* At, int64_t M, int64_t K) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
}

struct Model { const char* name; int H, F, V, L; };
static const Model models[] = {
  { "TinyLlama-1.1B", 2048, 5632, 32000, 22 },
  { "Llama-7B",       4096, 11008, 32000, 32 },
};

enum class Backend { CBLAS, BNNS, OURS };

int main() {
  const int S = 128, M = S;
  const int Nc = 64, Kc = 2048;

  std::printf("Measured end-to-end prefill GEMM time, S=%d, fp32. ours = pre-packed\n"
              "kernel (Nc=%d/Kc=%d), weights packed once at load; A-transpose timed.\n"
              "median of 15 passes (2 warmup) per policy.\n\n", S, Nc, Kc);

  for (const auto& m : models) {
    const int H = m.H, F = m.F, V = m.V, L = m.L;

    // Weights (one set, reused across layers -- timing is weight-data-independent).
    std::vector<float> Wq(size_t(H) * H), Wk(size_t(H) * H), Wv(size_t(H) * H),
        Wo(size_t(H) * H), W1(size_t(H) * F), W2(size_t(F) * H), Wlm(size_t(H) * V);
    auto fill = [](std::vector<float>& w) { for (size_t i = 0; i < w.size(); ++i) w[i] = float(i % 11) * 0.01f; };
    fill(Wq); fill(Wk); fill(Wv); fill(Wo); fill(W1); fill(W2); fill(Wlm);

    // Activation buffers.
    std::vector<float> X(size_t(S) * H), Q(size_t(S) * H), Kb(size_t(S) * H),
        Vb(size_t(S) * H), ao(size_t(S) * H), mid(size_t(S) * F), out(size_t(S) * H),
        logits(size_t(S) * V);
    for (size_t i = 0; i < X.size(); ++i) X[i] = float(i % 7) * 0.01f;

    // Transpose scratch (reused).
    std::vector<float> XtH(size_t(H) * M), XtF(size_t(F) * M);

    // --- pre-pack all weights once (untimed = model load) for the OURS path ---
    Packed pWq = prepack(Wq.data(), H, H, Nc, Kc);
    Packed pWk = prepack(Wk.data(), H, H, Nc, Kc);
    Packed pWv = prepack(Wv.data(), H, H, Nc, Kc);
    Packed pWo = prepack(Wo.data(), H, H, Nc, Kc);
    Packed pW1 = prepack(W1.data(), F, H, Nc, Kc);
    Packed pW2 = prepack(W2.data(), H, F, Nc, Kc);
    Packed pWlm = prepack(Wlm.data(), V, H, Nc, Kc);

    // --- BNNS descriptors + workspaces, one per distinct shape, reused -------
    auto mk_bnns = [&](const float* B, int64_t N, int64_t K, std::vector<char>& ws) {
      auto da = desc(nullptr, BNNSDataTypeFloat32, M, K);  // A set per call
      auto db = desc((void*)B, BNNSDataTypeFloat32, K, N);
      auto dc = desc(nullptr, BNNSDataTypeFloat32, M, N);  // C set per call
      size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
      ws.assign(wsz ? wsz : 1, 0);
    };
    std::vector<char> wsHH, wsHF, wsFH, wsHV;
    mk_bnns(Wq.data(), H, H, wsHH); mk_bnns(W1.data(), F, H, wsHF);
    mk_bnns(W2.data(), H, F, wsFH); mk_bnns(Wlm.data(), V, H, wsHV);

    auto bnns = [&](const float* A, const float* B, float* C, int64_t N, int64_t K, std::vector<char>& ws) {
      auto da = desc((void*)A, BNNSDataTypeFloat32, M, K);
      auto db = desc((void*)B, BNNSDataTypeFloat32, K, N);
      auto dc = desc((void*)C, BNNSDataTypeFloat32, M, N);
      BNNSMatMul(false, false, 1.0f, &da, &db, &dc, ws.empty() ? nullptr : ws.data(), nullptr);
    };
    auto cblas = [&](const float* A, const float* B, float* C, int64_t N, int64_t K) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    };

    // --- one full prefill forward (GEMM-only) under a given backend ----------
    auto forward = [&](Backend bk) {
      for (int layer = 0; layer < L; ++layer) {
        if (bk == Backend::OURS) {
          transpose_A(X.data(), XtH.data(), M, H);            // shared by Q,K,V
          ours_compute(pWq, XtH.data(), Q.data(),  M);
          ours_compute(pWk, XtH.data(), Kb.data(), M);
          ours_compute(pWv, XtH.data(), Vb.data(), M);
          transpose_A(Vb.data(), XtH.data(), M, H);
          ours_compute(pWo, XtH.data(), ao.data(), M);        // attn-out
          transpose_A(ao.data(), XtH.data(), M, H);
          ours_compute(pW1, XtH.data(), mid.data(), M);       // FFN1 (N=F,K=H)
          transpose_A(mid.data(), XtF.data(), M, F);
          ours_compute(pW2, XtF.data(), out.data(), M);       // FFN2 (N=H,K=F)
        } else if (bk == Backend::BNNS) {
          bnns(X.data(), Wq.data(), Q.data(),  H, H, wsHH);
          bnns(X.data(), Wk.data(), Kb.data(), H, H, wsHH);
          bnns(X.data(), Wv.data(), Vb.data(), H, H, wsHH);
          bnns(Vb.data(), Wo.data(), ao.data(), H, H, wsHH);
          bnns(ao.data(), W1.data(), mid.data(), F, H, wsHF);
          bnns(mid.data(), W2.data(), out.data(), H, F, wsFH);
        } else {
          cblas(X.data(), Wq.data(), Q.data(),  H, H);
          cblas(X.data(), Wk.data(), Kb.data(), H, H);
          cblas(X.data(), Wv.data(), Vb.data(), H, H);
          cblas(Vb.data(), Wo.data(), ao.data(), H, H);
          cblas(ao.data(), W1.data(), mid.data(), F, H);
          cblas(mid.data(), W2.data(), out.data(), H, F);
        }
        std::swap(X, out);
      }
      // LM head
      if (bk == Backend::OURS) {
        transpose_A(X.data(), XtH.data(), M, H);
        ours_compute(pWlm, XtH.data(), logits.data(), M);
      } else if (bk == Backend::BNNS) {
        bnns(X.data(), Wlm.data(), logits.data(), V, H, wsHV);
      } else {
        cblas(X.data(), Wlm.data(), logits.data(), V, H);
      }
    };

    auto bench = [&](Backend bk) {
      forward(bk); forward(bk);
      std::vector<double> ts;
      for (int i = 0; i < 15; ++i) {
        auto t0 = clk::now(); forward(bk); ts.push_back(ms(clk::now() - t0));
      }
      std::sort(ts.begin(), ts.end()); return ts[ts.size() / 2];
    };

    double tC = bench(Backend::CBLAS);
    double tB = bench(Backend::BNNS);
    double tO = bench(Backend::OURS);

    std::printf("=== %s (H=%d F=%d V=%d L=%d) ===\n", m.name, H, F, V, L);
    std::printf("  cblas   %8.2f ms   (%6.1f tok/s)\n", tC, S * 1e3 / tC);
    std::printf("  BNNS    %8.2f ms   (%6.1f tok/s)\n", tB, S * 1e3 / tB);
    std::printf("  ours    %8.2f ms   (%6.1f tok/s)\n", tO, S * 1e3 / tO);
    std::printf("  ours vs BNNS  %.3fx    ours vs cblas  %.3fx\n\n", tB / tO, tC / tO);
  }
  return 0;
}
