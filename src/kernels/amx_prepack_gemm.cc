#include "kernels/amx_prepack_gemm.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "amx/aarch64.h"

namespace inferc {
namespace rt {
namespace {

inline uint64_t Fma32Op(int z, int xo, bool first) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (first ? (1ULL << 27) : 0);
}
constexpr uint64_t kLdxPair = 1ULL << 62;

// Kc-cache-blocked compute from a pre-packed [K x Ncm] panel.
// See docs/PAPER_DRAFT.md section 3.1 for the blocking structure.
void ComputePanel(const float* At, const float* pB, float* C, int64_t M,
                  int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm) {
  for (int64_t pc = 0; pc < K; pc += Kc) {
    const int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        // Carry the partial accumulator from C for every K block after the
        // first. Without this the Z state leaks between tiles.
        if (!first_pc)
          for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 16; ++j)
              AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                      (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool f = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + (pc + kk) * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow) | (0ULL << 56) | kLdxPair);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | kLdxPair);
          AMX_FMA32(Fma32Op(0, 0, f));
          AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f));
          AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t)
          for (int j = 0; j < 16; ++j)
            AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                    (uint64_t(4 * j + t) << 56));
      }
  }
}

// Residual columns [j0, N): 16-wide AMX tiles + scalar remainder (reads B).
void TailCols(const float* At, const float* B, float* C, int64_t M, int64_t N,
              int64_t K, int64_t j0) {
  AMX_SET();
  for (; j0 + 16 <= N; j0 += 16)
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
  AMX_CLR();
  for (; j0 < N; ++j0)
    for (int64_t i = 0; i < M; ++i) {
      float s = 0;
      for (int64_t k = 0; k < K; ++k) s += At[k * M + i] * B[k * N + j0];
      C[i * N + j0] = s;
    }
}

}  // namespace

AmxPackedWeight AmxPackWeight(const float* B, int64_t N, int64_t K, int Nc, int Kc) {
  AmxPackedWeight W;
  W.N = N; W.K = K; W.Nc = Nc; W.Kc = Kc; W.B_tail = B;
  for (int64_t jc = 0; jc < N; jc += Nc) W.jc.push_back(jc);
  W.panels.resize(W.jc.size());
  for (size_t p = 0; p < W.jc.size(); ++p) {
    const int64_t Ncm = (std::min<int64_t>(Nc, N - W.jc[p]) / 64) * 64;
    if (Ncm <= 0) continue;
    W.covered = std::max<int64_t>(W.covered, W.jc[p] + Ncm);
    W.panels[p].resize(size_t(K) * Ncm);
    for (int64_t k = 0; k < K; ++k)
      std::memcpy(&W.panels[p][k * Ncm], &B[k * N + W.jc[p]], Ncm * sizeof(float));
  }
  return W;
}

void AmxPrepackedSgemm(const float* A, const AmxPackedWeight& W, float* C, int64_t M) {
  if (M % 16 != 0) throw std::invalid_argument("AmxPrepackedSgemm: M must be a multiple of 16");
  const int64_t N = W.N, K = W.K;
  // Transpose A once so a 16-row column slice is one contiguous LDY.
  std::vector<float> At(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
  const float* atp = At.data();
  const int nP = static_cast<int>(W.jc.size());
  const AmxPackedWeight* Wp = &W;
  dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
    const int64_t Ncm = (std::min<int64_t>(Wp->Nc, N - Wp->jc[w]) / 64) * 64;
    if (Ncm <= 0) return;
    AMX_SET();
    ComputePanel(atp, Wp->panels[w].data(), C, M, N, K, Wp->Kc, Wp->jc[w], Ncm);
    AMX_CLR();
  });
  if (W.covered < N) TailCols(atp, W.B_tail, C, M, N, K, W.covered);
}

}  // namespace rt
}  // namespace inferc
