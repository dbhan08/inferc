#pragma once

#include <cstdint>
#include <vector>

namespace inferc {
namespace rt {

// Pre-packed-weight fp32 GEMM on the Apple M1 AMX coprocessor.
//
// This is the kernel evaluated in docs/PAPER_DRAFT.md ("Above the Inner
// Loop"): C[M,N] = A[M,K] * B[K,N], row-major, with B (the model weight)
// packed ONCE into [K x Nc] column panels, and the per-call compute loop
// parallelised over those panels with Grand Central Dispatch so that both
// AMX blocks (P-cluster and E-cluster) are engaged.
//
// Numerics: the per-call arithmetic accumulates the same fp32 products in
// the same k order as Accelerate's sgemm, so the output is bit-identical to
// cblas_sgemm at the shapes the paper evaluates. tests/amx_prepack_test.cc
// enforces max-abs-diff == 0 over the FULL output matrix.
//
// Constraints: M % 16 == 0. Any N and K. Columns beyond the last full
// 64-wide panel are handled by a 16-wide AMX tail plus a scalar remainder.

struct AmxPackedWeight {
  int64_t N = 0, K = 0;
  int Nc = 64;                 // column-panel width (multi-thread granularity)
  int Kc = 2048;               // K-block depth (Z-accumulator reload period)
  std::vector<int64_t> jc;     // panel start columns
  std::vector<std::vector<float>> panels;  // one [K x Ncm] buffer per panel
  int64_t covered = 0;         // first column not covered by a 64-wide panel
  const float* B_tail = nullptr;  // borrowed pointer to B for the tail columns
};

// Pack B[K,N] (row-major) once. `B` must stay alive for the lifetime of the
// returned object (the tail columns, if any, read it directly).
AmxPackedWeight AmxPackWeight(const float* B, int64_t N, int64_t K,
                              int Nc = 64, int Kc = 2048);

// C[M,N] = A[M,K] * packed(B). A row-major with leading dimension K.
void AmxPrepackedSgemm(const float* A, const AmxPackedWeight& W, float* C,
                       int64_t M);

}  // namespace rt
}  // namespace inferc
