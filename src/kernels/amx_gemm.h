#pragma once

#include <cstdint>

namespace inferc {
namespace rt {

// PROOF-OF-CONCEPT custom fp32 GEMM using Apple AMX instructions directly
// (undocumented ISA, via corsix/dougallj encodings — see third_party/amx).
// C[M,N] = A[M,K] * B[K,N], row-major. Requires M % 16 == 0 and N % 16 == 0.
//
// Goal: measure whether a hand-written AMX kernel can approach/beat Accelerate's
// cblas_sgemm on our cache-bound transformer shapes. This is a NAIVE kernel
// (16x16 tiles, packs A columns, no cache blocking / multi-accumulator ILP), so
// it is expected to be memory-bound and below Accelerate; the thesis-grade
// in-place masked-outer-product technique would be the next step if promising.
void AmxSgemmF32(const float* A, const float* B, float* C,
                 int64_t M, int64_t N, int64_t K);

}  // namespace rt
}  // namespace inferc
