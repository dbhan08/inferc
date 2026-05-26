#include "kernels/amx_gemm.h"

#include <vector>

#include "amx/aarch64.h"  // vendored corsix/dougallj AMX instruction macros

namespace inferc {
namespace rt {

// fp32 AMX outer-product GEMM. matrix-mode fma32: z[j][i] += x[i]*y[j], 16-wide.
// For f32, output row j of an outer product lands in Z register (4*j + base),
// base 0..3 → FOUR independent Z accumulator banks. We use that for ILP:
// four 16-col N sub-blocks accumulate into banks 0..3, so four independent
// fma32 issue per k instead of one dependent Z chain (the single-accumulator
// PoC ran at FMA *latency*, ~10x off Accelerate; 4-way ILP recovers ~1.5x).
//
//   X reg t = B[k][j0+16t : +16]   (contiguous B-row sub-block)
//   Y reg 0 = packed column A[i0:i0+16][k]
//   fma32 bank t: z[4j+t][i] += x_t[i]*y[j]  →  C[i0+j][j0+16t+i]
//
// NOTE (Session 21 characterization): this plateaus at ~0.13x Accelerate. The
// bottleneck is the in-order ldx→fma32 dependency, not memory — cache blocking
// (B-panel reuse) actually regressed it. Beating Accelerate needs software-
// pipelined overlapping tiles + masked outer products (thesis-scale). Kept as a
// correct, bit-exact AMX-programming artifact, not a production kernel.
//
// Requires M % 16 == 0; N handled in 64-blocks with a 16-wide tail.
namespace {
inline uint64_t Fma32Op(int zbase, int x_off_bytes, bool first) {
  return (static_cast<uint64_t>(zbase) << 20) |
         (static_cast<uint64_t>(x_off_bytes) << 10) |
         (first ? (1ULL << 27) : 0);
}
}  // namespace

void AmxSgemmF32(const float* A, const float* B, float* C,
                 int64_t M, int64_t N, int64_t K) {
  std::vector<float> packA(static_cast<size_t>(K) * 16);
  AMX_SET();
  for (int64_t i0 = 0; i0 < M; i0 += 16) {
    // Pack A[i0:i0+16][0:K] column-major: packA[k*16 + i] = A[(i0+i)*K + k].
    for (int64_t k = 0; k < K; ++k) {
      float* dst = packA.data() + k * 16;
      const float* col = A + i0 * K + k;
      for (int i = 0; i < 16; ++i) dst[i] = col[i * K];
    }
    const float* pa = packA.data();

    int64_t j0 = 0;
    for (; j0 + 64 <= N; j0 += 64) {         // 4-wide: banks 0..3 (ILP)
      for (int64_t k = 0; k < K; ++k) {
        const bool first = (k == 0);
        AMX_LDY(reinterpret_cast<uint64_t>(pa + k * 16));            // Y reg 0
        const float* brow = B + k * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 16) | (1ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56));
        AMX_LDX(reinterpret_cast<uint64_t>(brow + 48) | (3ULL << 56));
        AMX_FMA32(Fma32Op(0, 0,   first));
        AMX_FMA32(Fma32Op(1, 64,  first));
        AMX_FMA32(Fma32Op(2, 128, first));
        AMX_FMA32(Fma32Op(3, 192, first));
      }
      for (int t = 0; t < 4; ++t)
        for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0 + 16 * t) |
                  (static_cast<uint64_t>(4 * j + t) << 56));
    }
    for (; j0 < N; j0 += 16) {               // 16-wide N tail (bank 0)
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_LDY(reinterpret_cast<uint64_t>(pa + k * 16));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) |
                (static_cast<uint64_t>(4 * j) << 56));
    }
  }
  AMX_CLR();
}

}  // namespace rt
}  // namespace inferc
