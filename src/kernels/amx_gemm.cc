#include "kernels/amx_gemm.h"

#include <vector>

#include "amx/aarch64.h"  // vendored corsix/dougallj AMX instruction macros

namespace inferc {
namespace rt {

// fp32 AMX outer-product GEMM. AMX matrix-mode fma32 computes z[j][i] += x[i]*y[j]
// over 16-wide X and Y vectors, with the j-th output row landing in Z register
// 4*j ("one row from each four" for f32). We set:
//   X = B[k][j0:j0+16]  (a contiguous row of B — no packing)
//   Y = packed column A[i0:i0+16][k]  (A packed column-major so it's contiguous)
// so z[j][i] = sum_k A[i0+j][k] * B[k][j0+i] = C[i0+j][j0+i] — Z row 4j is C row
// (i0+j), which STZ writes straight out.
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
    for (int64_t j0 = 0; j0 < N; j0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));      // X reg 0 = B row
        AMX_LDY(reinterpret_cast<uint64_t>(packA.data() + k * 16));  // Y reg 0 = A col
        const uint64_t fop = (k == 0) ? (1ULL << 27) : 0;  // first: z=x*y; else z+=x*y
        AMX_FMA32(fop);
      }
      for (int j = 0; j < 16; ++j) {  // Z row 4j == C row (i0+j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) |
                (static_cast<uint64_t>(4 * j) << 56));
      }
    }
  }
  AMX_CLR();
}

}  // namespace rt
}  // namespace inferc
