#pragma once

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// FusedMatMulAddGELU — computes  Y = GELU(X @ W + bias)  in one kernel.
//
// Shape constraints (validated in the fusion pass):
//   X:    [..., M, K]   — any leading "batch" dims; row-major contiguous
//   W:    [K, N]        — 2D weight matrix
//   bias: [N]           — 1D, broadcast along all leading dims of X@W
//
// Output: [..., M, N], same layout as X.
//
// Implementation: one `cblas_sgemm` (Apple Accelerate, AMX-backed) computes
// X @ W into the output buffer; then a single sweep over the output buffer
// adds the per-column bias and applies exact GELU per element. Saves four
// full passes over (M*N) memory compared to separate MatMul + Add + Div +
// Erf + Add + Mul + Mul kernels.
Tensor FusedMatMulAddGELU(const Tensor& x, const Tensor& w, const Tensor& bias);

}  // namespace rt
}  // namespace inferc
