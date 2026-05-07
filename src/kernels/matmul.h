#pragma once

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// MatMul with numpy semantics:
//   - Both inputs are at least 2D (or 1D with implicit prepend/append 1).
//   - Last two dims of each input do the matmul: [..., M, K] x [..., K, N] -> [..., M, N]
//   - Leading dims broadcast right-aligned.
// fp32 only for v1. Inputs are made contiguous internally before sgemm.
//
// The actual matmul routes through Apple Accelerate's cblas_sgemm (which on
// Apple Silicon dispatches to AMX). That's the path that earns this project
// its perf-engineering claim.
Tensor MatMul(const Tensor& a, const Tensor& b);

// Gemm: alpha*op(A)*op(B) + beta*C, where op(.) is optional transpose.
// 2D only. Used by the classifier head's two Gemm nodes.
Tensor Gemm(const Tensor& a, const Tensor& b, const Tensor* c,
            float alpha = 1.0f, float beta = 1.0f,
            bool trans_a = false, bool trans_b = false);

}  // namespace rt
}  // namespace inferc
