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

// AMX-aware decode dispatch (v2, Session 13). When enabled (default), a Gemm
// with a single output row (M == 1 — the autoregressive-decode shape: one
// token projected by a weight matrix) routes through Apple Accelerate's
// cblas_sgemv instead of cblas_sgemm. Session 12's microbench measured sgemv
// faster than single-row sgemm across GPT-2's projection shapes on M1, because
// a one-row GEMM can't fill the AMX systolic array. Toggleable for the paper's
// ablation (Table 2): off == the Session-11 sgemm baseline.
void SetGemvDecodeEnabled(bool on);
bool GemvDecodeEnabled();

}  // namespace rt
}  // namespace inferc
