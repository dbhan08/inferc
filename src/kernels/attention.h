#pragma once

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// Fused multi-head self-attention. Consumes the Q/K/V projection outputs in
// [B, S, H] layout (H = num_heads * head_dim; head h occupies contiguous columns
// [h*head_dim, (h+1)*head_dim)) and produces the context in [B, S, H] — i.e. it
// replaces the whole reshape → transpose → QKᵀ → scale → mask → softmax → ·V →
// transpose → reshape block with one op, never materializing the transposed
// [B,nH,S,dH] tensors or the [B,nH,S,S] scores across the graph.
//
// Per (batch, head): scores = (1/sqrt(head_dim)) * Q_h · K_hᵀ via a strided
// cblas_sgemm (lda = H, so it reads the head's columns in place — AMX still
// runs the matmul). `mask_cond` (bool, broadcast to [B,nH,S,S]) selects `fill`
// (e.g. -inf) before a row softmax; context = softmax · V_h via a second strided
// sgemm written straight into the [B,S,H] output (ldc = H). Parallel over B*nH.
//
// nH is derived as H / head_dim. scale = 1/sqrt(head_dim).
Tensor FusedAttention(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& mask_cond, int64_t head_dim, float fill);

}  // namespace rt
}  // namespace inferc
