#pragma once

#include <vector>

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). Matches DistilBERT export.
Tensor Gelu(const Tensor& x);

// Numerically stable softmax along `axis` (negative axes wrap from the right).
// Default axis = -1 (last dim), matching ONNX Softmax-13+ semantics.
Tensor Softmax(const Tensor& x, int64_t axis = -1);

// Fused LayerNorm along the last `normalized_dims` axes:
//   out = scale * (x - mean(x)) / sqrt(var(x) + eps) + bias
// `scale` and `bias` are 1D tensors of length equal to product of last
// `normalized_dims` dims. Used for BERT-style LN where normalized_dims=1
// (over the hidden dim).
Tensor LayerNorm(const Tensor& x, const Tensor& scale, const Tensor& bias,
                 float eps = 1e-12f, int normalized_dims = 1);

// ReduceMean along the given axes. If `keepdims`, axes are kept as size-1 dims.
Tensor ReduceMean(const Tensor& x, const std::vector<int64_t>& axes,
                  bool keepdims);

}  // namespace rt
}  // namespace inferc
