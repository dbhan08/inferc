#pragma once

#include <vector>

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// Reshape: returns a tensor with the new shape, sharing the underlying bytes
// when the input is contiguous (zero-copy). Falls back to a copy otherwise.
// `new_shape` may contain a single -1 to be inferred from total element count.
Tensor Reshape(const Tensor& x, Shape new_shape);

// Transpose by `perm`. If perm is empty, reverses all dims. Always returns a
// physical copy in this implementation (we don't expose stride-only views to
// downstream kernels yet).
Tensor Transpose(const Tensor& x, const std::vector<int64_t>& perm);

// Concat along `axis`.
Tensor Concat(const std::vector<Tensor>& parts, int64_t axis);

// Slice with starts/ends/axes/steps along `axes` (default: 0..rank-1).
Tensor Slice(const Tensor& x,
             const std::vector<int64_t>& starts,
             const std::vector<int64_t>& ends,
             const std::vector<int64_t>& axes = {},
             const std::vector<int64_t>& steps = {});

// Insert size-1 dims at positions in `axes`.
Tensor Unsqueeze(const Tensor& x, const std::vector<int64_t>& axes);

// Drop size-1 dims at positions in `axes`. If `axes` empty, drop all 1-dims.
Tensor Squeeze(const Tensor& x, const std::vector<int64_t>& axes);

// Cast to a new dtype. Float ↔ Float ↔ Int conversions.
Tensor Cast(const Tensor& x, DType to);

}  // namespace rt
}  // namespace inferc
