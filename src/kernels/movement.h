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

// Expand (broadcast) `x` to the given target shape. Numpy-style: each dim
// of x must either match the target or be 1.
Tensor Expand(const Tensor& x, const Shape& target);

// Produce the shape of `x` as an int64 1D tensor. Convenience for the
// executor's handling of ONNX `Shape` ops.
Tensor ShapeOf(const Tensor& x);

// ConstantOfShape: returns a tensor of the given shape, every element set to
// `fill_value` (broadcast from the one-element pattern in `fill_bytes`).
// `fill_bytes.size()` must equal `DTypeBytes(dtype)`; if empty, fills with zeros.
Tensor ConstantOfShape(const Shape& shape, DType dtype,
                       const std::vector<uint8_t>& fill_bytes);

// Split: slice `x` along `axis` into N pieces of sizes `split_sizes[i]`.
// Returns one tensor per slice. sum(split_sizes) must equal x.shape()[axis].
std::vector<Tensor> Split(const Tensor& x, int64_t axis,
                          const std::vector<int64_t>& split_sizes);

// Range (int64): produces a 1D int64 tensor [start, start+delta, ..., < limit).
Tensor RangeI64(int64_t start, int64_t limit, int64_t delta);
// Range (float32): same, but float32 output.
Tensor RangeF32(float start, float limit, float delta);

}  // namespace rt
}  // namespace inferc
