#pragma once

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// ONNX Gather: take rows from `data` at positions `indices` along `axis`.
// data:    arbitrary dtype, shape [..., D, ...] where D is the size at `axis`
// indices: int64 (or int32), arbitrary shape
// output:  data dtype, shape data[..axis] + indices.shape + data[axis+1..]
//
// Used by DistilBERT for token + position embeddings (axis=0 lookup into
// a [vocab, hidden] table).
Tensor Gather(const Tensor& data, const Tensor& indices, int64_t axis);

}  // namespace rt
}  // namespace inferc
