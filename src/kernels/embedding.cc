#include "kernels/embedding.h"

#include <cstring>
#include <stdexcept>

namespace inferc {
namespace rt {

Tensor Gather(const Tensor& data_in, const Tensor& idx_in, int64_t axis) {
  Tensor data = data_in.Contiguous();
  Tensor idx = idx_in.Contiguous();
  const int64_t data_rank = data.rank();
  if (axis < 0) axis += data_rank;
  if (axis < 0 || axis >= data_rank) {
    throw std::runtime_error("Gather: axis out of range");
  }
  const int64_t axis_dim = data.shape()[axis];

  // Output shape: data[..axis] + idx.shape + data[axis+1..]
  Shape out_shape;
  out_shape.reserve(data_rank - 1 + idx.rank());
  for (int64_t i = 0; i < axis; ++i) out_shape.push_back(data.shape()[i]);
  for (auto d : idx.shape()) out_shape.push_back(d);
  for (int64_t i = axis + 1; i < data_rank; ++i) {
    out_shape.push_back(data.shape()[i]);
  }
  Tensor out = Tensor::Zeros(data.dtype(), out_shape);

  // Slice sizes.
  // outer = product of data[0..axis), per_slice = product of data[axis+1..),
  // both in element counts. axis_dim is the lookup dimension.
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= data.shape()[i];
  int64_t per_slice = 1;
  for (int64_t i = axis + 1; i < data_rank; ++i) per_slice *= data.shape()[i];
  const int64_t elem_bytes = DTypeBytes(data.dtype());
  const int64_t slice_bytes = per_slice * elem_bytes;

  const uint8_t* data_p = data.bytes();
  uint8_t* out_p = out.bytes();

  // Read indices as int64. Tolerate int32 by casting up.
  std::vector<int64_t> indices;
  const int64_t n_idx = idx.numel();
  indices.reserve(n_idx);
  if (idx.dtype() == DType::kInt64) {
    const int64_t* p = idx.data<int64_t>();
    indices.assign(p, p + n_idx);
  } else if (idx.dtype() == DType::kInt32) {
    const int32_t* p = idx.data<int32_t>();
    for (int64_t i = 0; i < n_idx; ++i) indices.push_back(p[i]);
  } else {
    throw std::runtime_error("Gather: indices must be int32 or int64");
  }

  // For each (outer_pos, idx_pos), copy `slice_bytes` bytes.
  // Output layout: [outer, n_idx, per_slice] (linearly).
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t k = 0; k < n_idx; ++k) {
      int64_t i = indices[k];
      if (i < 0) i += axis_dim;
      if (i < 0 || i >= axis_dim) {
        throw std::runtime_error("Gather: index out of range");
      }
      const uint8_t* src = data_p +
          ((o * axis_dim + i) * per_slice) * elem_bytes;
      uint8_t* dst = out_p + ((o * n_idx + k) * per_slice) * elem_bytes;
      std::memcpy(dst, src, slice_bytes);
    }
  }
  return out;
}

}  // namespace rt
}  // namespace inferc
