#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ir/graph.h"

namespace inferc {
namespace rt {

// Tensor is a dtype + shape + strided view into byte storage.
//
// Storage is reference-counted (std::shared_ptr<uint8_t[]>) so View ops
// (Reshape, Transpose, Slice) can share the backing buffer cheaply. Most
// compute kernels assume contiguous row-major layout — call .Contiguous()
// to materialize a copy when stride-laden inputs reach a kernel that needs
// packed memory (e.g., sgemm).
//
// All sizes are element counts unless the name says "bytes".
class Tensor {
 public:
  Tensor() = default;
  Tensor(DType dtype, Shape shape);  // allocates owned, contiguous, zero-init
  static Tensor Zeros(DType dtype, Shape shape);
  // Allocate without zero-initializing — for kernels that fully overwrite their
  // output (skips a wasted memset; the dominant per-op overhead ORT avoids via
  // planned buffers). ONLY use when every element is written.
  static Tensor Uninit(DType dtype, Shape shape);
  static Tensor FromHostBytes(DType dtype, Shape shape, const void* src);
  static Tensor BorrowingView(DType dtype, Shape shape,
                              std::shared_ptr<uint8_t[]> storage,
                              size_t byte_offset, Shape strides);

  // Identity / metadata.
  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  const Shape& strides() const { return strides_; }
  int64_t numel() const;
  int64_t byte_size() const;     // numel * dtype_bytes (for *contiguous* view)
  int64_t rank() const { return static_cast<int64_t>(shape_.size()); }

  bool is_contiguous() const;

  // Raw byte pointer at the current view's start. Const overload too.
  uint8_t* bytes() { return storage_.get() + byte_offset_; }
  const uint8_t* bytes() const { return storage_.get() + byte_offset_; }

  // Typed pointer access — caller asserts dtype matches.
  template <typename T>
  T* data() { return reinterpret_cast<T*>(bytes()); }
  template <typename T>
  const T* data() const { return reinterpret_cast<const T*>(bytes()); }

  // Returns a contiguous-layout copy. If already contiguous, returns a copy
  // anyway (cheap-ish — caller wanted it). Used to "pack" before sgemm.
  Tensor Contiguous() const;

  // True equality of dtype + shape + (logically equal) values.
  // Allows arbitrary strides on either side.
  bool ApproximatelyEqualsFloat32(const Tensor& other, float tolerance) const;

  // Debug repr.
  std::string DebugString() const;

 private:
  DType dtype_ = DType::kUnknown;
  Shape shape_;
  Shape strides_;  // element strides (NOT byte strides). Same length as shape.
  std::shared_ptr<uint8_t[]> storage_;
  size_t byte_offset_ = 0;
};

// ---------------- Helpers ----------------

// Compute row-major (C-order) element strides for a contiguous shape.
Shape ContiguousStrides(const Shape& shape);

// Convert a multi-index (i0, i1, ..., iN-1) into a byte offset given strides
// and dtype. Strides are in elements; we multiply by dtype size at the end.
int64_t ByteOffset(const Shape& strides, const Shape& index, int64_t elem_bytes);

// Sequence iteration helper for non-contiguous tensors. Given a shape, yields
// each index lexicographically (row-major) by incrementing the rightmost dim
// first. Returns false when exhausted.
class IndexIterator {
 public:
  explicit IndexIterator(Shape shape);
  bool Next(Shape* out_index);
 private:
  Shape shape_;
  Shape current_;
  bool done_ = false;
  bool first_ = true;
};

}  // namespace rt
}  // namespace inferc
