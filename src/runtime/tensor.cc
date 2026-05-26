#include "runtime/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace inferc {
namespace rt {

Shape ContiguousStrides(const Shape& shape) {
  Shape s(shape.size(), 1);
  if (shape.empty()) return s;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    s[i] = s[i + 1] * shape[i + 1];
  }
  return s;
}

int64_t ByteOffset(const Shape& strides, const Shape& index, int64_t elem_bytes) {
  int64_t off = 0;
  for (size_t i = 0; i < strides.size(); ++i) off += strides[i] * index[i];
  return off * elem_bytes;
}

IndexIterator::IndexIterator(Shape shape) : shape_(std::move(shape)),
                                            current_(shape_.size(), 0) {
  for (auto d : shape_) {
    if (d == 0) { done_ = true; break; }
  }
}

bool IndexIterator::Next(Shape* out_index) {
  if (done_) return false;
  if (first_) {
    first_ = false;
    *out_index = current_;
    if (shape_.empty()) { done_ = true; }  // scalar: yields once
    return true;
  }
  // Increment rightmost dim, carrying.
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    if (++current_[i] < shape_[i]) {
      *out_index = current_;
      return true;
    }
    current_[i] = 0;
  }
  done_ = true;
  return false;
}

Tensor::Tensor(DType dtype, Shape shape)
    : dtype_(dtype), shape_(std::move(shape)) {
  strides_ = ContiguousStrides(shape_);
  int64_t bytes = numel() * DTypeBytes(dtype_);
  if (bytes < 0) bytes = 0;
  storage_.reset(new uint8_t[bytes](), std::default_delete<uint8_t[]>());
}

Tensor Tensor::Zeros(DType dtype, Shape shape) {
  return Tensor(dtype, std::move(shape));
}

Tensor Tensor::Uninit(DType dtype, Shape shape) {
  Tensor t;
  t.dtype_ = dtype;
  t.shape_ = std::move(shape);
  t.strides_ = ContiguousStrides(t.shape_);
  int64_t bytes = t.numel() * DTypeBytes(dtype);
  if (bytes < 0) bytes = 0;
  // new[] without () — uninitialized (no memset). Caller must fully overwrite.
  t.storage_.reset(new uint8_t[bytes], std::default_delete<uint8_t[]>());
  return t;
}

Tensor Tensor::FromHostBytes(DType dtype, Shape shape, const void* src) {
  Tensor t(dtype, std::move(shape));
  std::memcpy(t.bytes(), src, static_cast<size_t>(t.byte_size()));
  return t;
}

Tensor Tensor::BorrowingView(DType dtype, Shape shape,
                             std::shared_ptr<uint8_t[]> storage,
                             size_t byte_offset, Shape strides) {
  Tensor t;
  t.dtype_ = dtype;
  t.shape_ = std::move(shape);
  t.strides_ = std::move(strides);
  t.storage_ = std::move(storage);
  t.byte_offset_ = byte_offset;
  return t;
}

int64_t Tensor::numel() const {
  if (shape_.empty()) return 1;  // scalar
  int64_t n = 1;
  for (auto d : shape_) {
    if (d < 0) return -1;
    n *= d;
  }
  return n;
}

int64_t Tensor::byte_size() const {
  int64_t n = numel();
  if (n < 0) return -1;
  return n * DTypeBytes(dtype_);
}

bool Tensor::is_contiguous() const {
  Shape ideal = ContiguousStrides(shape_);
  return strides_ == ideal;
}

Tensor Tensor::Contiguous() const {
  if (is_contiguous() && byte_offset_ == 0) {
    // Already contiguous: share storage (shared_ptr), no copy. Kernels call
    // Contiguous() to guarantee row-major layout for read-only input, not to
    // get a private mutable buffer — and every kernel writes to a *separate*
    // output. Copying here meant every Gather of the [50257,768] embedding and
    // every MatMul on the LM-head weight deep-copied 154 MB per call (~20 ms).
    return *this;
  }
  Tensor out(dtype_, shape_);
  const int64_t elem_bytes = DTypeBytes(dtype_);
  IndexIterator it(shape_);
  Shape idx;
  uint8_t* dst = out.bytes();
  while (it.Next(&idx)) {
    int64_t src_off = ByteOffset(strides_, idx, elem_bytes);
    std::memcpy(dst, bytes() + src_off, static_cast<size_t>(elem_bytes));
    dst += elem_bytes;
  }
  return out;
}

bool Tensor::ApproximatelyEqualsFloat32(const Tensor& other, float tol) const {
  if (dtype_ != DType::kFloat32 || other.dtype_ != DType::kFloat32) return false;
  if (shape_ != other.shape_) return false;
  Tensor a = Contiguous();
  Tensor b = other.Contiguous();
  const float* pa = a.data<float>();
  const float* pb = b.data<float>();
  int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    float d = std::fabs(pa[i] - pb[i]);
    if (d > tol || std::isnan(d)) return false;
  }
  return true;
}

std::string Tensor::DebugString() const {
  std::ostringstream oss;
  oss << "Tensor(dtype=" << DTypeName(dtype_)
      << ", shape=" << ShapeToString(shape_)
      << ", strides=" << ShapeToString(strides_)
      << ", contiguous=" << (is_contiguous() ? "yes" : "no")
      << ", numel=" << numel() << ")";
  return oss.str();
}

}  // namespace rt
}  // namespace inferc
