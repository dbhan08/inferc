#include "kernels/elementwise.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace inferc {
namespace rt {

namespace {

Shape Broadcast(const Shape& a, const Shape& b) {
  const size_t r = std::max(a.size(), b.size());
  Shape out(r);
  for (size_t i = 0; i < r; ++i) {
    int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
    int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
    int64_t d;
    if (da == db || db == 1) d = da;
    else if (da == 1)        d = db;
    else throw std::runtime_error("elementwise: broadcast mismatch");
    out[r - 1 - i] = d;
  }
  return out;
}

// Compute the per-input flat offset for a given output index, treating
// broadcast dims (input has 1 where output doesn't) as stride 0.
inline int64_t BroadcastOffset(const Shape& in_shape, const Shape& out_shape,
                               const Shape& out_idx) {
  const int64_t skip = static_cast<int64_t>(out_shape.size()) -
                       static_cast<int64_t>(in_shape.size());
  int64_t lin = 0;
  int64_t stride = 1;
  for (int i = static_cast<int>(in_shape.size()) - 1; i >= 0; --i) {
    int64_t coord = (in_shape[i] == 1) ? 0 : out_idx[skip + i];
    lin += coord * stride;
    stride *= in_shape[i];
  }
  return lin;
}

// Templated zip-with-fn over an element type T (= float or int64_t).
template <typename T, typename Fn>
Tensor BinaryBroadcastTyped(const Tensor& a_in, const Tensor& b_in,
                            DType dtype, Fn fn) {
  Tensor a = a_in.Contiguous();
  Tensor b = b_in.Contiguous();
  Shape out_shape = Broadcast(a.shape(), b.shape());
  Tensor out = Tensor::Zeros(dtype, out_shape);
  const T* pa = a.data<T>();
  const T* pb = b.data<T>();
  T* po = out.data<T>();
  if (a.shape() == b.shape() && a.shape() == out_shape) {
    int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i) po[i] = fn(pa[i], pb[i]);
    return out;
  }
  IndexIterator it(out_shape);
  Shape idx;
  int64_t out_off = 0;
  while (it.Next(&idx)) {
    int64_t a_off = BroadcastOffset(a.shape(), out_shape, idx);
    int64_t b_off = BroadcastOffset(b.shape(), out_shape, idx);
    po[out_off] = fn(pa[a_off], pb[b_off]);
    ++out_off;
  }
  return out;
}

// Dtype-dispatch wrapper: route fp32 → BinaryBroadcastTyped<float>, int64
// → BinaryBroadcastTyped<int64_t>. Used by Add/Sub/Mul/Div/Pow which all
// take a same-type binary op `fn`. The caller passes a generic lambda
// that works on both (e.g., [](auto x, auto y) { return x + y; }).
template <typename Fn>
Tensor BinaryBroadcast(const Tensor& a_in, const Tensor& b_in, Fn fn) {
  if (a_in.dtype() != b_in.dtype()) {
    throw std::runtime_error("elementwise: dtype mismatch");
  }
  switch (a_in.dtype()) {
    case DType::kFloat32:
      return BinaryBroadcastTyped<float>(a_in, b_in, DType::kFloat32, fn);
    case DType::kInt64:
      return BinaryBroadcastTyped<int64_t>(a_in, b_in, DType::kInt64, fn);
    default:
      throw std::runtime_error("elementwise: only float32 and int64 supported");
  }
}

template <typename Fn>
Tensor UnaryPointwise(const Tensor& a_in, Fn fn) {
  if (a_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("elementwise: float32 only in v1");
  }
  Tensor a = a_in.Contiguous();
  Tensor out = Tensor::Zeros(DType::kFloat32, a.shape());
  const float* pa = a.data<float>();
  float* po = out.data<float>();
  int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) po[i] = fn(pa[i]);
  return out;
}

}  // namespace

Tensor Add(const Tensor& a, const Tensor& b) {
  return BinaryBroadcast(a, b, [](auto x, auto y) { return x + y; });
}
Tensor Sub(const Tensor& a, const Tensor& b) {
  return BinaryBroadcast(a, b, [](auto x, auto y) { return x - y; });
}
Tensor Mul(const Tensor& a, const Tensor& b) {
  return BinaryBroadcast(a, b, [](auto x, auto y) { return x * y; });
}
Tensor Div(const Tensor& a, const Tensor& b) {
  // Integer division for int64 path; float for fp32. Same '/' operator.
  return BinaryBroadcast(a, b, [](auto x, auto y) { return x / y; });
}
Tensor Pow(const Tensor& a, const Tensor& b) {
  // For int path we cast to float for pow; fp32 path uses std::pow directly.
  if (a.dtype() == DType::kInt64) {
    return BinaryBroadcast(a, b, [](auto x, auto y) {
      return static_cast<int64_t>(std::pow(static_cast<double>(x),
                                           static_cast<double>(y)));
    });
  }
  return BinaryBroadcast(a, b, [](auto x, auto y) {
    return static_cast<float>(std::pow(static_cast<double>(x),
                                       static_cast<double>(y)));
  });
}

namespace {

// Read a single element at logical index `idx` from `t` (assumed contiguous),
// promoted to int64. Supports int64/int32/float32 inputs (the dtypes that
// actually flow through DistilBERT's Equal nodes).
int64_t ReadAsInt64(const Tensor& t, int64_t idx) {
  switch (t.dtype()) {
    case DType::kInt64:  return t.data<int64_t>()[idx];
    case DType::kInt32:  return static_cast<int64_t>(t.data<int32_t>()[idx]);
    case DType::kFloat32: return static_cast<int64_t>(t.data<float>()[idx]);
    case DType::kBool:   return t.data<uint8_t>()[idx] ? 1 : 0;
    default: throw std::runtime_error("Equal: unsupported dtype");
  }
}

}  // namespace

Tensor Equal(const Tensor& a_in, const Tensor& b_in) {
  Tensor a = a_in.Contiguous();
  Tensor b = b_in.Contiguous();
  Shape out_shape = Broadcast(a.shape(), b.shape());
  Tensor out = Tensor::Zeros(DType::kBool, out_shape);
  uint8_t* po = out.data<uint8_t>();
  IndexIterator it(out_shape);
  Shape idx;
  int64_t off = 0;
  while (it.Next(&idx)) {
    int64_t ai = BroadcastOffset(a.shape(), out_shape, idx);
    int64_t bi = BroadcastOffset(b.shape(), out_shape, idx);
    po[off++] = (ReadAsInt64(a, ai) == ReadAsInt64(b, bi)) ? 1 : 0;
  }
  return out;
}

Tensor Where(const Tensor& cond_in, const Tensor& x_in, const Tensor& y_in) {
  if (cond_in.dtype() != DType::kBool) {
    throw std::runtime_error("Where: condition must be bool");
  }
  if (x_in.dtype() != y_in.dtype()) {
    throw std::runtime_error("Where: x and y dtype mismatch");
  }
  Tensor c = cond_in.Contiguous();
  Tensor x = x_in.Contiguous();
  Tensor y = y_in.Contiguous();
  Shape s1 = Broadcast(c.shape(), x.shape());
  Shape out_shape = Broadcast(s1, y.shape());
  Tensor out = Tensor::Zeros(x.dtype(), out_shape);
  const int64_t elem_bytes = DTypeBytes(x.dtype());
  const uint8_t* pc = c.data<uint8_t>();
  const uint8_t* px = x.bytes();
  const uint8_t* py = y.bytes();
  uint8_t* po = out.bytes();
  IndexIterator it(out_shape);
  Shape idx;
  int64_t off = 0;
  while (it.Next(&idx)) {
    int64_t ci = BroadcastOffset(c.shape(), out_shape, idx);
    int64_t xi = BroadcastOffset(x.shape(), out_shape, idx);
    int64_t yi = BroadcastOffset(y.shape(), out_shape, idx);
    bool b = pc[ci] != 0;
    const uint8_t* src = b ? (px + xi * elem_bytes) : (py + yi * elem_bytes);
    std::memcpy(po + off * elem_bytes, src, static_cast<size_t>(elem_bytes));
    ++off;
  }
  return out;
}

Tensor Sqrt(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return std::sqrt(x); });
}
Tensor Erf(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return std::erf(x); });
}
Tensor Relu(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return x > 0.0f ? x : 0.0f; });
}
Tensor Tanh(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return std::tanh(x); });
}
Tensor Neg(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return -x; });
}
Tensor Abs(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return std::fabs(x); });
}

}  // namespace rt
}  // namespace inferc
