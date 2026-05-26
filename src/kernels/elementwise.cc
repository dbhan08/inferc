#include "kernels/elementwise.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "util/parallel.h"

namespace inferc {
namespace rt {

namespace {

// Tag so the equal-shape float fast path can dispatch to vDSP.
enum class BinOp { kAdd, kSub, kMul, kDiv, kGeneric };

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

// Per-output-axis stride of an input (right-aligned to the output), with 0 on
// broadcast axes — lets us step the input offset with an odometer instead of
// recomputing it per element.
inline void BroadcastStrides(const Shape& in_shape, const Shape& out_shape,
                             std::vector<int64_t>* str) {
  const int r = static_cast<int>(out_shape.size());
  str->assign(r, 0);
  const int skip = r - static_cast<int>(in_shape.size());
  int64_t st = 1;
  for (int i = static_cast<int>(in_shape.size()) - 1; i >= 0; --i) {
    (*str)[skip + i] = (in_shape[i] == 1) ? 0 : st;
    st *= in_shape[i];
  }
}

// Templated zip-with-fn over an element type T (= float or int64_t).
template <typename T, typename Fn>
Tensor BinaryBroadcastTyped(const Tensor& a_in, const Tensor& b_in,
                            DType dtype, BinOp op, Fn fn) {
  Tensor a = a_in.Contiguous();
  Tensor b = b_in.Contiguous();
  Shape out_shape = Broadcast(a.shape(), b.shape());
  Tensor out = Tensor::Uninit(dtype, out_shape);
  const T* pa = a.data<T>();
  const T* pb = b.data<T>();
  T* po = out.data<T>();
  const int64_t n = out.numel();
  if (n == 0) return out;

  // Scalar-operand fast path (float): one operand is a single value broadcast
  // over the other. Use vDSP vector-scalar ops (vsadd/vsmul) — far faster than
  // the general broadcast machinery. Covers e.g. attention's scores/sqrt(d_k).
  if constexpr (std::is_same_v<T, float>) {
    const bool b_scalar = (b.numel() == 1);
    const bool a_scalar = (a.numel() == 1);
    if ((b_scalar || a_scalar) && out_shape == (b_scalar ? a.shape() : b.shape())) {
      const auto N = static_cast<vDSP_Length>(n);
      if (b_scalar) {
        const float c = pb[0];
        switch (op) {
          case BinOp::kAdd: { float k = c;  vDSP_vsadd(pa,1,&k,po,1,N); return out; }
          case BinOp::kSub: { float k = -c; vDSP_vsadd(pa,1,&k,po,1,N); return out; }
          case BinOp::kMul: { float k = c;  vDSP_vsmul(pa,1,&k,po,1,N); return out; }
          case BinOp::kDiv: { float k = 1.0f/c; vDSP_vsmul(pa,1,&k,po,1,N); return out; }
          case BinOp::kGeneric: break;
        }
      } else {  // a is the scalar
        const float c = pa[0];
        switch (op) {
          case BinOp::kAdd: { float k = c; vDSP_vsadd(pb,1,&k,po,1,N); return out; }
          case BinOp::kMul: { float k = c; vDSP_vsmul(pb,1,&k,po,1,N); return out; }
          // Sub/Div with scalar numerator aren't commutative — fall through.
          default: break;
        }
      }
    }
  }

  // Equal-shape fast path — contiguous, no broadcasting. (Not parallelized:
  // these ops are already vDSP-fast and called many times per inference, so
  // per-op dispatch overhead outweighs the gain — measured net-negative.)
  if (a.shape() == b.shape() && a.shape() == out_shape) {
    if constexpr (std::is_same_v<T, float>) {
      const auto N = static_cast<vDSP_Length>(n);  // vsub/vdiv take B first
      switch (op) {
        case BinOp::kAdd: vDSP_vadd(pa, 1, pb, 1, po, 1, N); return out;
        case BinOp::kSub: vDSP_vsub(pb, 1, pa, 1, po, 1, N); return out;
        case BinOp::kMul: vDSP_vmul(pa, 1, pb, 1, po, 1, N); return out;
        case BinOp::kDiv: vDSP_vdiv(pb, 1, pa, 1, po, 1, N); return out;
        case BinOp::kGeneric: break;
      }
    }
    for (int64_t i = 0; i < n; ++i) po[i] = fn(pa[i], pb[i]);
    return out;
  }

  // Broadcast path — odometer-step a_off/b_off (O(1) amortized per element)
  // instead of recomputing BroadcastOffset (O(rank)) for both inputs per element.
  const int r = static_cast<int>(out_shape.size());
  std::vector<int64_t> a_str, b_str;
  BroadcastStrides(a.shape(), out_shape, &a_str);
  BroadcastStrides(b.shape(), out_shape, &b_str);

  // Contiguous-inner-run fast path: when both inputs are contiguous on the
  // innermost axis (the common case — e.g. the attention-mask Add where the
  // mask broadcasts over the head axis but the inner [seq] axis is full), the
  // inner axis is a contiguous run for both, so vectorize it (vDSP for float)
  // and step the odometer only over the outer axes.
  if (r >= 1 && a_str[r - 1] == 1 && b_str[r - 1] == 1) {
    const int64_t run = out_shape[r - 1];
    const int64_t outer = n / run;
    std::vector<int64_t> coord(r, 0);
    int64_t a_off = 0, b_off = 0, o_off = 0;
    for (int64_t blk = 0; blk < outer; ++blk) {
      const T* aa = pa + a_off;
      const T* bb = pb + b_off;
      T* oo = po + o_off;
      bool done = false;
      if constexpr (std::is_same_v<T, float>) {
        const auto N = static_cast<vDSP_Length>(run);
        switch (op) {
          case BinOp::kAdd: vDSP_vadd(aa, 1, bb, 1, oo, 1, N); done = true; break;
          case BinOp::kSub: vDSP_vsub(bb, 1, aa, 1, oo, 1, N); done = true; break;
          case BinOp::kMul: vDSP_vmul(aa, 1, bb, 1, oo, 1, N); done = true; break;
          case BinOp::kDiv: vDSP_vdiv(bb, 1, aa, 1, oo, 1, N); done = true; break;
          case BinOp::kGeneric: break;
        }
      }
      if (!done)
        for (int64_t i = 0; i < run; ++i) oo[i] = fn(aa[i], bb[i]);
      o_off += run;
      for (int d = r - 2; d >= 0; --d) {
        a_off += a_str[d];
        b_off += b_str[d];
        if (++coord[d] < out_shape[d]) break;
        coord[d] = 0;
        a_off -= a_str[d] * out_shape[d];
        b_off -= b_str[d] * out_shape[d];
      }
    }
    return out;
  }

  std::vector<int64_t> coord(r, 0);
  int64_t a_off = 0, b_off = 0;
  for (int64_t o = 0; o < n; ++o) {
    po[o] = fn(pa[a_off], pb[b_off]);
    for (int d = r - 1; d >= 0; --d) {
      a_off += a_str[d];
      b_off += b_str[d];
      if (++coord[d] < out_shape[d]) break;
      coord[d] = 0;
      a_off -= a_str[d] * out_shape[d];
      b_off -= b_str[d] * out_shape[d];
    }
  }
  return out;
}

// Dtype-dispatch wrapper: route fp32 → BinaryBroadcastTyped<float>, int64
// → BinaryBroadcastTyped<int64_t>. `op` selects the vDSP fast path for the
// equal-shape float case; `fn` handles int64 + every broadcast case.
template <typename Fn>
Tensor BinaryBroadcast(const Tensor& a_in, const Tensor& b_in, BinOp op, Fn fn) {
  if (a_in.dtype() != b_in.dtype()) {
    throw std::runtime_error("elementwise: dtype mismatch");
  }
  switch (a_in.dtype()) {
    case DType::kFloat32:
      return BinaryBroadcastTyped<float>(a_in, b_in, DType::kFloat32, op, fn);
    case DType::kInt64:
      return BinaryBroadcastTyped<int64_t>(a_in, b_in, DType::kInt64, op, fn);
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
  Tensor out = Tensor::Uninit(DType::kFloat32, a.shape());
  const float* pa = a.data<float>();
  float* po = out.data<float>();
  int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) po[i] = fn(pa[i]);
  return out;
}

}  // namespace

Tensor Add(const Tensor& a, const Tensor& b) {
  return BinaryBroadcast(a, b, BinOp::kAdd, [](auto x, auto y) { return x + y; });
}
Tensor Sub(const Tensor& a, const Tensor& b) {
  return BinaryBroadcast(a, b, BinOp::kSub, [](auto x, auto y) { return x - y; });
}
Tensor Mul(const Tensor& a, const Tensor& b) {
  return BinaryBroadcast(a, b, BinOp::kMul, [](auto x, auto y) { return x * y; });
}
Tensor Div(const Tensor& a, const Tensor& b) {
  // Integer division for int64 path; float for fp32. Same '/' operator.
  return BinaryBroadcast(a, b, BinOp::kDiv, [](auto x, auto y) { return x / y; });
}
Tensor Pow(const Tensor& a, const Tensor& b) {
  // For int path we cast to float for pow; fp32 path uses std::pow directly.
  if (a.dtype() == DType::kInt64) {
    return BinaryBroadcast(a, b, BinOp::kGeneric, [](auto x, auto y) {
      return static_cast<int64_t>(std::pow(static_cast<double>(x),
                                           static_cast<double>(y)));
    });
  }
  return BinaryBroadcast(a, b, BinOp::kGeneric, [](auto x, auto y) {
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
  Tensor out = Tensor::Uninit(DType::kBool, out_shape);
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
  Tensor out = Tensor::Uninit(x.dtype(), out_shape);
  const int64_t elem_bytes = DTypeBytes(x.dtype());
  const uint8_t* pc = c.data<uint8_t>();
  const uint8_t* px = x.bytes();
  const uint8_t* py = y.bytes();
  uint8_t* po = out.bytes();
  const int64_t n = out.numel();
  if (n == 0) return out;
  // Odometer-step c/x/y offsets instead of recomputing three BroadcastOffsets
  // per element (the attention-mask Where runs on millions of elements).
  const int r = static_cast<int>(out_shape.size());
  std::vector<int64_t> c_str, x_str, y_str;
  BroadcastStrides(c.shape(), out_shape, &c_str);
  BroadcastStrides(x.shape(), out_shape, &x_str);
  BroadcastStrides(y.shape(), out_shape, &y_str);
  // Parallelize over the output range; each block reconstructs its starting
  // c/x/y offsets by decomposing its first index, then odometers. Where is a
  // handful of big calls per inference (attention mask), so dispatch overhead
  // amortizes well (unlike the many small broadcast Adds, which we leave serial).
  par::ParallelFor(n, /*grain=*/16384, [&](int64_t o0, int64_t o1) {
    std::vector<int64_t> coord(r, 0);
    int64_t rem = o0, ci = 0, xi = 0, yi = 0;
    for (int d = r - 1; d >= 0; --d) {
      coord[d] = rem % out_shape[d];
      rem /= out_shape[d];
      ci += coord[d] * c_str[d];
      xi += coord[d] * x_str[d];
      yi += coord[d] * y_str[d];
    }
    for (int64_t o = o0; o < o1; ++o) {
      const uint8_t* src = (pc[ci] != 0) ? (px + xi * elem_bytes)
                                         : (py + yi * elem_bytes);
      std::memcpy(po + o * elem_bytes, src, static_cast<size_t>(elem_bytes));
      for (int d = r - 1; d >= 0; --d) {
        ci += c_str[d]; xi += x_str[d]; yi += y_str[d];
        if (++coord[d] < out_shape[d]) break;
        coord[d] = 0;
        ci -= c_str[d] * out_shape[d];
        xi -= x_str[d] * out_shape[d];
        yi -= y_str[d] * out_shape[d];
      }
    }
  });
  return out;
}

Tensor Sqrt(const Tensor& a) {
  if (a.dtype() == DType::kFloat32) {
    Tensor x = a.Contiguous();
    Tensor out = Tensor::Uninit(DType::kFloat32, x.shape());
    const int n = static_cast<int>(x.numel());
    vvsqrtf(out.data<float>(), x.data<float>(), &n);  // Accelerate vForce
    return out;
  }
  return UnaryPointwise(a, [](float x) { return std::sqrt(x); });
}
Tensor Erf(const Tensor& a) {
  // vForce has no erf; keep scalar (DistilBERT's erf is inside the fused GELU).
  return UnaryPointwise(a, [](float x) { return std::erf(x); });
}
Tensor Relu(const Tensor& a) {
  return UnaryPointwise(a, [](float x) { return x > 0.0f ? x : 0.0f; });
}
Tensor Tanh(const Tensor& a) {
  if (a.dtype() == DType::kFloat32) {
    Tensor x = a.Contiguous();
    Tensor out = Tensor::Uninit(DType::kFloat32, x.shape());
    const int n = static_cast<int>(x.numel());
    vvtanhf(out.data<float>(), x.data<float>(), &n);  // Accelerate vForce
    return out;
  }
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
