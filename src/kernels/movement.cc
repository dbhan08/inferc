#include "kernels/movement.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

namespace inferc {
namespace rt {

namespace {

template <typename DstT, typename SrcT>
void CastLoop(const SrcT* src, DstT* dst, int64_t n) {
  for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<DstT>(src[i]);
}

}  // namespace

Tensor Reshape(const Tensor& x, Shape new_shape) {
  // Handle -1: infer that dim from total element count.
  int neg_idx = -1;
  int64_t known = 1;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      if (neg_idx >= 0) throw std::runtime_error("Reshape: more than one -1");
      neg_idx = static_cast<int>(i);
    } else if (new_shape[i] < 0) {
      throw std::runtime_error("Reshape: negative dim other than -1");
    } else {
      known *= new_shape[i];
    }
  }
  int64_t total = x.numel();
  if (neg_idx >= 0) {
    if (known == 0 || total % known != 0) {
      throw std::runtime_error("Reshape: cannot infer -1");
    }
    new_shape[neg_idx] = total / known;
  } else if (known != total) {
    throw std::runtime_error("Reshape: element count mismatch");
  }
  // Force contiguous backing then re-wrap with new shape (zero-copy on the data,
  // we just produce a fresh contiguous Tensor for simplicity).
  Tensor src = x.Contiguous();
  return Tensor::FromHostBytes(src.dtype(), new_shape, src.bytes());
}

Tensor Transpose(const Tensor& x_in, const std::vector<int64_t>& perm_in) {
  Tensor x = x_in.Contiguous();
  const int64_t r = x.rank();
  std::vector<int64_t> perm = perm_in;
  if (perm.empty()) {
    perm.resize(r);
    for (int64_t i = 0; i < r; ++i) perm[i] = r - 1 - i;
  }
  if (static_cast<int64_t>(perm.size()) != r) {
    throw std::runtime_error("Transpose: perm size != rank");
  }
  Shape out_shape(r);
  for (int64_t i = 0; i < r; ++i) {
    int64_t p = perm[i];
    if (p < 0 || p >= r) throw std::runtime_error("Transpose: bad perm");
    out_shape[i] = x.shape()[p];
  }
  Tensor out = Tensor::Zeros(x.dtype(), out_shape);

  // Compute input strides (contiguous).
  Shape in_strides = ContiguousStrides(x.shape());
  Shape out_strides = ContiguousStrides(out_shape);

  const int64_t elem_bytes = DTypeBytes(x.dtype());
  IndexIterator it(out_shape);
  Shape out_idx;
  while (it.Next(&out_idx)) {
    // Map output index → input index by inverse perm.
    Shape in_idx(r);
    for (int64_t i = 0; i < r; ++i) in_idx[perm[i]] = out_idx[i];
    int64_t in_off = 0;
    for (int64_t i = 0; i < r; ++i) in_off += in_idx[i] * in_strides[i];
    int64_t out_off = 0;
    for (int64_t i = 0; i < r; ++i) out_off += out_idx[i] * out_strides[i];
    std::memcpy(out.bytes() + out_off * elem_bytes,
                x.bytes() + in_off * elem_bytes,
                static_cast<size_t>(elem_bytes));
  }
  return out;
}

Tensor Concat(const std::vector<Tensor>& parts, int64_t axis) {
  if (parts.empty()) throw std::runtime_error("Concat: no inputs");
  const int64_t r = parts[0].rank();
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) throw std::runtime_error("Concat: axis out of range");
  // Build output shape; sum along `axis`.
  Shape out_shape = parts[0].shape();
  out_shape[axis] = 0;
  DType dtype = parts[0].dtype();
  for (const auto& p : parts) {
    if (p.dtype() != dtype) throw std::runtime_error("Concat: dtype mismatch");
    if (p.rank() != r) throw std::runtime_error("Concat: rank mismatch");
    for (int64_t i = 0; i < r; ++i) {
      if (i == axis) continue;
      if (p.shape()[i] != out_shape[i]) {
        throw std::runtime_error("Concat: dim mismatch outside axis");
      }
    }
    out_shape[axis] += p.shape()[axis];
  }
  Tensor out = Tensor::Zeros(dtype, out_shape);
  const int64_t elem_bytes = DTypeBytes(dtype);

  // outer = product of dims < axis; inner = product of dims > axis.
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= out_shape[i];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < r; ++i) inner *= out_shape[i];

  uint8_t* dst = out.bytes();
  for (int64_t o = 0; o < outer; ++o) {
    for (const auto& p_in : parts) {
      Tensor p = p_in.Contiguous();
      const int64_t a_dim = p.shape()[axis];
      const int64_t bytes = a_dim * inner * elem_bytes;
      const uint8_t* src = p.bytes() + o * a_dim * inner * elem_bytes;
      std::memcpy(dst + o * out_shape[axis] * inner * elem_bytes
                      + (dst - out.bytes() == 0 ? 0 : 0),  // (no-op, kept for clarity)
                  src, static_cast<size_t>(bytes));
      dst += bytes;
    }
    // Wind back: dst was advanced past one whole "outer" group; reset to base
    // so the next outer iteration starts at the right place. We compute via
    // index instead.
  }
  // The pointer arithmetic above is fiddly — redo more simply.
  std::memset(out.bytes(), 0, out.byte_size());
  uint8_t* base = out.bytes();
  int64_t out_axis = out_shape[axis];
  for (int64_t o = 0; o < outer; ++o) {
    int64_t axis_pos = 0;
    for (const auto& p_in : parts) {
      Tensor p = p_in.Contiguous();
      const int64_t a_dim = p.shape()[axis];
      const int64_t bytes = a_dim * inner * elem_bytes;
      const uint8_t* src = p.bytes() + o * a_dim * inner * elem_bytes;
      uint8_t* d = base + (o * out_axis + axis_pos) * inner * elem_bytes;
      std::memcpy(d, src, static_cast<size_t>(bytes));
      axis_pos += a_dim;
    }
  }
  return out;
}

Tensor Slice(const Tensor& x_in,
             const std::vector<int64_t>& starts,
             const std::vector<int64_t>& ends,
             const std::vector<int64_t>& axes_in,
             const std::vector<int64_t>& steps_in) {
  Tensor x = x_in.Contiguous();
  const int64_t r = x.rank();
  std::vector<int64_t> axes = axes_in;
  if (axes.empty()) {
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) axes[i] = static_cast<int64_t>(i);
  }
  std::vector<int64_t> steps = steps_in;
  if (steps.empty()) steps.assign(starts.size(), 1);
  if (starts.size() != ends.size() || starts.size() != axes.size() ||
      starts.size() != steps.size()) {
    throw std::runtime_error("Slice: starts/ends/axes/steps size mismatch");
  }

  // Compute per-axis (start, end_inclusive_via_step, step) → length.
  Shape effective_starts(r, 0);
  Shape effective_lens = x.shape();
  Shape effective_steps(r, 1);
  for (size_t i = 0; i < starts.size(); ++i) {
    int64_t ax = axes[i] < 0 ? axes[i] + r : axes[i];
    int64_t step = steps[i];
    if (step == 0) throw std::runtime_error("Slice: step == 0");
    int64_t dim = x.shape()[ax];
    int64_t s = starts[i], e = ends[i];
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = std::max<int64_t>(0, std::min(s, dim));
    e = std::max<int64_t>(0, std::min(e, dim));
    int64_t len = step > 0
        ? std::max<int64_t>(0, (e - s + step - 1) / step)
        : std::max<int64_t>(0, (s - e - step - 1) / -step);
    effective_starts[ax] = s;
    effective_lens[ax] = len;
    effective_steps[ax] = step;
  }
  Shape out_shape = effective_lens;
  Tensor out = Tensor::Zeros(x.dtype(), out_shape);
  const int64_t elem_bytes = DTypeBytes(x.dtype());

  Shape in_strides = ContiguousStrides(x.shape());
  IndexIterator it(out_shape);
  Shape oidx;
  int64_t out_lin = 0;
  while (it.Next(&oidx)) {
    int64_t in_off = 0;
    for (int64_t i = 0; i < r; ++i) {
      int64_t coord = effective_starts[i] + oidx[i] * effective_steps[i];
      in_off += coord * in_strides[i];
    }
    std::memcpy(out.bytes() + out_lin * elem_bytes,
                x.bytes() + in_off * elem_bytes,
                static_cast<size_t>(elem_bytes));
    ++out_lin;
  }
  return out;
}

Tensor Unsqueeze(const Tensor& x, const std::vector<int64_t>& axes_in) {
  Shape s = x.shape();
  const int64_t out_rank = static_cast<int64_t>(s.size()) +
                           static_cast<int64_t>(axes_in.size());
  std::vector<int64_t> axes = axes_in;
  for (auto& a : axes) if (a < 0) a += out_rank;
  std::sort(axes.begin(), axes.end());
  Shape out;
  size_t in_idx = 0, ax_idx = 0;
  for (int64_t i = 0; i < out_rank; ++i) {
    if (ax_idx < axes.size() && axes[ax_idx] == i) {
      out.push_back(1);
      ++ax_idx;
    } else if (in_idx < s.size()) {
      out.push_back(s[in_idx++]);
    }
  }
  return Reshape(x, out);
}

Tensor Squeeze(const Tensor& x, const std::vector<int64_t>& axes_in) {
  const int64_t r = x.rank();
  std::set<int64_t> drop;
  if (axes_in.empty()) {
    for (int64_t i = 0; i < r; ++i) if (x.shape()[i] == 1) drop.insert(i);
  } else {
    for (auto a : axes_in) drop.insert(a < 0 ? a + r : a);
  }
  Shape out;
  for (int64_t i = 0; i < r; ++i) if (!drop.count(i)) out.push_back(x.shape()[i]);
  return Reshape(x, out);
}

Tensor Expand(const Tensor& x_in, const Shape& target) {
  Tensor x = x_in.Contiguous();
  // Right-align x.shape() against target; each dim must equal target dim
  // or be 1 (broadcast).
  const size_t r_in = x.shape().size();
  const size_t r_out = target.size();
  if (r_in > r_out) throw std::runtime_error("Expand: input rank > target rank");
  Shape padded(r_out, 1);
  for (size_t i = 0; i < r_in; ++i) padded[r_out - r_in + i] = x.shape()[i];
  for (size_t i = 0; i < r_out; ++i) {
    if (padded[i] != target[i] && padded[i] != 1) {
      throw std::runtime_error("Expand: shape not broadcastable");
    }
  }
  Tensor out = Tensor::Zeros(x.dtype(), target);
  const int64_t elem_bytes = DTypeBytes(x.dtype());
  IndexIterator it(target);
  Shape idx;
  int64_t out_off = 0;
  // Compute input strides on padded shape.
  Shape in_strides(r_out, 0);
  int64_t s = 1;
  for (int i = static_cast<int>(r_out) - 1; i >= 0; --i) {
    in_strides[i] = (padded[i] == 1) ? 0 : s;
    s *= padded[i];
  }
  while (it.Next(&idx)) {
    int64_t in_off = 0;
    for (size_t i = 0; i < r_out; ++i) {
      in_off += (padded[i] == 1 ? 0 : idx[i]) * in_strides[i];
    }
    std::memcpy(out.bytes() + out_off * elem_bytes,
                x.bytes() + in_off * elem_bytes,
                static_cast<size_t>(elem_bytes));
    ++out_off;
  }
  return out;
}

Tensor ShapeOf(const Tensor& x) {
  Shape s = x.shape();
  Tensor out = Tensor::Zeros(DType::kInt64, {static_cast<int64_t>(s.size())});
  int64_t* p = out.data<int64_t>();
  for (size_t i = 0; i < s.size(); ++i) p[i] = s[i];
  return out;
}

Tensor Cast(const Tensor& x_in, DType to) {
  Tensor x = x_in.Contiguous();
  if (x.dtype() == to) return x;
  Tensor out = Tensor::Zeros(to, x.shape());
  const int64_t n = x.numel();
  // Cover the conversions that actually appear in DistilBERT (mostly bool/int
  // ↔ float32 around the attention mask) plus a few common ones.
  if (x.dtype() == DType::kFloat32 && to == DType::kInt64) {
    CastLoop<int64_t>(x.data<float>(), out.data<int64_t>(), n);
  } else if (x.dtype() == DType::kInt64 && to == DType::kFloat32) {
    CastLoop<float>(x.data<int64_t>(), out.data<float>(), n);
  } else if (x.dtype() == DType::kInt32 && to == DType::kFloat32) {
    CastLoop<float>(x.data<int32_t>(), out.data<float>(), n);
  } else if (x.dtype() == DType::kFloat32 && to == DType::kInt32) {
    CastLoop<int32_t>(x.data<float>(), out.data<int32_t>(), n);
  } else if (x.dtype() == DType::kBool && to == DType::kFloat32) {
    const uint8_t* p = x.data<uint8_t>();
    float* q = out.data<float>();
    for (int64_t i = 0; i < n; ++i) q[i] = p[i] ? 1.0f : 0.0f;
  } else if (x.dtype() == DType::kInt64 && to == DType::kInt32) {
    CastLoop<int32_t>(x.data<int64_t>(), out.data<int32_t>(), n);
  } else if (x.dtype() == DType::kInt32 && to == DType::kInt64) {
    CastLoop<int64_t>(x.data<int32_t>(), out.data<int64_t>(), n);
  } else {
    throw std::runtime_error("Cast: unsupported dtype pair");
  }
  return out;
}

}  // namespace rt
}  // namespace inferc
