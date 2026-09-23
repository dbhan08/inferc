#include "kernels/movement.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

#include "util/parallel.h"

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
  Tensor out = Tensor::Uninit(x.dtype(), out_shape);
  const int64_t n = out.numel();
  if (n == 0) return out;

  // Walk the output contiguously (out_off = o, the loop counter) and track the
  // matching input offset with an odometer instead of reconstructing a full
  // multi-index per element. `ostride_in[d]` is the input stride for the input
  // axis that output axis d maps from (perm[d]); incrementing output coordinate
  // d steps the input offset by that stride.
  //
  // The previous implementation allocated a `Shape` (heap) *per element*, which
  // turned GPT-2's per-step transpose of the [50257, 768] tied LM-head weight
  // (38.6M elements) into ~15 s. This version is allocation-free: O(n) with
  // O(rank) work only on odometer rollover.
  const Shape in_strides = ContiguousStrides(x.shape());
  std::vector<int64_t> ostride_in(static_cast<size_t>(r));
  for (int64_t i = 0; i < r; ++i) ostride_in[i] = in_strides[perm[i]];

  const int64_t elem_bytes = DTypeBytes(x.dtype());
  const uint8_t* src = x.bytes();
  uint8_t* dst = out.bytes();

  // Contiguous-run fast path: when the innermost output axis maps from a
  // contiguous input axis (ostride_in[r-1] == 1 — true for the common attention
  // reshape where the head_dim axis is preserved), the inner axis is a
  // contiguous run, so bulk-memcpy `run` elements instead of one at a time and
  // step the odometer only over the outer r-1 axes. Turns N strided 4-byte
  // copies into N/run cache-friendly block copies.
  if (r >= 1 && ostride_in[r - 1] == 1) {
    const int64_t run = out_shape[r - 1];
    const int64_t run_bytes = run * elem_bytes;
    const int64_t outer = n / run;
    // Parallelize over the outer runs; each block reconstructs its starting
    // in_off by decomposing its first block index, then odometers.
    par::ParallelFor(outer, /*grain=*/256, [&](int64_t blk0, int64_t blk1) {
      std::vector<int64_t> coord(static_cast<size_t>(r), 0);
      int64_t rem = blk0, in_off = 0;
      for (int64_t d = r - 2; d >= 0; --d) {
        coord[d] = rem % out_shape[d];
        rem /= out_shape[d];
        in_off += coord[d] * ostride_in[d];
      }
      uint8_t* d0 = dst + blk0 * run_bytes;
      for (int64_t o = blk0; o < blk1; ++o) {
        std::memcpy(d0, src + in_off * elem_bytes, static_cast<size_t>(run_bytes));
        d0 += run_bytes;
        for (int64_t d = r - 2; d >= 0; --d) {  // step outer axes only
          in_off += ostride_in[d];
          if (++coord[d] < out_shape[d]) break;
          coord[d] = 0;
          in_off -= ostride_in[d] * out_shape[d];
        }
      }
    });
    return out;
  }

  std::vector<int64_t> coord(static_cast<size_t>(r), 0);
  int64_t in_off = 0;
  for (int64_t o = 0; o < n; ++o) {
    std::memcpy(dst + o * elem_bytes, src + in_off * elem_bytes,
                static_cast<size_t>(elem_bytes));
    // Increment the odometer (last axis fastest), updating in_off in step.
    for (int64_t d = r - 1; d >= 0; --d) {
      in_off += ostride_in[d];
      if (++coord[d] < out_shape[d]) break;
      coord[d] = 0;
      in_off -= ostride_in[d] * out_shape[d];
    }
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
  Tensor out = Tensor::Uninit(dtype, out_shape);
  const int64_t elem_bytes = DTypeBytes(dtype);

  // outer = product of dims < axis; inner = product of dims > axis.
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= out_shape[i];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < r; ++i) inner *= out_shape[i];

  // For each outer "row" and each part, copy a slab of `a_dim * inner` elements.
  // Output layout (axis-concatenated): for each outer position, all parts'
  // slabs are laid out contiguously along the axis dim.
  uint8_t* base = out.bytes();
  const int64_t out_axis = out_shape[axis];
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
  Tensor out = Tensor::Uninit(x.dtype(), out_shape);
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

Tensor Expand(const Tensor& x_in, const Shape& target_in) {
  Tensor x = x_in.Contiguous();
  // ONNX Expand is a BIDIRECTIONAL broadcast: output dim = max(input, target)
  // where either side is 1. Right-align the shorter shape.
  const size_t r_in = x.shape().size();
  const size_t r_out = std::max(r_in, target_in.size());
  Shape padded(r_out, 1), target(r_out, 1);
  for (size_t i = 0; i < r_in; ++i) padded[r_out - r_in + i] = x.shape()[i];
  for (size_t i = 0; i < target_in.size(); ++i)
    target[r_out - target_in.size() + i] = target_in[i];
  for (size_t i = 0; i < r_out; ++i) {
    if (target[i] == 1) target[i] = padded[i];
    else if (padded[i] != target[i] && padded[i] != 1)
      throw std::runtime_error("Expand: shape not broadcastable");
  }
  Tensor out = Tensor::Uninit(x.dtype(), target);
  const int64_t elem_bytes = DTypeBytes(x.dtype());
  const int64_t n = out.numel();
  if (n == 0) return out;
  // Input stride per output axis (0 on broadcast axes), stepped by an odometer
  // instead of recomputing the input offset per element.
  const int r = static_cast<int>(r_out);
  Shape in_strides(r_out, 0);
  int64_t s = 1;
  for (int i = r - 1; i >= 0; --i) {
    in_strides[i] = (padded[i] == 1) ? 0 : s;
    s *= padded[i];
  }
  const uint8_t* src = x.bytes();
  uint8_t* dst = out.bytes();
  par::ParallelFor(n, /*grain=*/16384, [&](int64_t o0, int64_t o1) {
    std::vector<int64_t> coord(r_out, 0);
    int64_t rem = o0, in_off = 0;
    for (int d = r - 1; d >= 0; --d) {
      coord[d] = rem % target[d];
      rem /= target[d];
      in_off += coord[d] * in_strides[d];
    }
    for (int64_t o = o0; o < o1; ++o) {
      std::memcpy(dst + o * elem_bytes, src + in_off * elem_bytes,
                  static_cast<size_t>(elem_bytes));
      for (int d = r - 1; d >= 0; --d) {
        in_off += in_strides[d];
        if (++coord[d] < target[d]) break;
        coord[d] = 0;
        in_off -= in_strides[d] * target[d];
      }
    }
  });
  return out;
}

Tensor ShapeOf(const Tensor& x) {
  Shape s = x.shape();
  Tensor out = Tensor::Uninit(DType::kInt64, {static_cast<int64_t>(s.size())});
  int64_t* p = out.data<int64_t>();
  for (size_t i = 0; i < s.size(); ++i) p[i] = s[i];
  return out;
}

Tensor Cast(const Tensor& x_in, DType to) {
  Tensor x = x_in.Contiguous();
  if (x.dtype() == to) return x;
  Tensor out = Tensor::Uninit(to, x.shape());
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
    // Generic path: any numeric/bool source -> any numeric/bool destination,
    // through double (exact for every dtype in play except int64 > 2^53).
    auto read = [&](int64_t i) -> double {
      switch (x.dtype()) {
        case DType::kFloat32: return x.data<float>()[i];
        case DType::kFloat64: return x.data<double>()[i];
        case DType::kInt64:   return static_cast<double>(x.data<int64_t>()[i]);
        case DType::kInt32:   return x.data<int32_t>()[i];
        case DType::kInt8:    return x.data<int8_t>()[i];
        case DType::kUint8:   return x.data<uint8_t>()[i];
        case DType::kBool:    return x.data<uint8_t>()[i] ? 1.0 : 0.0;
        default: throw std::runtime_error("Cast: unsupported source dtype");
      }
    };
    for (int64_t i = 0; i < n; ++i) {
      const double v = read(i);
      switch (to) {
        case DType::kFloat32: out.data<float>()[i] = static_cast<float>(v); break;
        case DType::kFloat64: out.data<double>()[i] = v; break;
        case DType::kInt64:   out.data<int64_t>()[i] = static_cast<int64_t>(v); break;
        case DType::kInt32:   out.data<int32_t>()[i] = static_cast<int32_t>(v); break;
        case DType::kInt8:    out.data<int8_t>()[i] = static_cast<int8_t>(v); break;
        case DType::kUint8:   out.data<uint8_t>()[i] = static_cast<uint8_t>(v); break;
        case DType::kBool:    out.data<uint8_t>()[i] = (v != 0.0) ? 1 : 0; break;
        default: throw std::runtime_error("Cast: unsupported destination dtype");
      }
    }
  }
  return out;
}

Tensor ConstantOfShape(const Shape& shape, DType dtype,
                       const std::vector<uint8_t>& fill_bytes) {
  Tensor out = Tensor::Uninit(dtype, shape);
  const int64_t n = out.numel();
  const int64_t elem_bytes = DTypeBytes(dtype);
  if (fill_bytes.empty() || elem_bytes == 0) return out;  // already zero
  if (static_cast<int64_t>(fill_bytes.size()) != elem_bytes) {
    throw std::runtime_error("ConstantOfShape: fill_bytes size != dtype size");
  }
  uint8_t* p = out.bytes();
  for (int64_t i = 0; i < n; ++i) {
    std::memcpy(p + i * elem_bytes, fill_bytes.data(),
                static_cast<size_t>(elem_bytes));
  }
  return out;
}

std::vector<Tensor> Split(const Tensor& x_in, int64_t axis,
                          const std::vector<int64_t>& split_sizes) {
  Tensor x = x_in.Contiguous();
  const int64_t r = x.rank();
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) throw std::runtime_error("Split: axis out of range");
  int64_t total = 0;
  for (auto s : split_sizes) total += s;
  if (total != x.shape()[axis]) {
    throw std::runtime_error("Split: sum(split_sizes) != x.shape[axis]");
  }
  // outer = product of dims < axis; inner = product of dims > axis.
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= x.shape()[i];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < r; ++i) inner *= x.shape()[i];
  const int64_t elem_bytes = DTypeBytes(x.dtype());
  const int64_t axis_full = x.shape()[axis];

  std::vector<Tensor> outs;
  outs.reserve(split_sizes.size());
  int64_t axis_offset = 0;
  for (int64_t sz : split_sizes) {
    Shape s = x.shape();
    s[axis] = sz;
    Tensor o = Tensor::Uninit(x.dtype(), s);
    // For each "outer" block, copy a slab of size [sz * inner] from x at
    // axis_offset within that block.
    uint8_t* dst = o.bytes();
    const uint8_t* src = x.bytes();
    const int64_t bytes_per_block = sz * inner * elem_bytes;
    for (int64_t o_i = 0; o_i < outer; ++o_i) {
      const uint8_t* src_block = src + (o_i * axis_full + axis_offset) * inner * elem_bytes;
      uint8_t* dst_block = dst + o_i * sz * inner * elem_bytes;
      std::memcpy(dst_block, src_block, static_cast<size_t>(bytes_per_block));
    }
    outs.push_back(std::move(o));
    axis_offset += sz;
  }
  return outs;
}

Tensor RangeI64(int64_t start, int64_t limit, int64_t delta) {
  if (delta == 0) throw std::runtime_error("Range: delta must be non-zero");
  int64_t n;
  if (delta > 0) {
    n = (limit > start) ? (limit - start + delta - 1) / delta : 0;
  } else {
    n = (start > limit) ? (start - limit + (-delta) - 1) / (-delta) : 0;
  }
  Tensor out = Tensor::Uninit(DType::kInt64, {n});
  int64_t* p = out.data<int64_t>();
  for (int64_t i = 0; i < n; ++i) p[i] = start + i * delta;
  return out;
}

Tensor RangeF32(float start, float limit, float delta) {
  if (delta == 0.0f) throw std::runtime_error("Range: delta must be non-zero");
  int64_t n;
  if (delta > 0) {
    n = (limit > start) ? static_cast<int64_t>((limit - start + delta - 1e-7f) / delta) : 0;
  } else {
    n = (start > limit) ? static_cast<int64_t>((start - limit + (-delta) - 1e-7f) / (-delta)) : 0;
  }
  if (n < 0) n = 0;
  Tensor out = Tensor::Uninit(DType::kFloat32, {n});
  float* p = out.data<float>();
  for (int64_t i = 0; i < n; ++i) p[i] = start + static_cast<float>(i) * delta;
  return out;
}

Tensor Trilu(const Tensor& x_in, int64_t k, bool upper) {
  if (x_in.rank() < 2) throw std::runtime_error("Trilu: rank must be >= 2");
  Tensor x = x_in.Contiguous();
  // Contiguous() may share storage; take a private copy before mutating.
  Tensor out = Tensor::FromHostBytes(x.dtype(), x.shape(), x.bytes());
  const int64_t eb = DTypeBytes(x.dtype());
  const int64_t rows = x.shape()[x.rank() - 2], cols = x.shape()[x.rank() - 1];
  const int64_t batch = (rows * cols) ? x.numel() / (rows * cols) : 0;
  uint8_t* p = out.bytes();
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t j = 0; j < cols; ++j) {
        const bool keep = upper ? (j - i >= k) : (j - i <= k);
        if (!keep) std::memset(p + ((b * rows + i) * cols + j) * eb, 0, static_cast<size_t>(eb));
      }
  return out;
}

Tensor ScatterND(const Tensor& data_in, const Tensor& indices_in, const Tensor& updates_in) {
  Tensor data = data_in.Contiguous();
  Tensor indices = indices_in.Contiguous();
  Tensor updates = updates_in.Contiguous();
  if (indices.dtype() != DType::kInt64) throw std::runtime_error("ScatterND: indices must be int64");
  if (updates.dtype() != data.dtype()) throw std::runtime_error("ScatterND: dtype mismatch");
  // Private copy of data (Contiguous() may alias the input's storage).
  Tensor out = Tensor::FromHostBytes(data.dtype(), data.shape(), data.bytes());
  const int64_t eb = DTypeBytes(data.dtype());
  const int64_t q = indices.shape().back();
  if (q > data.rank()) throw std::runtime_error("ScatterND: index depth exceeds data rank");
  // Byte size of one update slice = product of data dims [q..rank).
  int64_t slice_elems = 1;
  for (int64_t d = q; d < data.rank(); ++d) slice_elems *= data.shape()[d];
  // Row-major strides of data (in elements).
  std::vector<int64_t> stride(data.rank(), 1);
  for (int64_t d = data.rank() - 2; d >= 0; --d) stride[d] = stride[d + 1] * data.shape()[d + 1];
  const int64_t n_idx = q ? indices.numel() / q : 0;
  const int64_t* ip = indices.data<int64_t>();
  const uint8_t* up = updates.bytes();
  uint8_t* op = out.bytes();
  for (int64_t t = 0; t < n_idx; ++t) {
    int64_t off = 0;
    for (int64_t d = 0; d < q; ++d) {
      int64_t v = ip[t * q + d];
      if (v < 0) v += data.shape()[d];
      if (v < 0 || v >= data.shape()[d]) throw std::runtime_error("ScatterND: index out of range");
      off += v * stride[d];
    }
    std::memcpy(op + off * eb, up + t * slice_elems * eb, static_cast<size_t>(slice_elems * eb));
  }
  return out;
}

}  // namespace rt
}  // namespace inferc
