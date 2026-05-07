#include "kernels/activation.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace inferc {
namespace rt {

Tensor Gelu(const Tensor& x_in) {
  if (x_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("Gelu: float32 only");
  }
  Tensor x = x_in.Contiguous();
  Tensor out = Tensor::Zeros(DType::kFloat32, x.shape());
  const float* p = x.data<float>();
  float* q = out.data<float>();
  const int64_t n = x.numel();
  constexpr float kInvSqrt2 = 0.70710678118654752440f;  // 1/sqrt(2)
  for (int64_t i = 0; i < n; ++i) {
    q[i] = p[i] * 0.5f * (1.0f + std::erf(p[i] * kInvSqrt2));
  }
  return out;
}

Tensor Softmax(const Tensor& x_in, int64_t axis) {
  if (x_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("Softmax: float32 only");
  }
  Tensor x = x_in.Contiguous();
  const int64_t r = x.rank();
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) throw std::runtime_error("Softmax: axis out of range");

  // Compute strides for the contiguous output.
  // We process in groups: outer * axis_dim * inner.
  int64_t axis_dim = x.shape()[axis];
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= x.shape()[i];
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < r; ++i) inner *= x.shape()[i];

  Tensor out = Tensor::Zeros(DType::kFloat32, x.shape());
  const float* p = x.data<float>();
  float* q = out.data<float>();

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t in = 0; in < inner; ++in) {
      const float* row_p = p + o * axis_dim * inner + in;
      float* row_q = q + o * axis_dim * inner + in;
      // 1) max for numerical stability
      float m = -INFINITY;
      for (int64_t k = 0; k < axis_dim; ++k) {
        float v = row_p[k * inner];
        if (v > m) m = v;
      }
      // 2) exp(x - max), sum
      float sum = 0.0f;
      for (int64_t k = 0; k < axis_dim; ++k) {
        float e = std::exp(row_p[k * inner] - m);
        row_q[k * inner] = e;
        sum += e;
      }
      // 3) divide
      const float inv = 1.0f / sum;
      for (int64_t k = 0; k < axis_dim; ++k) {
        row_q[k * inner] *= inv;
      }
    }
  }
  return out;
}

Tensor LayerNorm(const Tensor& x_in, const Tensor& scale_in, const Tensor& bias_in,
                 float eps, int normalized_dims) {
  if (x_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("LayerNorm: float32 only");
  }
  if (normalized_dims < 1 || normalized_dims > x_in.rank()) {
    throw std::runtime_error("LayerNorm: bad normalized_dims");
  }
  Tensor x = x_in.Contiguous();
  Tensor scale = scale_in.Contiguous();
  Tensor bias = bias_in.Contiguous();

  // The last `normalized_dims` axes flatten into a "feature" axis.
  int64_t feat = 1;
  for (int i = 0; i < normalized_dims; ++i) {
    feat *= x.shape()[x.rank() - 1 - i];
  }
  if (scale.numel() != feat || bias.numel() != feat) {
    throw std::runtime_error("LayerNorm: scale/bias size mismatch with feature dim");
  }
  int64_t outer = x.numel() / feat;

  Tensor out = Tensor::Zeros(DType::kFloat32, x.shape());
  const float* p = x.data<float>();
  const float* s = scale.data<float>();
  const float* b = bias.data<float>();
  float* q = out.data<float>();

  for (int64_t o = 0; o < outer; ++o) {
    const float* row_p = p + o * feat;
    float* row_q = q + o * feat;
    // mean
    float sum = 0.0f;
    for (int64_t i = 0; i < feat; ++i) sum += row_p[i];
    const float mean = sum / static_cast<float>(feat);
    // var
    float sq = 0.0f;
    for (int64_t i = 0; i < feat; ++i) {
      float d = row_p[i] - mean;
      sq += d * d;
    }
    const float var = sq / static_cast<float>(feat);
    const float inv_std = 1.0f / std::sqrt(var + eps);
    // scale + shift
    for (int64_t i = 0; i < feat; ++i) {
      row_q[i] = (row_p[i] - mean) * inv_std * s[i] + b[i];
    }
  }
  return out;
}

Tensor ReduceMean(const Tensor& x_in, const std::vector<int64_t>& axes_in,
                  bool keepdims) {
  if (x_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("ReduceMean: float32 only");
  }
  Tensor x = x_in.Contiguous();
  const int64_t r = x.rank();
  std::set<int64_t> ax_set;
  if (axes_in.empty()) {
    for (int64_t i = 0; i < r; ++i) ax_set.insert(i);
  } else {
    for (auto a : axes_in) ax_set.insert(a < 0 ? a + r : a);
  }

  // Output shape.
  Shape out_shape;
  for (int64_t i = 0; i < r; ++i) {
    if (ax_set.count(i)) {
      if (keepdims) out_shape.push_back(1);
    } else {
      out_shape.push_back(x.shape()[i]);
    }
  }

  Tensor out = Tensor::Zeros(DType::kFloat32, out_shape);
  const float* p = x.data<float>();
  float* q = out.data<float>();

  // Generic reducer: for each input element, compute the corresponding output
  // index by zeroing-out reduced dims (or dropping them) and accumulate.
  Shape in_strides = ContiguousStrides(x.shape());
  // For output: build strides that match either kept or reduced layout.
  // Easier: compute output index per input index lexicographically.
  IndexIterator it(x.shape());
  Shape idx;
  int64_t reduce_count = 1;
  for (auto a : ax_set) reduce_count *= x.shape()[a];

  while (it.Next(&idx)) {
    Shape out_idx;
    out_idx.reserve(out_shape.size());
    for (int64_t i = 0; i < r; ++i) {
      if (ax_set.count(i)) {
        if (keepdims) out_idx.push_back(0);
      } else {
        out_idx.push_back(idx[i]);
      }
    }
    // Linearize out_idx with respect to out_shape.
    int64_t lin = 0;
    int64_t stride = 1;
    for (int i = static_cast<int>(out_shape.size()) - 1; i >= 0; --i) {
      lin += out_idx[i] * stride;
      stride *= out_shape[i];
    }
    // Linearize input idx.
    int64_t in_lin = 0;
    int64_t s = 1;
    for (int i = static_cast<int>(r) - 1; i >= 0; --i) {
      in_lin += idx[i] * s;
      s *= x.shape()[i];
    }
    q[lin] += p[in_lin];
  }
  // Divide.
  int64_t out_n = out.numel();
  for (int64_t i = 0; i < out_n; ++i) q[i] /= static_cast<float>(reduce_count);
  return out;
}

}  // namespace rt
}  // namespace inferc
