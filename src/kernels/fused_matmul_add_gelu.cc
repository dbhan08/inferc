#include "kernels/fused_matmul_add_gelu.h"

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <stdexcept>

namespace inferc {
namespace rt {

Tensor FusedMatMulAddGELU(const Tensor& x_in, const Tensor& w_in,
                          const Tensor& bias_in) {
  if (x_in.dtype() != DType::kFloat32 || w_in.dtype() != DType::kFloat32 ||
      bias_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("FusedMatMulAddGELU: float32 only");
  }
  if (w_in.rank() != 2) {
    throw std::runtime_error("FusedMatMulAddGELU: W must be 2D [K, N]");
  }
  if (bias_in.rank() != 1) {
    throw std::runtime_error("FusedMatMulAddGELU: bias must be 1D [N]");
  }
  const Tensor x = x_in.Contiguous();
  const Tensor w = w_in.Contiguous();
  const Tensor bias = bias_in.Contiguous();

  if (x.rank() < 2) {
    throw std::runtime_error("FusedMatMulAddGELU: X must be at least 2D");
  }
  const int64_t K = w.shape()[0];
  const int64_t N = w.shape()[1];
  if (x.shape()[x.rank() - 1] != K) {
    throw std::runtime_error("FusedMatMulAddGELU: K mismatch between X and W");
  }
  if (bias.shape()[0] != N) {
    throw std::runtime_error("FusedMatMulAddGELU: bias size must equal N");
  }

  // Flatten X's leading dims into one "M" axis for a single sgemm.
  int64_t M = 1;
  for (int i = 0; i < x.rank() - 1; ++i) M *= x.shape()[i];

  Shape out_shape = x.shape();
  out_shape.back() = N;
  Tensor out = Tensor::Zeros(DType::kFloat32, out_shape);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
              1.0f, x.data<float>(), static_cast<int>(K),
              w.data<float>(), static_cast<int>(N),
              0.0f, out.data<float>(), static_cast<int>(N));

  // Single fused sweep: bias-add (broadcasted along last dim) + exact GELU.
  // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  float* y = out.data<float>();
  const float* b = bias.data<float>();
  for (int64_t row = 0; row < M; ++row) {
    float* yr = y + row * N;
    for (int64_t c = 0; c < N; ++c) {
      float v = yr[c] + b[c];
      yr[c] = v * 0.5f * (1.0f + std::erf(v * kInvSqrt2));
    }
  }
  return out;
}

}  // namespace rt
}  // namespace inferc
