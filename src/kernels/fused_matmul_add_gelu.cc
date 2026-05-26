#include "kernels/fused_matmul_add_gelu.h"

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <stdexcept>

#include "util/parallel.h"

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
  Tensor out = Tensor::Uninit(DType::kFloat32, out_shape);  // sgemm+sweep fully write

  // Parallelize over M (output rows): each block does an independent sgemm on
  // its row range + the fused bias+GELU sweep on those rows. Row blocks write
  // disjoint output regions (no races). Splitting M=128 into ~8 blocks of 16
  // keeps each sub-GEMM above the AMX engagement threshold (Session 12) while
  // using all cores. GELU is exact: 0.5*x*(1+erf(x/sqrt2)). (A vectorized
  // erf-approximation was tried and measured slower — this op is sgemm-bound,
  // see CHALLENGES.md C11.)
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  const float* xd = x.data<float>();
  const float* wd = w.data<float>();
  float* od = out.data<float>();
  const float* b = bias.data<float>();
  const int ni = static_cast<int>(N), ki = static_cast<int>(K);
  par::ParallelFor(M, /*grain=*/16, [&](int64_t r0, int64_t r1) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(r1 - r0), ni, ki, 1.0f, xd + r0 * K, ki,
                wd, ni, 0.0f, od + r0 * N, ni);
    for (int64_t row = r0; row < r1; ++row) {
      float* yr = od + row * N;
      for (int64_t c = 0; c < N; ++c) {
        float v = yr[c] + b[c];
        yr[c] = v * 0.5f * (1.0f + std::erf(v * kInvSqrt2));
      }
    }
  });
  return out;
}

}  // namespace rt
}  // namespace inferc
