#include "kernels/attention.h"

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "util/parallel.h"

namespace inferc {
namespace rt {

namespace {
// Per-output-axis stride of `in_shape` broadcast to [B,nH,S,S] (0 on broadcast
// axes), so the mask can be indexed per (b,h,i,j) regardless of its rank.
void MaskStrides(const Shape& in_shape, int64_t B, int64_t nH, int64_t S,
                 int64_t s[4]) {
  const int64_t out[4] = {B, nH, S, S};
  int64_t st = 1;
  s[0] = s[1] = s[2] = s[3] = 0;
  const int r = static_cast<int>(in_shape.size());
  for (int i = r - 1; i >= 0; --i) {
    const int od = 4 - (r - i);  // right-align input axis i to out axis od
    if (od >= 0) s[od] = (in_shape[i] == 1) ? 0 : st;
    st *= in_shape[i];
  }
  (void)out;
}
}  // namespace

Tensor FusedAttention(const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
                      const Tensor& mask_cond, int64_t head_dim, float fill) {
  Tensor q = q_in.Contiguous(), k = k_in.Contiguous(), v = v_in.Contiguous();
  if (q.dtype() != DType::kFloat32) throw std::runtime_error("FusedAttention: fp32 only");
  if (q.rank() != 3) throw std::runtime_error("FusedAttention: expect [B,S,H]");
  const int64_t B = q.shape()[0], S = q.shape()[1], H = q.shape()[2];
  if (head_dim <= 0 || H % head_dim != 0)
    throw std::runtime_error("FusedAttention: H not divisible by head_dim");
  const int64_t nH = H / head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  Tensor out = Tensor::Uninit(DType::kFloat32, {B, S, H});
  const float* Q = q.data<float>();
  const float* K = k.data<float>();
  const float* V = v.data<float>();
  float* O = out.data<float>();

  // Mask broadcast strides against [B,nH,S,S] (may be empty if no mask).
  // A default/absent Tensor has empty shape (rank 0) but numel()==1, so gate
  // on rank, not numel, to avoid dereferencing null storage.
  const bool have_mask = !mask_cond.shape().empty() && mask_cond.numel() > 0;
  Tensor mc = have_mask ? mask_cond.Contiguous() : Tensor();
  const uint8_t* M = have_mask ? mc.data<uint8_t>() : nullptr;
  int64_t ms[4] = {0, 0, 0, 0};
  if (have_mask) MaskStrides(mc.shape(), B, nH, S, ms);

  // One independent attention per (batch, head).
  par::ParallelFor(B * nH, /*grain=*/1, [&](int64_t bh0, int64_t bh1) {
    std::vector<float> scores(static_cast<size_t>(S * S));
    const int Si = static_cast<int>(S), di = static_cast<int>(head_dim),
              Hi = static_cast<int>(H);
    for (int64_t bh = bh0; bh < bh1; ++bh) {
      const int64_t b = bh / nH, h = bh % nH;
      const float* Qh = Q + b * S * H + h * head_dim;  // [S, dH], row-stride H
      const float* Kh = K + b * S * H + h * head_dim;
      const float* Vh = V + b * S * H + h * head_dim;
      float* Oh = O + b * S * H + h * head_dim;

      // scores[S,S] = scale * Qh · Khᵀ   (strided reads via lda/ldb = H)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Si, Si, di, scale,
                  Qh, Hi, Kh, Hi, 0.0f, scores.data(), Si);

      // mask + row softmax (numerically stable). exp is vectorized via vForce.
      for (int64_t i = 0; i < S; ++i) {
        float* row = scores.data() + i * S;
        if (have_mask) {
          for (int64_t j = 0; j < S; ++j) {
            const uint8_t c = M[b * ms[0] + h * ms[1] + i * ms[2] + j * ms[3]];
            if (c) row[j] = fill;
          }
        }
        float m = row[0];
        for (int64_t j = 1; j < S; ++j) m = row[j] > m ? row[j] : m;
        const float neg_m = -m;
        for (int64_t j = 0; j < S; ++j) row[j] += neg_m;  // row - max
        vvexpf(row, row, &Si);                            // exp, vectorized
        float sum = 0.0f;
        for (int64_t j = 0; j < S; ++j) sum += row[j];
        const float inv = 1.0f / sum;
        for (int64_t j = 0; j < S; ++j) row[j] *= inv;
      }

      // context[S,dH] = softmax · Vh, written straight into O (ldc = H)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Si, di, Si, 1.0f,
                  scores.data(), Si, Vh, Hi, 0.0f, Oh, Hi);
    }
  });
  return out;
}

}  // namespace rt
}  // namespace inferc
