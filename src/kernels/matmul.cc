#include "kernels/matmul.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

#include "util/parallel.h"

namespace inferc {
namespace rt {

namespace {

// Right-align broadcast of two shapes (used for leading "batch" dims).
Shape BroadcastBatchShapes(const Shape& a, const Shape& b) {
  const size_t r = std::max(a.size(), b.size());
  Shape out(r);
  for (size_t i = 0; i < r; ++i) {
    int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
    int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
    int64_t d;
    if (da == db || db == 1) d = da;
    else if (da == 1)        d = db;
    else throw std::runtime_error("MatMul: incompatible broadcast batch dims");
    out[r - 1 - i] = d;
  }
  return out;
}

// Step a multi-index in `batch_shape` and compute the linear "batch index"
// in a contiguous batch tensor of that shape. For broadcasting, the per-input
// stride along a broadcast dim is 0 (we re-use the same matrix).
//
// Given input batch shape `in_b` (right-aligned to `out_b`), returns the
// per-input index for a given output index.
int64_t MatchBroadcastIndex(const Shape& in_b, const Shape& out_b,
                            const Shape& out_idx) {
  // Right-align: input dim i corresponds to output dim (out_b.size()-in_b.size()+i).
  const int64_t skip = static_cast<int64_t>(out_b.size()) -
                       static_cast<int64_t>(in_b.size());
  int64_t lin = 0;
  int64_t stride = 1;
  for (int i = static_cast<int>(in_b.size()) - 1; i >= 0; --i) {
    int64_t out_dim = out_idx[skip + i];
    int64_t coord = (in_b[i] == 1) ? 0 : out_dim;
    lin += coord * stride;
    stride *= in_b[i];
  }
  return lin;
}

// AMX-aware decode dispatch toggle (default on). See matmul.h.
bool g_gemv_decode_enabled = true;

}  // namespace

void SetGemvDecodeEnabled(bool on) { g_gemv_decode_enabled = on; }
bool GemvDecodeEnabled() { return g_gemv_decode_enabled; }

Tensor MatMul(const Tensor& a_in, const Tensor& b_in) {
  if (a_in.dtype() != DType::kFloat32 || b_in.dtype() != DType::kFloat32) {
    throw std::runtime_error("MatMul: only float32 supported in v1");
  }
  if (a_in.rank() < 1 || b_in.rank() < 1) {
    throw std::runtime_error("MatMul: tensors must be at least 1D");
  }

  // Promote 1D to 2D (numpy-style): a_1d -> [1, K], b_1d -> [K, 1].
  Tensor a = a_in.Contiguous();
  Tensor b = b_in.Contiguous();
  bool a_was_1d = false, b_was_1d = false;
  if (a.rank() == 1) {
    Shape s = a.shape(); s.insert(s.begin(), 1);
    a = Tensor::FromHostBytes(DType::kFloat32, s, a.bytes());
    a_was_1d = true;
  }
  if (b.rank() == 1) {
    Shape s = b.shape(); s.push_back(1);
    b = Tensor::FromHostBytes(DType::kFloat32, s, b.bytes());
    b_was_1d = true;
  }

  // Split each shape into [batch_dims..., M, K] / [batch_dims..., K, N].
  Shape a_batch(a.shape().begin(), a.shape().end() - 2);
  Shape b_batch(b.shape().begin(), b.shape().end() - 2);
  const int64_t M = a.shape()[a.rank() - 2];
  const int64_t K = a.shape()[a.rank() - 1];
  const int64_t Kb = b.shape()[b.rank() - 2];
  const int64_t N = b.shape()[b.rank() - 1];
  if (K != Kb) {
    throw std::runtime_error("MatMul: inner dim mismatch (K!=K)");
  }

  Shape out_batch = BroadcastBatchShapes(a_batch, b_batch);
  Shape out_shape = out_batch;
  out_shape.push_back(M);
  out_shape.push_back(N);
  Tensor out = Tensor::Uninit(DType::kFloat32, out_shape);

  // Iterate over the output batch grid, calling sgemm on each MxK x KxN slice.
  // For batch=() (2D inputs), the loop runs once.
  int64_t batch_count = 1;
  for (auto d : out_batch) batch_count *= d;

  const float* a_data = a.data<float>();
  const float* b_data = b.data<float>();
  float* out_data = out.data<float>();
  const int64_t a_mat_size = M * K;
  const int64_t b_mat_size = K * N;
  const int64_t out_mat_size = M * N;

  if (batch_count == 1) {
    // Single sgemm. (Splitting M across cores was measured net-negative here —
    // the projection GEMMs are already fast and 24 dispatches/inference cost
    // more than they save; Accelerate's own GEMM is efficient. See C13.)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f, a_data, static_cast<int>(K),
                b_data, static_cast<int>(N),
                0.0f, out_data, static_cast<int>(N));
  } else {
    // Batched matmul: each batch slice is an independent sgemm into a disjoint
    // output region — parallelize across the batch (e.g. attention heads).
    const int mi = static_cast<int>(M), ni = static_cast<int>(N),
              ki = static_cast<int>(K);
    par::ParallelFor(batch_count, /*grain=*/1, [&](int64_t bb0, int64_t bb1) {
      Shape idx(out_batch.size());
      for (int64_t ob = bb0; ob < bb1; ++ob) {
        // Decompose linear batch index ob into a multi-index in out_batch.
        int64_t rem = ob;
        for (int i = static_cast<int>(out_batch.size()) - 1; i >= 0; --i) {
          idx[i] = rem % out_batch[i];
          rem /= out_batch[i];
        }
        int64_t a_b = MatchBroadcastIndex(a_batch, out_batch, idx);
        int64_t b_b = MatchBroadcastIndex(b_batch, out_batch, idx);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mi, ni, ki, 1.0f,
                    a_data + a_b * a_mat_size, ki, b_data + b_b * b_mat_size, ni,
                    0.0f, out_data + ob * out_mat_size, ni);
      }
    });
  }

  // Squeeze back the implicit 1D-to-2D promotions.
  if (a_was_1d) {
    Shape s = out.shape();
    s.erase(s.end() - 2);
    out = Tensor::FromHostBytes(DType::kFloat32, s, out.bytes());
  }
  if (b_was_1d) {
    Shape s = out.shape();
    s.pop_back();
    out = Tensor::FromHostBytes(DType::kFloat32, s, out.bytes());
  }
  return out;
}

Tensor Gemm(const Tensor& a_in, const Tensor& b_in, const Tensor* c,
            float alpha, float beta, bool trans_a, bool trans_b) {
  if (a_in.rank() != 2 || b_in.rank() != 2) {
    throw std::runtime_error("Gemm: 2D inputs only");
  }
  Tensor a = a_in.Contiguous();
  Tensor b = b_in.Contiguous();
  const int64_t M = trans_a ? a.shape()[1] : a.shape()[0];
  const int64_t K = trans_a ? a.shape()[0] : a.shape()[1];
  const int64_t Kb = trans_b ? b.shape()[1] : b.shape()[0];
  const int64_t N = trans_b ? b.shape()[0] : b.shape()[1];
  if (K != Kb) throw std::runtime_error("Gemm: K mismatch");

  Tensor out = Tensor::Zeros(DType::kFloat32, {M, N});
  // Initialize with beta * C if provided.
  if (c != nullptr && beta != 0.0f) {
    Tensor cc = c->Contiguous();
    if (cc.numel() == M * N) {
      // Direct copy.
      std::memcpy(out.bytes(), cc.bytes(), out.byte_size());
    } else if (cc.shape().size() == 1 && cc.shape()[0] == N) {
      // Broadcast bias [N] across rows.
      const float* src = cc.data<float>();
      float* dst = out.data<float>();
      for (int64_t i = 0; i < M; ++i) {
        std::memcpy(dst + i * N, src, N * sizeof(float));
      }
    } else if (cc.shape().size() == 1 && cc.shape()[0] == 1) {
      std::fill_n(out.data<float>(), M * N, cc.data<float>()[0]);
    } else {
      throw std::runtime_error("Gemm: unsupported C shape for broadcast");
    }
  }

  // AMX-aware decode dispatch: a single-row Gemm (M == 1) is the autoregressive
  // decode shape. We route it through cblas_sgemv ONLY when the weight is stored
  // [N, K] (trans_b == true), where the matrix-vector product is the contiguous
  // CblasNoTrans form `y = B·x` — exactly the sgemv variant Session 12 measured
  // faster than a one-row sgemm on M1. For trans_b == false the weight is [K, N]
  // and sgemv would need CblasTrans (column-strided access), which Accelerate
  // does NOT accelerate and which measured *slower* than sgemm — so we leave
  // those on sgemm. (GPT-2's projections are all trans_b == false, so its decode
  // stays on sgemm; a transB=1 export would take the fast sgemv path. Pre-
  // transposing the weight to unlock sgemv is future work.) `out` is pre-filled
  // with C, so sgemv's beta*y term reproduces Gemm's beta*C. The activation row
  // is K contiguous floats for either trans_a, so trans_a needs no special case.
  if (g_gemv_decode_enabled && M == 1 && trans_b) {
    const float* x = a.data<float>();  // length-K activation vector
    const float* Bp = b.data<float>();
    float* y = out.data<float>();      // length-N output, pre-filled with C
    // B is [N, K] row-major (op(B) = Bᵀ); want y = alpha * B·x + beta*y.
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(N),
                static_cast<int>(K), alpha, Bp, static_cast<int>(K), x, 1,
                beta, y, 1);
    return out;
  }

  cblas_sgemm(CblasRowMajor,
              trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans,
              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
              alpha, a.data<float>(),
              static_cast<int>(trans_a ? M : K),
              b.data<float>(),
              static_cast<int>(trans_b ? K : N),
              beta, out.data<float>(), static_cast<int>(N));
  return out;
}

}  // namespace rt
}  // namespace inferc
