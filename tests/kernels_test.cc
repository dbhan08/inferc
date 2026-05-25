#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/activation.h"
#include "kernels/elementwise.h"
#include "kernels/embedding.h"
#include "kernels/matmul.h"
#include "kernels/movement.h"
#include "runtime/tensor.h"

namespace {

using inferc::DType;
using inferc::Shape;
using inferc::rt::Tensor;
namespace rt = inferc::rt;

Tensor MakeF32(Shape s, std::vector<float> data) {
  Tensor t(DType::kFloat32, s);
  std::copy(data.begin(), data.end(), t.data<float>());
  return t;
}

Tensor MakeI64(Shape s, std::vector<int64_t> data) {
  Tensor t(DType::kInt64, s);
  std::copy(data.begin(), data.end(), t.data<int64_t>());
  return t;
}

}  // namespace

// ===================== Tensor =====================

TEST(Tensor, ZerosAndShape) {
  Tensor t = Tensor::Zeros(DType::kFloat32, {2, 3});
  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.byte_size(), 24);
  EXPECT_TRUE(t.is_contiguous());
  for (int i = 0; i < 6; ++i) EXPECT_EQ(t.data<float>()[i], 0.0f);
}

TEST(Tensor, ContiguousMakesCopy) {
  Tensor a = MakeF32({2, 2}, {1, 2, 3, 4});
  Tensor b = a.Contiguous();
  b.data<float>()[0] = 99.0f;
  EXPECT_EQ(a.data<float>()[0], 1.0f) << "Contiguous should not alias";
}

// ===================== MatMul =====================

TEST(MatMul, Basic2D) {
  // [2,3] @ [3,2] = [2,2]
  Tensor a = MakeF32({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = MakeF32({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor c = rt::MatMul(a, b);
  ASSERT_EQ(c.shape(), Shape({2, 2}));
  // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
  // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
  EXPECT_NEAR(c.data<float>()[0], 22.0f, 1e-5);
  EXPECT_NEAR(c.data<float>()[1], 28.0f, 1e-5);
  EXPECT_NEAR(c.data<float>()[2], 49.0f, 1e-5);
  EXPECT_NEAR(c.data<float>()[3], 64.0f, 1e-5);
}

TEST(MatMul, BatchedBroadcast) {
  // [2, 2, 3] @ [3, 4] -> [2, 2, 4]
  std::vector<float> a_data(2 * 2 * 3);
  for (int i = 0; i < (int)a_data.size(); ++i) a_data[i] = static_cast<float>(i + 1);
  std::vector<float> b_data(3 * 4);
  for (int i = 0; i < (int)b_data.size(); ++i) b_data[i] = static_cast<float>(i + 1);
  Tensor a = MakeF32({2, 2, 3}, a_data);
  Tensor b = MakeF32({3, 4}, b_data);
  Tensor c = rt::MatMul(a, b);
  ASSERT_EQ(c.shape(), Shape({2, 2, 4}));
  // Spot-check first batch, first row: a[0,0,:] = [1,2,3], b cols give [38, 44, 50, 56]
  EXPECT_NEAR(c.data<float>()[0], 38.0f, 1e-4);
  EXPECT_NEAR(c.data<float>()[1], 44.0f, 1e-4);
}

TEST(Gemm, BiasAdded) {
  Tensor a = MakeF32({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = MakeF32({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor bias = MakeF32({2}, {10, 20});
  Tensor c = rt::Gemm(a, b, &bias, 1.0f, 1.0f, false, false);
  // 22+10, 28+20, 49+10, 64+20
  EXPECT_NEAR(c.data<float>()[0], 32.0f, 1e-5);
  EXPECT_NEAR(c.data<float>()[1], 48.0f, 1e-5);
  EXPECT_NEAR(c.data<float>()[2], 59.0f, 1e-5);
  EXPECT_NEAR(c.data<float>()[3], 84.0f, 1e-5);
}

// ---- AMX-aware decode dispatch (Session 13): M==1 Gemm routes to cblas_sgemv.
// The GEMV path must be numerically identical to the sgemm path it replaces.

// trans_b=false: the GEMV path is gated OFF (sgemv would need slow CblasTrans),
// so both toggle states use sgemm and must agree with the hand-computed value.
TEST(Gemm, DecodeGemvMatchesSgemm_NoTransB) {
  Tensor a = MakeF32({1, 3}, {1, 2, 3});          // single-row activation
  Tensor b = MakeF32({3, 2}, {1, 2, 3, 4, 5, 6}); // [K,N], transB=0
  Tensor bias = MakeF32({2}, {10, 20});
  // out[0] = 1*1+2*3+3*5 + 10 = 22+10 = 32 ; out[1] = 1*2+2*4+3*6 + 20 = 28+20 = 48
  rt::SetGemvDecodeEnabled(true);
  Tensor g = rt::Gemm(a, b, &bias, 1.0f, 1.0f, false, false);
  rt::SetGemvDecodeEnabled(false);
  Tensor s = rt::Gemm(a, b, &bias, 1.0f, 1.0f, false, false);
  rt::SetGemvDecodeEnabled(true);  // restore default
  EXPECT_NEAR(g.data<float>()[0], 32.0f, 1e-5);
  EXPECT_NEAR(g.data<float>()[1], 48.0f, 1e-5);
  EXPECT_NEAR(g.data<float>()[0], s.data<float>()[0], 1e-5);
  EXPECT_NEAR(g.data<float>()[1], s.data<float>()[1], 1e-5);
}

// transB=1: weight stored [N,K]. GEMV path branches on trans_b, must still match.
TEST(Gemm, DecodeGemvMatchesSgemm_TransB) {
  Tensor a = MakeF32({1, 3}, {1, 2, 3});            // [1,K]
  Tensor b = MakeF32({2, 3}, {1, 3, 5, 2, 4, 6});   // [N,K], transB=1 -> op(B)=[K,N]
  // Equivalent to the no-trans case above (same logical weight), so same out.
  rt::SetGemvDecodeEnabled(true);
  Tensor g = rt::Gemm(a, b, nullptr, 1.0f, 0.0f, false, true);
  rt::SetGemvDecodeEnabled(false);
  Tensor s = rt::Gemm(a, b, nullptr, 1.0f, 0.0f, false, true);
  rt::SetGemvDecodeEnabled(true);
  ASSERT_EQ(g.numel(), 2);
  EXPECT_NEAR(g.data<float>()[0], 22.0f, 1e-5);  // 1*1+2*3+3*5
  EXPECT_NEAR(g.data<float>()[1], 28.0f, 1e-5);  // 1*2+2*4+3*6
  EXPECT_NEAR(g.data<float>()[0], s.data<float>()[0], 1e-5);
  EXPECT_NEAR(g.data<float>()[1], s.data<float>()[1], 1e-5);
}

// alpha/beta scaling must be honored on the GEMV path.
TEST(Gemm, DecodeGemvAlphaBeta) {
  Tensor a = MakeF32({1, 2}, {1, 1});
  Tensor b = MakeF32({2, 2}, {1, 2, 3, 4});
  Tensor c = MakeF32({2}, {100, 100});
  // alpha*(A*B) + beta*C = 2*[4,6] + 0.5*[100,100] = [8,12] + [50,50] = [58,62]
  rt::SetGemvDecodeEnabled(true);
  Tensor g = rt::Gemm(a, b, &c, 2.0f, 0.5f, false, false);
  rt::SetGemvDecodeEnabled(false);
  Tensor s = rt::Gemm(a, b, &c, 2.0f, 0.5f, false, false);
  rt::SetGemvDecodeEnabled(true);
  EXPECT_NEAR(g.data<float>()[0], 58.0f, 1e-4);
  EXPECT_NEAR(g.data<float>()[1], 62.0f, 1e-4);
  EXPECT_NEAR(g.data<float>()[0], s.data<float>()[0], 1e-4);
  EXPECT_NEAR(g.data<float>()[1], s.data<float>()[1], 1e-4);
}

// ===================== Elementwise =====================

TEST(Elementwise, AddNoBroadcast) {
  Tensor a = MakeF32({2, 2}, {1, 2, 3, 4});
  Tensor b = MakeF32({2, 2}, {10, 20, 30, 40});
  Tensor c = rt::Add(a, b);
  EXPECT_NEAR(c.data<float>()[0], 11.0f, 1e-6);
  EXPECT_NEAR(c.data<float>()[3], 44.0f, 1e-6);
}

TEST(Elementwise, BroadcastScalar) {
  Tensor a = MakeF32({3}, {1, 2, 3});
  Tensor b = MakeF32({}, {10});  // scalar
  Tensor c = rt::Add(a, b);
  ASSERT_EQ(c.shape(), Shape({3}));
  EXPECT_NEAR(c.data<float>()[0], 11.0f, 1e-6);
  EXPECT_NEAR(c.data<float>()[2], 13.0f, 1e-6);
}

TEST(Elementwise, BroadcastRowAndColumn) {
  Tensor a = MakeF32({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = MakeF32({3}, {10, 20, 30});  // broadcasts across rows
  Tensor c = rt::Add(a, b);
  ASSERT_EQ(c.shape(), Shape({2, 3}));
  EXPECT_NEAR(c.data<float>()[0], 11, 1e-6);
  EXPECT_NEAR(c.data<float>()[5], 36, 1e-6);
}

TEST(Elementwise, MulSubDivPow) {
  Tensor a = MakeF32({2}, {2, 3});
  Tensor b = MakeF32({2}, {4, 5});
  EXPECT_NEAR(rt::Mul(a, b).data<float>()[0], 8.0f, 1e-6);
  EXPECT_NEAR(rt::Sub(a, b).data<float>()[1], -2.0f, 1e-6);
  EXPECT_NEAR(rt::Div(b, a).data<float>()[0], 2.0f, 1e-6);
  EXPECT_NEAR(rt::Pow(a, b).data<float>()[0], 16.0f, 1e-5);
}

// ===================== Gelu / Softmax / LayerNorm / ReduceMean =====================

TEST(Gelu, KnownValues) {
  // GELU(0) = 0, GELU(1) ≈ 0.8413, GELU(-1) ≈ -0.1587
  Tensor x = MakeF32({3}, {0.0f, 1.0f, -1.0f});
  Tensor y = rt::Gelu(x);
  EXPECT_NEAR(y.data<float>()[0], 0.0f, 1e-5);
  EXPECT_NEAR(y.data<float>()[1], 0.8413447f, 1e-5);
  EXPECT_NEAR(y.data<float>()[2], -0.1586553f, 1e-5);
}

TEST(Softmax, NumericallyStable) {
  Tensor x = MakeF32({1, 4}, {1000.0f, 1001.0f, 999.0f, 1002.0f});
  Tensor y = rt::Softmax(x, -1);
  // Should not be NaN/inf despite huge inputs.
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(std::isfinite(y.data<float>()[i]));
  // Softmax sums to 1.
  float sum = 0;
  for (int i = 0; i < 4; ++i) sum += y.data<float>()[i];
  EXPECT_NEAR(sum, 1.0f, 1e-5);
  // Largest input (1002) gets the highest probability.
  EXPECT_GT(y.data<float>()[3], y.data<float>()[2]);
  EXPECT_GT(y.data<float>()[3], y.data<float>()[1]);
}

TEST(Softmax, AxisMidDim) {
  // Softmax along axis=1 of a [2, 3, 2] tensor.
  std::vector<float> data;
  for (int i = 0; i < 12; ++i) data.push_back(static_cast<float>(i));
  Tensor x = MakeF32({2, 3, 2}, data);
  Tensor y = rt::Softmax(x, 1);
  // For each (b, i), sum across the axis dim should be 1.
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 2; ++i) {
      float sum = 0;
      for (int k = 0; k < 3; ++k) sum += y.data<float>()[b * 6 + k * 2 + i];
      EXPECT_NEAR(sum, 1.0f, 1e-5);
    }
  }
}

TEST(LayerNorm, MeanZeroVarOne) {
  // After LN with scale=1 and bias=0, output has mean ~0 and var ~1 along last dim.
  Tensor x = MakeF32({2, 4}, {1, 2, 3, 4,   10, 20, 30, 40});
  Tensor scale = MakeF32({4}, {1, 1, 1, 1});
  Tensor bias = MakeF32({4}, {0, 0, 0, 0});
  Tensor y = rt::LayerNorm(x, scale, bias, 1e-5f, 1);
  for (int row = 0; row < 2; ++row) {
    float mean = 0;
    for (int i = 0; i < 4; ++i) mean += y.data<float>()[row * 4 + i];
    mean /= 4;
    EXPECT_NEAR(mean, 0.0f, 1e-4);
    float var = 0;
    for (int i = 0; i < 4; ++i) {
      float d = y.data<float>()[row * 4 + i] - mean;
      var += d * d;
    }
    var /= 4;
    EXPECT_NEAR(var, 1.0f, 5e-3);  // small slack for eps
  }
}

TEST(ReduceMean, AlongAxis) {
  // Mean along axis=1 of [2, 3] = [2, 1] with keepdims, or [2] without.
  Tensor x = MakeF32({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor y = rt::ReduceMean(x, {1}, true);
  ASSERT_EQ(y.shape(), Shape({2, 1}));
  EXPECT_NEAR(y.data<float>()[0], 2.0f, 1e-6);
  EXPECT_NEAR(y.data<float>()[1], 5.0f, 1e-6);
  Tensor z = rt::ReduceMean(x, {1}, false);
  ASSERT_EQ(z.shape(), Shape({2}));
}

// ===================== Gather =====================

TEST(Gather, RowsFromTable) {
  // data: [4, 3] table, indices = [0, 2] -> output [2, 3]
  Tensor data = MakeF32({4, 3}, {1,1,1,  2,2,2,  3,3,3,  4,4,4});
  Tensor idx = MakeI64({2}, {0, 2});
  Tensor y = rt::Gather(data, idx, 0);
  ASSERT_EQ(y.shape(), Shape({2, 3}));
  EXPECT_NEAR(y.data<float>()[0], 1.0f, 0);  // row 0
  EXPECT_NEAR(y.data<float>()[3], 3.0f, 0);  // row 2
}

TEST(Gather, MatrixIndices) {
  // data [3, 2], indices [2, 2] (matrix of int64) -> [2, 2, 2]
  Tensor data = MakeF32({3, 2}, {1,2,  3,4,  5,6});
  Tensor idx = MakeI64({2, 2}, {0, 1, 2, 0});
  Tensor y = rt::Gather(data, idx, 0);
  ASSERT_EQ(y.shape(), Shape({2, 2, 2}));
  // y[0,0,:] = data[0] = [1,2]; y[1,1,:] = data[0] = [1,2]
  EXPECT_NEAR(y.data<float>()[0], 1.0f, 0);
  EXPECT_NEAR(y.data<float>()[7], 2.0f, 0);
}

// ===================== Movement =====================

TEST(Reshape, FlattenWithMinusOne) {
  Tensor x = MakeF32({2, 3, 4}, std::vector<float>(24, 1.0f));
  Tensor y = rt::Reshape(x, {6, -1});
  ASSERT_EQ(y.shape(), Shape({6, 4}));
}

TEST(Transpose, SwapsDims) {
  Tensor x = MakeF32({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor y = rt::Transpose(x, {1, 0});
  ASSERT_EQ(y.shape(), Shape({3, 2}));
  // x[i,j] -> y[j,i]
  EXPECT_NEAR(y.data<float>()[0], 1.0f, 0);   // y[0,0]=x[0,0]=1
  EXPECT_NEAR(y.data<float>()[1], 4.0f, 0);   // y[0,1]=x[1,0]=4
  EXPECT_NEAR(y.data<float>()[5], 6.0f, 0);   // y[2,1]=x[1,2]=6
}

TEST(Concat, AlongFirstAxis) {
  Tensor a = MakeF32({1, 3}, {1, 2, 3});
  Tensor b = MakeF32({2, 3}, {10, 20, 30, 40, 50, 60});
  Tensor y = rt::Concat({a, b}, 0);
  ASSERT_EQ(y.shape(), Shape({3, 3}));
  EXPECT_NEAR(y.data<float>()[0], 1.0f, 0);
  EXPECT_NEAR(y.data<float>()[3], 10.0f, 0);
  EXPECT_NEAR(y.data<float>()[8], 60.0f, 0);
}

TEST(Slice, MiddleSlice) {
  Tensor x = MakeF32({5}, {0, 1, 2, 3, 4});
  Tensor y = rt::Slice(x, {1}, {4});
  ASSERT_EQ(y.shape(), Shape({3}));
  EXPECT_NEAR(y.data<float>()[0], 1.0f, 0);
  EXPECT_NEAR(y.data<float>()[2], 3.0f, 0);
}

TEST(Cast, Int64ToFloat32) {
  Tensor x = MakeI64({3}, {1, 2, 3});
  Tensor y = rt::Cast(x, DType::kFloat32);
  ASSERT_EQ(y.dtype(), DType::kFloat32);
  EXPECT_NEAR(y.data<float>()[0], 1.0f, 0);
  EXPECT_NEAR(y.data<float>()[2], 3.0f, 0);
}

TEST(Unsqueeze, InsertsAxis) {
  Tensor x = MakeF32({3}, {1, 2, 3});
  Tensor y = rt::Unsqueeze(x, {0});
  ASSERT_EQ(y.shape(), Shape({1, 3}));
  Tensor z = rt::Unsqueeze(x, {1});
  ASSERT_EQ(z.shape(), Shape({3, 1}));
}

TEST(ConstantOfShape, FillsWithValue) {
  std::vector<uint8_t> fill(sizeof(float));
  float v = 3.5f;
  std::memcpy(fill.data(), &v, sizeof(float));
  Tensor out = rt::ConstantOfShape({2, 3}, DType::kFloat32, fill);
  ASSERT_EQ(out.shape(), Shape({2, 3}));
  for (int64_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(out.data<float>()[i], 3.5f);
}

TEST(ConstantOfShape, ZeroFillWhenEmpty) {
  Tensor out = rt::ConstantOfShape({2, 2}, DType::kFloat32, {});
  for (int64_t i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out.data<float>()[i], 0.0f);
}

TEST(Split, EvenSplitAxis0) {
  Tensor x = MakeF32({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  auto parts = rt::Split(x, /*axis=*/0, {2, 2});
  ASSERT_EQ(parts.size(), 2u);
  ASSERT_EQ(parts[0].shape(), Shape({2, 2}));
  EXPECT_FLOAT_EQ(parts[0].data<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(parts[0].data<float>()[3], 4.0f);
  EXPECT_FLOAT_EQ(parts[1].data<float>()[0], 5.0f);
  EXPECT_FLOAT_EQ(parts[1].data<float>()[3], 8.0f);
}

TEST(Split, UnevenSplitAxis1) {
  // Shape [2, 5], split along axis=1 into sizes [2, 3].
  Tensor x = MakeF32({2, 5}, {1, 2, 3, 4, 5,
                              6, 7, 8, 9, 10});
  auto parts = rt::Split(x, /*axis=*/1, {2, 3});
  ASSERT_EQ(parts.size(), 2u);
  ASSERT_EQ(parts[0].shape(), Shape({2, 2}));
  ASSERT_EQ(parts[1].shape(), Shape({2, 3}));
  // First part row 0: {1, 2}; row 1: {6, 7}
  EXPECT_FLOAT_EQ(parts[0].data<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(parts[0].data<float>()[1], 2.0f);
  EXPECT_FLOAT_EQ(parts[0].data<float>()[2], 6.0f);
  EXPECT_FLOAT_EQ(parts[0].data<float>()[3], 7.0f);
  // Second part row 0: {3, 4, 5}; row 1: {8, 9, 10}
  EXPECT_FLOAT_EQ(parts[1].data<float>()[0], 3.0f);
  EXPECT_FLOAT_EQ(parts[1].data<float>()[2], 5.0f);
  EXPECT_FLOAT_EQ(parts[1].data<float>()[5], 10.0f);
}

TEST(Split, ThreeWayQKV) {
  // Simulates GPT-2's Q/K/V split: [1, 1, 6] split along axis=2 into 3 parts of 2.
  Tensor x = MakeF32({1, 1, 6}, {1, 2, 3, 4, 5, 6});
  auto parts = rt::Split(x, /*axis=*/2, {2, 2, 2});
  ASSERT_EQ(parts.size(), 3u);
  EXPECT_FLOAT_EQ(parts[0].data<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(parts[0].data<float>()[1], 2.0f);
  EXPECT_FLOAT_EQ(parts[1].data<float>()[0], 3.0f);
  EXPECT_FLOAT_EQ(parts[1].data<float>()[1], 4.0f);
  EXPECT_FLOAT_EQ(parts[2].data<float>()[0], 5.0f);
  EXPECT_FLOAT_EQ(parts[2].data<float>()[1], 6.0f);
}

TEST(Range, Int64BasicPositiveDelta) {
  Tensor r = rt::RangeI64(0, 5, 1);
  ASSERT_EQ(r.shape(), Shape({5}));
  ASSERT_EQ(r.dtype(), DType::kInt64);
  for (int64_t i = 0; i < 5; ++i) EXPECT_EQ(r.data<int64_t>()[i], i);
}

TEST(Range, Int64NonUnitDelta) {
  Tensor r = rt::RangeI64(10, 22, 3);  // 10, 13, 16, 19
  ASSERT_EQ(r.shape(), Shape({4}));
  EXPECT_EQ(r.data<int64_t>()[0], 10);
  EXPECT_EQ(r.data<int64_t>()[1], 13);
  EXPECT_EQ(r.data<int64_t>()[2], 16);
  EXPECT_EQ(r.data<int64_t>()[3], 19);
}

TEST(Range, Float32) {
  Tensor r = rt::RangeF32(0.0f, 1.0f, 0.25f);  // 0.0, 0.25, 0.5, 0.75
  ASSERT_EQ(r.shape(), Shape({4}));
  ASSERT_EQ(r.dtype(), DType::kFloat32);
  EXPECT_FLOAT_EQ(r.data<float>()[0], 0.0f);
  EXPECT_FLOAT_EQ(r.data<float>()[1], 0.25f);
  EXPECT_FLOAT_EQ(r.data<float>()[2], 0.5f);
  EXPECT_FLOAT_EQ(r.data<float>()[3], 0.75f);
}
