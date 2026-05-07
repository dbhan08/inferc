#pragma once

#include "runtime/tensor.h"

namespace inferc {
namespace rt {

// Numpy-style elementwise binary ops with broadcasting on float32.
// Output shape is the right-aligned broadcast of input shapes.
Tensor Add(const Tensor& a, const Tensor& b);
Tensor Sub(const Tensor& a, const Tensor& b);
Tensor Mul(const Tensor& a, const Tensor& b);
Tensor Div(const Tensor& a, const Tensor& b);
Tensor Pow(const Tensor& a, const Tensor& b);

// Comparison: bool output.
Tensor Equal(const Tensor& a, const Tensor& b);

// 3-way select: bool condition, x and y same dtype.
Tensor Where(const Tensor& cond, const Tensor& x, const Tensor& y);

// Unary pointwise.
Tensor Sqrt(const Tensor& a);
Tensor Erf(const Tensor& a);
Tensor Relu(const Tensor& a);
Tensor Tanh(const Tensor& a);
Tensor Neg(const Tensor& a);
Tensor Abs(const Tensor& a);

}  // namespace rt
}  // namespace inferc
