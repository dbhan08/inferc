#pragma once

#include "ir/graph.h"

namespace inferc {
namespace passes {

// Flagship fusion: collapses `MatMul → Add(bias) → Gelu` into a single
// `FusedMatMulAddGELU` op in the "inferc" custom domain.
//
// Requirements for a match:
//   - Both intermediate tensors (MatMul output, Add output) are single-use.
//   - MatMul has 2 inputs (no bias parameter exists in stock ONNX MatMul).
//   - Add has 2 inputs: one is the MatMul output, the other is *some* tensor
//     (typically a bias initializer; the kernel handles arbitrary broadcast).
//
// The fused op's inputs are (X, W, bias) where X = MatMul.inputs[0],
// W = MatMul.inputs[1], bias = the non-MatMul input to Add.
//
// Returns the number of fusion patterns applied.
int FuseMatMulAddGelu(Graph* graph);

}  // namespace passes
}  // namespace inferc
