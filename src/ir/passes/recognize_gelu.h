#pragma once

#include "ir/graph.h"

namespace inferc {
namespace passes {

// Recognizes the Erf-decomposed GELU pattern emitted by `torch.jit` /
// `transformers` ONNX export and folds it into a single `Gelu` op.
//
// Pattern (output of the matched MatMul+bias-add, call this X):
//
//   X → Div(X, sqrt(2)) → Erf → Add(·, 1.0) ──┐
//   X ────────────────────────────────────── Mul(X, ·) → Mul(·, 0.5) → Y
//
// Replaced with:  X → Gelu(X) → Y
//
// All intermediate tensors must be single-use; otherwise the chain isn't
// safe to fold. Constants are matched by approximate value (sqrt(2)≈1.4142,
// 1.0, 0.5).
//
// Returns the number of GELU patterns folded.
int RecognizeGelu(Graph* graph);

}  // namespace passes
}  // namespace inferc
