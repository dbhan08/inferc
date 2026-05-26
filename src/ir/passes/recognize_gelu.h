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

// Recognizes the *tanh approximation* of GELU that GPT-2 uses, folding the
// 8-node decomposition into a single `GeluTanh` op (inferc domain):
//
//   0.5*x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
//
// emitted as:  Mul(x,0.5), Pow(x,3), Mul(·,0.044715), Add(x,·),
//              Mul(·,sqrt(2/pi)), Tanh, Add(·,1), Mul(0.5x, 1+tanh).
//
// Numerically distinct from the exact (erf) GELU, so it gets its own op/kernel.
// Returns the number of patterns folded.
int RecognizeGeluTanh(Graph* graph);

}  // namespace passes
}  // namespace inferc
