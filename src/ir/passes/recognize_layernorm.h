#pragma once

#include "ir/graph.h"

namespace inferc {
namespace passes {

// Recognizes the decomposed LayerNorm that PyTorch / transformers ONNX export
// emits and folds it into a single `FusedLayerNorm` op (inferc domain).
//
// Pattern (normalize over the last axis):
//
//   mean = ReduceMean(X)
//   d    = Sub(X, mean)
//   sq   = Pow(d, 2)         (or Mul(d, d))
//   var  = ReduceMean(sq)
//   veps = Add(var, eps)
//   std  = Sqrt(veps)
//   norm = Div(d, std)
//   sc   = Mul(norm, gamma)  (gamma = 1D scale initializer)
//   Y    = Add(sc, beta)     (beta  = 1D bias initializer)
//
// Replaced with:  Y = FusedLayerNorm(X, gamma, beta)  [attr: epsilon]
//
// `eps` is read from the matched constant and preserved (DistilBERT uses 1e-12,
// GPT-2 1e-5). gamma/beta initializers are left intact and become op inputs.
// Returns the number of LayerNorm patterns folded.
int RecognizeLayerNorm(Graph* graph);

}  // namespace passes
}  // namespace inferc
