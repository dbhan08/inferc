#pragma once

#include "ir/graph.h"

namespace inferc {
namespace passes {

// Constant folding for Transpose-of-initializer.
//
// Some ONNX exports transpose a constant weight on every forward pass. The
// flagship case is GPT-2's tied LM head: `logits = hidden @ wteᵀ`, exported as
// `Transpose(transformer.wte.weight [50257, 768]) → MatMul`. Recomputing that
// 38.6M-element transpose every decode step dominated per-token latency.
//
// This pass evaluates each `Transpose` whose single input is a graph
// initializer, materializes the result as a *new* initializer (the original is
// left intact — GPT-2's wte is also read by the embedding Gather), and removes
// the node. The executor then seeds the folded constant directly into its tape.
//
// Outputs that are graph outputs are left alone (the executor returns those
// from the tape). Returns the number of nodes folded.
int FoldConstantTranspose(Graph* graph);

}  // namespace passes
}  // namespace inferc
