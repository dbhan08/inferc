#pragma once

#include <map>
#include <string>

#include "ir/graph.h"

namespace inferc {
namespace passes {

// Number of times each tensor name is consumed. A tensor is "used" by:
//   - every appearance in a downstream node's inputs (+1 each)
//   - being a graph output (+1)
//
// This is the key safety check for fusion: a pattern is only safe to fuse if
// the intermediate tensors are single-use. If something else reads them, we
// can't elide the producing node without changing semantics.
using UseCounts = std::map<std::string, int>;
UseCounts ComputeUseCounts(const Graph& graph);

// Try to extract a Constant node's value as a single float (scalar or
// single-element tensor). Returns false if the node is not a Constant op or
// its value isn't a single float.
bool TryGetConstantScalarFloat(const Node& node, float* out);

// Lookup helpers — find a Node (or its index in graph.nodes) by its first
// output's tensor name. Returns nullptr / -1 on miss.
const Node* FindNodeByOutput(const Graph& graph, const std::string& output_name);
int FindNodeIndexByOutput(const Graph& graph, const std::string& output_name);

// Approximate-equal check for ONNX-exported scalar constants (e.g., sqrt(2))
// where the export may have rounded.
inline bool ApproxEq(float a, float b, float rel = 1e-5f, float abs = 1e-6f) {
  float diff = a > b ? (a - b) : (b - a);
  float mag  = a > b ? a : b;
  if (mag < 0) mag = -mag;
  return diff <= abs || diff <= rel * mag;
}

}  // namespace passes
}  // namespace inferc
