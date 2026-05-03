#pragma once

#include <string>
#include <vector>

#include "ir/graph.h"

namespace inferc {

// Numpy-style right-aligned broadcast. Returns true on success and writes
// the resulting shape into `out`. If either operand has -1 in a dim, the
// result is -1 in that dim (conservative). Returns false if a concrete
// mismatch is detected (e.g., 3 vs 5 with neither being 1).
bool BroadcastShapes(const Shape& a, const Shape& b, Shape* out);

// Forward shape inference: walks `graph->nodes` in order, dispatches per
// op_type, and fills in shapes/dtypes on output tensors.
//
// Returns true if every node's output tensors got a non-empty shape (which
// may include symbolic -1 dims). On failure, `error_message` names the
// offending node and op_type.
//
// `unsupported_ops`, if non-null, is populated with the set of op_types
// the inferencer didn't recognize (output shapes left unset for those).
bool InferShapes(Graph* graph,
                 std::string* error_message,
                 std::vector<std::string>* unsupported_ops = nullptr);

}  // namespace inferc
