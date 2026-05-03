#pragma once

#include <string>

#include "ir/graph.h"
#include "onnx.pb.h"

namespace inferc {

// Convert an ONNX ModelProto to our IR Graph. On success returns true and
// populates `out_graph`. On failure returns false; `error_message` is set
// to a human-readable description.
//
// The conversion does NOT run shape inference; outputs of computed nodes
// will have empty/unknown shapes until InferShapes() is run.
bool ConvertOnnxToIR(const onnx::ModelProto& model,
                     Graph* out_graph,
                     std::string* error_message);

}  // namespace inferc
