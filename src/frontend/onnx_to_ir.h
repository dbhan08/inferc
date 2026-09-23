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
// `external_data_dir`: directory used to resolve initializers stored with
// data_location == EXTERNAL (e.g. "model.onnx_data" next to the model). Pass
// the model's directory; leave empty for models with inline weights.
bool ConvertOnnxToIR(const onnx::ModelProto& model,
                     Graph* out_graph,
                     std::string* error_message,
                     const std::string& external_data_dir = "");

// Convert a bare GraphProto (used for If/Loop subgraph bodies). Subgraph
// initializers land in out_graph->tensors; inputs/outputs are listed by name.
bool ConvertGraphProtoToIR(const onnx::GraphProto& g,
                           Graph* out_graph,
                           std::string* error_message,
                           const std::string& external_data_dir = "");

}  // namespace inferc
