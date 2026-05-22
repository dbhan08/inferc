#include "ir/passes/pass_utils.h"

#include <cstring>

namespace inferc {
namespace passes {

UseCounts ComputeUseCounts(const Graph& graph) {
  UseCounts counts;
  for (const auto& node : graph.nodes) {
    for (const auto& in : node.inputs) {
      if (!in.empty()) counts[in] += 1;
    }
  }
  for (const auto& out : graph.outputs) counts[out] += 1;
  return counts;
}

bool TryGetConstantScalarFloat(const Node& node, float* out) {
  if (node.op_type != "Constant") return false;
  const auto* val = node.GetAttr("value");
  if (val && val->type() == onnx::AttributeProto::TENSOR) {
    const auto& t = val->t();
    if (t.data_type() != onnx::TensorProto::FLOAT) return false;
    // Scalar (0-d) or single-element (e.g. [1], [1,1]) only.
    int64_t n = 1;
    for (auto d : t.dims()) n *= d;
    if (t.dims_size() != 0 && n != 1) return false;
    if (!t.raw_data().empty() && t.raw_data().size() >= sizeof(float)) {
      std::memcpy(out, t.raw_data().data(), sizeof(float));
      return true;
    }
    if (t.float_data_size() >= 1) {
      *out = t.float_data(0);
      return true;
    }
    return false;
  }
  if (const auto* v = node.GetAttr("value_float")) {
    *out = v->f();
    return true;
  }
  return false;
}

const Node* FindNodeByOutput(const Graph& graph, const std::string& output_name) {
  for (const auto& n : graph.nodes) {
    for (const auto& o : n.outputs) {
      if (o == output_name) return &n;
    }
  }
  return nullptr;
}

int FindNodeIndexByOutput(const Graph& graph, const std::string& output_name) {
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    for (const auto& o : graph.nodes[i].outputs) {
      if (o == output_name) return i;
    }
  }
  return -1;
}

}  // namespace passes
}  // namespace inferc
