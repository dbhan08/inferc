#include "ir/passes/fuse_matmul_add_gelu.h"

#include <set>
#include <string>
#include <vector>

#include "ir/passes/pass_utils.h"

namespace inferc {
namespace passes {

namespace {

// Was `tensor` declared as a graph output?
bool IsGraphOutput(const Graph& g, const std::string& tensor) {
  for (const auto& o : g.outputs) if (o == tensor) return true;
  return false;
}

}  // namespace

int FuseMatMulAddGelu(Graph* graph) {
  const UseCounts uses = ComputeUseCounts(*graph);

  // Map producer-output-name → node index, for fast "find the producer".
  std::map<std::string, int> producer_of;
  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    for (const auto& o : graph->nodes[i].outputs) producer_of[o] = i;
  }

  struct Match {
    int matmul_idx;
    int add_idx;
    int gelu_idx;
    std::string x_input;
    std::string w_input;
    std::string bias_input;
    std::string final_output;
  };
  std::vector<Match> matches;

  // Scan for MatMul nodes whose output is single-use and consumed by an Add.
  for (int mi = 0; mi < static_cast<int>(graph->nodes.size()); ++mi) {
    const Node& mm = graph->nodes[mi];
    if (mm.op_type != "MatMul" || mm.inputs.size() != 2 || mm.outputs.empty())
      continue;
    const std::string& mm_out = mm.outputs[0];
    if (IsGraphOutput(*graph, mm_out)) continue;
    auto uc = uses.find(mm_out);
    if (uc == uses.end() || uc->second != 1) continue;

    // Find the Add that consumes mm_out.
    int add_idx = -1;
    for (int j = 0; j < static_cast<int>(graph->nodes.size()); ++j) {
      const Node& n = graph->nodes[j];
      if (n.op_type != "Add" || n.inputs.size() != 2) continue;
      if (n.inputs[0] == mm_out || n.inputs[1] == mm_out) { add_idx = j; break; }
    }
    if (add_idx < 0) continue;
    const Node& add = graph->nodes[add_idx];
    if (add.outputs.empty()) continue;
    const std::string& add_out = add.outputs[0];
    if (IsGraphOutput(*graph, add_out)) continue;
    auto uc2 = uses.find(add_out);
    if (uc2 == uses.end() || uc2->second != 1) continue;

    // Identify bias input (the one that's not mm_out).
    const std::string bias =
        (add.inputs[0] == mm_out) ? add.inputs[1] : add.inputs[0];

    // Find the Gelu that consumes add_out.
    int gelu_idx = -1;
    for (int j = 0; j < static_cast<int>(graph->nodes.size()); ++j) {
      const Node& n = graph->nodes[j];
      if (n.op_type != "Gelu" || n.inputs.size() != 1) continue;
      if (n.inputs[0] == add_out) { gelu_idx = j; break; }
    }
    if (gelu_idx < 0) continue;
    const Node& gelu = graph->nodes[gelu_idx];
    if (gelu.outputs.empty()) continue;

    Match m;
    m.matmul_idx = mi;
    m.add_idx = add_idx;
    m.gelu_idx = gelu_idx;
    m.x_input = mm.inputs[0];
    m.w_input = mm.inputs[1];
    m.bias_input = bias;
    m.final_output = gelu.outputs[0];
    matches.push_back(std::move(m));
  }

  if (matches.empty()) return 0;

  // Apply: drop the three matched nodes, insert one fused node at the
  // MatMul's original index.
  std::set<int> removed;
  std::map<int, Node> inserts;
  for (const auto& m : matches) {
    removed.insert(m.matmul_idx);
    removed.insert(m.add_idx);
    removed.insert(m.gelu_idx);
    Node f;
    f.op_type = "FusedMatMulAddGELU";
    f.domain = "inferc";
    f.name = "FusedMatMulAddGELU_" + std::to_string(m.matmul_idx);
    f.inputs = {m.x_input, m.w_input, m.bias_input};
    f.outputs = {m.final_output};
    inserts[m.matmul_idx] = std::move(f);
  }

  std::vector<Node> new_nodes;
  new_nodes.reserve(graph->nodes.size());
  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    auto ins = inserts.find(i);
    if (ins != inserts.end()) new_nodes.push_back(std::move(ins->second));
    if (removed.count(i)) continue;
    new_nodes.push_back(std::move(graph->nodes[i]));
  }
  graph->nodes = std::move(new_nodes);
  return static_cast<int>(matches.size());
}

}  // namespace passes
}  // namespace inferc
