#include "ir/passes/recognize_layernorm.h"

#include <set>
#include <string>
#include <vector>

#include "ir/passes/pass_utils.h"

namespace inferc {
namespace passes {

namespace {

const Node* UniqueConsumer(const Graph& g, const std::string& tensor) {
  const Node* found = nullptr;
  int count = 0;
  for (const auto& n : g.nodes) {
    for (const auto& in : n.inputs) {
      if (in == tensor) { found = &n; ++count; break; }
    }
    if (count > 1) return nullptr;
  }
  return count == 1 ? found : nullptr;
}

bool IsGraphOutput(const Graph& g, const std::string& tensor) {
  for (const auto& o : g.outputs) if (o == tensor) return true;
  return false;
}

// Identify the non-`chain` operand of a 2-input commutative op and require it
// to be a 1D initializer (the gamma/beta weight). Returns its tensor name.
bool MatchScaleBias(const Graph& g, const Node& n, const std::string& chain,
                    std::string* weight) {
  if (n.inputs.size() != 2) return false;
  for (int i = 0; i < 2; ++i) {
    if (n.inputs[i] != chain) continue;
    const std::string& w = n.inputs[1 - i];
    const Tensor* t = g.GetTensor(w);
    if (t == nullptr || !t->IsInitializer() || t->shape.size() != 1) return false;
    *weight = w;
    return true;
  }
  return false;
}

}  // namespace

int RecognizeLayerNorm(Graph* graph) {
  struct Match {
    std::set<int> remove_indices;
    std::string final_output;  // Y (last Add's output)
    std::string root_input;    // X
    std::string gamma, beta;   // 1D scale / bias initializers
    float eps = 1e-5f;
    int insert_at = -1;
  };
  std::vector<Match> matches;

  // Anchor on Sqrt — rare enough to make the scan cheap and unambiguous.
  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    const Node& sqrt = graph->nodes[i];
    if (sqrt.op_type != "Sqrt" || sqrt.inputs.size() != 1 || sqrt.outputs.empty())
      continue;

    // veps = Add(var, eps_const)
    const Node* add_eps = FindNodeByOutput(*graph, sqrt.inputs[0]);
    if (!add_eps || add_eps->op_type != "Add" || add_eps->inputs.size() != 2) continue;
    std::string var_name;
    float eps = 0.0f;
    bool got_eps = false;
    for (int k = 0; k < 2; ++k) {
      const Node* p = FindNodeByOutput(*graph, add_eps->inputs[k]);
      float v;
      if (p && TryGetConstantScalarFloat(*p, &v)) {
        eps = v; got_eps = true; var_name = add_eps->inputs[1 - k];
      }
    }
    if (!got_eps || eps <= 0.0f || eps > 1e-1f) continue;

    // var = ReduceMean(sq)
    const Node* rm2 = FindNodeByOutput(*graph, var_name);
    if (!rm2 || rm2->op_type != "ReduceMean" || rm2->inputs.empty()) continue;

    // sq = Pow(d, 2) or Mul(d, d)
    const Node* sq = FindNodeByOutput(*graph, rm2->inputs[0]);
    if (!sq || sq->inputs.empty()) continue;
    std::string d_name;
    if (sq->op_type == "Pow" && sq->inputs.size() == 2) {
      const Node* two = FindNodeByOutput(*graph, sq->inputs[1]);
      float v;
      if (!two || !TryGetConstantScalarFloat(*two, &v) || !ApproxEq(v, 2.0f)) continue;
      d_name = sq->inputs[0];
    } else if (sq->op_type == "Mul" && sq->inputs.size() == 2 &&
               sq->inputs[0] == sq->inputs[1]) {
      d_name = sq->inputs[0];
    } else {
      continue;
    }

    // d = Sub(X, mean); mean = ReduceMean(X)
    const Node* sub = FindNodeByOutput(*graph, d_name);
    if (!sub || sub->op_type != "Sub" || sub->inputs.size() != 2) continue;
    const std::string& X = sub->inputs[0];
    const Node* rm1 = FindNodeByOutput(*graph, sub->inputs[1]);
    if (!rm1 || rm1->op_type != "ReduceMean" || rm1->inputs.empty()) continue;
    if (rm1->inputs[0] != X) continue;

    // std feeds exactly one Div(d, std)
    const Node* div = UniqueConsumer(*graph, sqrt.outputs[0]);
    if (!div || div->op_type != "Div" || div->inputs.size() != 2) continue;
    if (div->inputs[0] != d_name || div->inputs[1] != sqrt.outputs[0]) continue;

    // norm feeds Mul(norm, gamma) → Add(·, beta)
    const Node* mul_g = UniqueConsumer(*graph, div->outputs[0]);
    if (!mul_g || mul_g->op_type != "Mul" || mul_g->outputs.empty()) continue;
    std::string gamma;
    if (!MatchScaleBias(*graph, *mul_g, div->outputs[0], &gamma)) continue;

    const Node* add_b = UniqueConsumer(*graph, mul_g->outputs[0]);
    if (!add_b || add_b->op_type != "Add" || add_b->outputs.empty()) continue;
    std::string beta;
    if (!MatchScaleBias(*graph, *add_b, mul_g->outputs[0], &beta)) continue;
    if (IsGraphOutput(*graph, add_b->outputs[0])) {
      // Folding still produces add_b->outputs[0], so a graph output is fine.
    }

    Match m;
    m.root_input = X;
    m.final_output = add_b->outputs[0];
    m.gamma = gamma;
    m.beta = beta;
    m.eps = eps;
    m.insert_at = i;
    auto rm = [&](const std::string& out) {
      int idx = FindNodeIndexByOutput(*graph, out);
      if (idx >= 0) m.remove_indices.insert(idx);
    };
    rm(rm1->outputs[0]); rm(sub->outputs[0]); rm(sq->outputs[0]);
    rm(rm2->outputs[0]); rm(add_eps->outputs[0]); rm(sqrt.outputs[0]);
    rm(div->outputs[0]); rm(mul_g->outputs[0]); rm(add_b->outputs[0]);
    // Drop the eps (and Pow's "2") constants if single-use.
    for (int k = 0; k < 2; ++k) {
      const Node* p = FindNodeByOutput(*graph, add_eps->inputs[k]);
      float v;
      if (p && TryGetConstantScalarFloat(*p, &v) && ApproxEq(v, eps)) {
        if (UniqueConsumer(*graph, p->outputs[0]) == add_eps) rm(p->outputs[0]);
      }
    }
    if (sq->op_type == "Pow") {
      const Node* two = FindNodeByOutput(*graph, sq->inputs[1]);
      if (two && UniqueConsumer(*graph, two->outputs[0]) == sq) rm(two->outputs[0]);
    }
    matches.push_back(std::move(m));
  }

  if (matches.empty()) return 0;

  std::set<int> all_removed;
  std::map<int, Node> inserts;
  for (const auto& m : matches) {
    for (int idx : m.remove_indices) all_removed.insert(idx);
    Node n;
    n.op_type = "FusedLayerNorm";
    n.domain = "inferc";
    n.name = "LayerNorm_fused_" + std::to_string(m.insert_at);
    n.inputs = {m.root_input, m.gamma, m.beta};
    n.outputs = {m.final_output};
    onnx::AttributeProto eps_attr;
    eps_attr.set_name("epsilon");
    eps_attr.set_type(onnx::AttributeProto::FLOAT);
    eps_attr.set_f(m.eps);
    n.attributes.push_back(std::move(eps_attr));
    inserts[m.insert_at] = std::move(n);
  }

  std::vector<Node> new_nodes;
  new_nodes.reserve(graph->nodes.size());
  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    auto ins = inserts.find(i);
    if (ins != inserts.end()) new_nodes.push_back(std::move(ins->second));
    if (all_removed.count(i)) continue;
    new_nodes.push_back(std::move(graph->nodes[i]));
  }
  graph->nodes = std::move(new_nodes);
  return static_cast<int>(matches.size());
}

}  // namespace passes
}  // namespace inferc
