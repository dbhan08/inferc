#include "ir/passes/recognize_gelu.h"

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "ir/passes/pass_utils.h"

namespace inferc {
namespace passes {

namespace {

constexpr float kSqrt2 = 1.41421356f;

// Find the unique node consuming `tensor`. Returns nullptr if zero or many.
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

// Was `tensor` declared as a graph output?
bool IsGraphOutput(const Graph& g, const std::string& tensor) {
  for (const auto& o : g.outputs) if (o == tensor) return true;
  return false;
}

// Walk Add or Mul (both commutative): given the producing node, identify
// which input is the "constant" (matching cval ± ε) and which is the "other".
// Returns false if the constant doesn't match.
bool MatchCommutativeWithConst(const Graph& g, const Node& n, float cval,
                               std::string* other_input,
                               const Node** const_node_out) {
  if (n.inputs.size() != 2) return false;
  for (int i = 0; i < 2; ++i) {
    const std::string& cand = n.inputs[i];
    const Node* prod = FindNodeByOutput(g, cand);
    if (!prod) continue;
    float v;
    if (!TryGetConstantScalarFloat(*prod, &v)) continue;
    if (!ApproxEq(v, cval, 1e-4f, 1e-5f)) continue;
    *other_input = n.inputs[1 - i];
    *const_node_out = prod;
    return true;
  }
  return false;
}

}  // namespace

int RecognizeGelu(Graph* graph) {
  int folds = 0;
  // Iterate by index since we may rebuild the node list. We collect matches
  // first, then apply them in one pass.
  struct Match {
    // Indices of nodes to remove from graph.nodes.
    std::set<int> remove_indices;
    // Output tensor name to keep (= last Mul's output).
    std::string final_output;
    // Input tensor to Gelu (= X).
    std::string root_input;
    // Index of the Div node — we'll insert the new Gelu at this slot.
    int insert_at = -1;
  };
  std::vector<Match> matches;

  // Index nodes by their first output for quick "find producer" lookups
  // (kept in sync only at the start of this scan — fine because we don't
  // mutate inside the loop).
  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    const Node& div = graph->nodes[i];
    if (div.op_type != "Div" || div.inputs.size() != 2 || div.outputs.empty())
      continue;
    if (IsGraphOutput(*graph, div.outputs[0])) continue;

    const std::string& X = div.inputs[0];
    const Node* sqrt2_const = FindNodeByOutput(*graph, div.inputs[1]);
    if (!sqrt2_const) continue;
    float c;
    if (!TryGetConstantScalarFloat(*sqrt2_const, &c)) continue;
    if (!ApproxEq(c, kSqrt2, 1e-4f, 1e-5f)) continue;

    // Div output must feed exactly one Erf.
    const Node* erf = UniqueConsumer(*graph, div.outputs[0]);
    if (!erf || erf->op_type != "Erf" || erf->outputs.empty()) continue;

    // Erf output must feed exactly one Add(·, 1.0).
    const Node* add1 = UniqueConsumer(*graph, erf->outputs[0]);
    if (!add1 || add1->op_type != "Add" || add1->outputs.empty()) continue;
    std::string add1_other;
    const Node* add1_const = nullptr;
    if (!MatchCommutativeWithConst(*graph, *add1, 1.0f, &add1_other, &add1_const))
      continue;
    if (add1_other != erf->outputs[0]) continue;

    // Add output must feed exactly one Mul whose other input is X.
    const Node* mul_x = UniqueConsumer(*graph, add1->outputs[0]);
    if (!mul_x || mul_x->op_type != "Mul" || mul_x->outputs.empty()) continue;
    if (mul_x->inputs.size() != 2) continue;
    if (!((mul_x->inputs[0] == add1->outputs[0] && mul_x->inputs[1] == X) ||
          (mul_x->inputs[1] == add1->outputs[0] && mul_x->inputs[0] == X)))
      continue;

    // Mul_x output must feed exactly one Mul(·, 0.5).
    const Node* mul_half = UniqueConsumer(*graph, mul_x->outputs[0]);
    if (!mul_half || mul_half->op_type != "Mul" || mul_half->outputs.empty())
      continue;
    std::string mul_half_other;
    const Node* mul_half_const = nullptr;
    if (!MatchCommutativeWithConst(*graph, *mul_half, 0.5f,
                                   &mul_half_other, &mul_half_const))
      continue;
    if (mul_half_other != mul_x->outputs[0]) continue;

    // All checks passed. Build the match.
    Match m;
    m.root_input = X;
    m.final_output = mul_half->outputs[0];
    m.insert_at = i;
    // Collect indices to remove (chain ops + their orphaned constants).
    auto add_index_for_output = [&](const std::string& tensor) {
      int idx = FindNodeIndexByOutput(*graph, tensor);
      if (idx >= 0) m.remove_indices.insert(idx);
    };
    add_index_for_output(div.outputs[0]);
    add_index_for_output(erf->outputs[0]);
    add_index_for_output(add1->outputs[0]);
    add_index_for_output(mul_x->outputs[0]);
    add_index_for_output(mul_half->outputs[0]);
    // The 3 Constant nodes are single-use (verified above implicitly: we
    // matched the consumer-of-each-tensor uniquely).
    add_index_for_output(sqrt2_const->outputs[0]);
    add_index_for_output(add1_const->outputs[0]);
    add_index_for_output(mul_half_const->outputs[0]);

    matches.push_back(std::move(m));
    ++folds;
  }

  if (matches.empty()) return 0;

  // Build the new node list. For each removed index we drop the original
  // node; at the "insert_at" slot of each match we insert one new Gelu node.
  std::set<int> all_removed;
  std::map<int, Node> inserts;
  for (const auto& m : matches) {
    for (int idx : m.remove_indices) all_removed.insert(idx);
    Node g;
    g.op_type = "Gelu";
    g.name = "Gelu_fused_" + std::to_string(m.insert_at);
    g.inputs = {m.root_input};
    g.outputs = {m.final_output};
    inserts[m.insert_at] = std::move(g);
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
  return folds;
}

int RecognizeGeluTanh(Graph* graph) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  constexpr float kCoef = 0.044715f;

  struct Match {
    std::set<int> remove_indices;
    std::string final_output;  // Mul_3 output
    std::string root_input;    // X
    int insert_at = -1;
  };
  std::vector<Match> matches;

  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    const Node& tanh = graph->nodes[i];
    if (tanh.op_type != "Tanh" || tanh.inputs.size() != 1 || tanh.outputs.empty())
      continue;

    // mul2 = Mul(add, sqrt(2/pi))
    const Node* mul2 = FindNodeByOutput(*graph, tanh.inputs[0]);
    if (!mul2 || mul2->op_type != "Mul") continue;
    std::string add_out;
    const Node* c3 = nullptr;
    if (!MatchCommutativeWithConst(*graph, *mul2, kSqrt2OverPi, &add_out, &c3)) continue;

    // add = Add(X, mul1) where mul1 = Mul(Pow(X,3), 0.044715)
    const Node* add = FindNodeByOutput(*graph, add_out);
    if (!add || add->op_type != "Add" || add->inputs.size() != 2) continue;
    std::string X;
    const Node* mul1 = nullptr;
    const Node* pow = nullptr;
    for (int k = 0; k < 2 && !mul1; ++k) {
      const Node* cand = FindNodeByOutput(*graph, add->inputs[k]);
      if (!cand || cand->op_type != "Mul") continue;
      std::string pow_out;
      const Node* c2 = nullptr;
      if (!MatchCommutativeWithConst(*graph, *cand, kCoef, &pow_out, &c2)) continue;
      const Node* pw = FindNodeByOutput(*graph, pow_out);
      if (!pw || pw->op_type != "Pow" || pw->inputs.size() != 2) continue;
      const Node* three = FindNodeByOutput(*graph, pw->inputs[1]);
      float tv;
      if (!three || !TryGetConstantScalarFloat(*three, &tv) || !ApproxEq(tv, 3.0f))
        continue;
      const std::string cand_X = add->inputs[1 - k];
      if (pw->inputs[0] != cand_X) continue;  // Pow base must be X
      mul1 = cand; pow = pw; X = cand_X;
    }
    if (!mul1) continue;

    // add1 = Add(tanh, 1.0)
    const Node* add1 = UniqueConsumer(*graph, tanh.outputs[0]);
    if (!add1 || add1->op_type != "Add") continue;
    std::string a1_other;
    const Node* c1 = nullptr;
    if (!MatchCommutativeWithConst(*graph, *add1, 1.0f, &a1_other, &c1)) continue;
    if (a1_other != tanh.outputs[0]) continue;

    // mul3 = Mul(0.5x, add1); 0.5x = Mul(X, 0.5)
    const Node* mul3 = UniqueConsumer(*graph, add1->outputs[0]);
    if (!mul3 || mul3->op_type != "Mul" || mul3->inputs.size() != 2) continue;
    std::string halfx_out;
    for (int k = 0; k < 2; ++k)
      if (mul3->inputs[k] == add1->outputs[0]) halfx_out = mul3->inputs[1 - k];
    if (halfx_out.empty()) continue;
    const Node* halfx = FindNodeByOutput(*graph, halfx_out);
    if (!halfx || halfx->op_type != "Mul") continue;
    std::string hx_other;
    const Node* c0 = nullptr;
    if (!MatchCommutativeWithConst(*graph, *halfx, 0.5f, &hx_other, &c0)) continue;
    if (hx_other != X) continue;

    Match m;
    m.root_input = X;
    m.final_output = mul3->outputs[0];
    m.insert_at = i;
    auto rm = [&](const std::string& out) {
      int idx = FindNodeIndexByOutput(*graph, out);
      if (idx >= 0) m.remove_indices.insert(idx);
    };
    rm(mul2->outputs[0]); rm(add->outputs[0]); rm(mul1->outputs[0]);
    rm(pow->outputs[0]); rm(tanh.outputs[0]); rm(add1->outputs[0]);
    rm(halfx->outputs[0]); rm(mul3->outputs[0]);
    // Single-use constants (0.5, 1.0, sqrt(2/pi)); the 3 and 0.044715 below.
    rm(c0->outputs[0]); rm(c1->outputs[0]); rm(c3->outputs[0]);
    // Pow's "3" const and Mul_1's 0.044715 const:
    { const Node* th = FindNodeByOutput(*graph, pow->inputs[1]);
      if (th && UniqueConsumer(*graph, th->outputs[0]) == pow) rm(th->outputs[0]); }
    for (int k = 0; k < 2; ++k) {
      const Node* cc = FindNodeByOutput(*graph, mul1->inputs[k]);
      float v;
      if (cc && TryGetConstantScalarFloat(*cc, &v) && ApproxEq(v, kCoef) &&
          UniqueConsumer(*graph, cc->outputs[0]) == mul1)
        rm(cc->outputs[0]);
    }
    matches.push_back(std::move(m));
  }

  if (matches.empty()) return 0;

  std::set<int> all_removed;
  std::map<int, Node> inserts;
  for (const auto& m : matches) {
    for (int idx : m.remove_indices) all_removed.insert(idx);
    Node n;
    n.op_type = "GeluTanh";
    n.domain = "inferc";
    n.name = "GeluTanh_fused_" + std::to_string(m.insert_at);
    n.inputs = {m.root_input};
    n.outputs = {m.final_output};
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
