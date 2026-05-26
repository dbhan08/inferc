#include "ir/passes/recognize_attention.h"

#include <cmath>
#include <set>
#include <string>

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

// Producer of `tensor` if it has op_type `op` (and >=1 output); else nullptr.
const Node* ProducerOfType(const Graph& g, const std::string& tensor,
                           const char* op) {
  const Node* p = FindNodeByOutput(g, tensor);
  if (p && p->op_type == op && !p->outputs.empty()) return p;
  return nullptr;
}

// Reshape-then-Transpose feeding `t`; returns the Reshape's data input (the
// [B,S,H] projection output) or "" on mismatch.
std::string ProjBehind(const Graph& g, const std::string& t,
                       std::set<int>* remove) {
  const Node* tr = ProducerOfType(g, t, "Transpose");
  if (!tr || tr->inputs.empty()) return "";
  const Node* rs = ProducerOfType(g, tr->inputs[0], "Reshape");
  if (!rs || rs->inputs.empty()) return "";
  remove->insert(FindNodeIndexByOutput(g, tr->outputs[0]));
  remove->insert(FindNodeIndexByOutput(g, rs->outputs[0]));
  return rs->inputs[0];
}

}  // namespace

int RecognizeAttention(Graph* graph) {
  struct Match {
    std::set<int> remove;
    std::string final_output;          // ctx reshape output
    std::string q, k, v, mask_cond;    // op inputs
    int64_t head_dim = 0;
    float fill = 0.0f;
    int insert_at = -1;
  };
  std::vector<Match> matches;

  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    const Node& sm = graph->nodes[i];
    if (sm.op_type != "Softmax" || sm.inputs.size() != 1 || sm.outputs.empty())
      continue;

    Match m;
    auto add = [&](const Node* n) {
      if (n) m.remove.insert(FindNodeIndexByOutput(*graph, n->outputs[0]));
    };

    // softmax <- Where(cond, fill, scores)
    const Node* where = ProducerOfType(*graph, sm.inputs[0], "Where");
    if (!where || where->inputs.size() != 3) continue;
    const Node* fillc = FindNodeByOutput(*graph, where->inputs[1]);
    if (!fillc || !TryGetConstantScalarFloat(*fillc, &m.fill)) continue;

    // Peel Cast/Expand off the mask cond. The graph Expands the mask to the
    // scores shape via Shape(scores) — but we delete scores, and the kernel
    // broadcasts the mask itself, so consume the pre-broadcast mask [B,1,1,S]
    // and remove the Cast/Expand (+ the orphaned Shape-of-scores).
    std::string cond = where->inputs[0];
    if (const Node* cst = ProducerOfType(*graph, cond, "Cast")) {
      add(cst); cond = cst->inputs[0];
    }
    if (const Node* exp = ProducerOfType(*graph, cond, "Expand")) {
      add(exp);
      if (exp->inputs.size() >= 2) {
        const Node* shp = ProducerOfType(*graph, exp->inputs[1], "Shape");
        if (shp) add(shp);
      }
      cond = exp->inputs[0];
    }
    m.mask_cond = cond;

    // scores <- MatMul(Div(Qt, √d), Kt)
    const Node* smm = ProducerOfType(*graph, where->inputs[2], "MatMul");
    if (!smm || smm->inputs.size() != 2) continue;
    const Node* div = ProducerOfType(*graph, smm->inputs[0], "Div");
    if (!div || div->inputs.size() != 2) continue;
    float sc = 0.0f;
    const Node* scc = FindNodeByOutput(*graph, div->inputs[1]);
    if (!scc || !TryGetConstantScalarFloat(*scc, &sc) || sc <= 0.0f) continue;
    m.head_dim = static_cast<int64_t>(std::llround(static_cast<double>(sc) * sc));
    if (m.head_dim <= 0) continue;

    m.q = ProjBehind(*graph, div->inputs[0], &m.remove);  // Div's input = Qt
    m.k = ProjBehind(*graph, smm->inputs[1], &m.remove);
    if (m.q.empty() || m.k.empty()) continue;

    // softmax -> MatMul(probs, Vt) -> Transpose -> Reshape (ctx)
    const Node* cmm = UniqueConsumer(*graph, sm.outputs[0]);
    if (!cmm || cmm->op_type != "MatMul" || cmm->inputs.size() != 2) continue;
    if (cmm->inputs[0] != sm.outputs[0]) continue;
    m.v = ProjBehind(*graph, cmm->inputs[1], &m.remove);
    if (m.v.empty()) continue;
    const Node* ctr = UniqueConsumer(*graph, cmm->outputs[0]);
    if (!ctr || ctr->op_type != "Transpose" || ctr->outputs.empty()) continue;
    const Node* crs = UniqueConsumer(*graph, ctr->outputs[0]);
    if (!crs || crs->op_type != "Reshape" || crs->outputs.empty()) continue;

    m.final_output = crs->outputs[0];
    m.insert_at = i;
    add(div); add(smm); add(where); add(&sm); add(cmm); add(ctr); add(crs);
    // single-use scale / fill constants
    if (UniqueConsumer(*graph, scc->outputs[0]) == div) add(scc);
    if (UniqueConsumer(*graph, fillc->outputs[0]) == where) add(fillc);
    m.remove.erase(-1);
    matches.push_back(std::move(m));
  }

  if (matches.empty()) return 0;

  std::set<int> all_removed;
  std::map<int, Node> inserts;
  for (const auto& m : matches) {
    for (int idx : m.remove) all_removed.insert(idx);
    Node n;
    n.op_type = "FusedAttention";
    n.domain = "inferc";
    n.name = "FusedAttention_" + std::to_string(m.insert_at);
    n.inputs = {m.q, m.k, m.v, m.mask_cond};
    n.outputs = {m.final_output};
    onnx::AttributeProto hd;
    hd.set_name("head_dim");
    hd.set_type(onnx::AttributeProto::INT);
    hd.set_i(m.head_dim);
    n.attributes.push_back(std::move(hd));
    onnx::AttributeProto fa;
    fa.set_name("fill");
    fa.set_type(onnx::AttributeProto::FLOAT);
    fa.set_f(m.fill);
    n.attributes.push_back(std::move(fa));
    inserts[m.insert_at] = std::move(n);
  }

  std::vector<Node> new_nodes;
  new_nodes.reserve(graph->nodes.size());
  for (int i = 0; i < static_cast<int>(graph->nodes.size()); ++i) {
    auto it = inserts.find(i);
    if (it != inserts.end()) new_nodes.push_back(std::move(it->second));
    if (all_removed.count(i)) continue;
    new_nodes.push_back(std::move(graph->nodes[i]));
  }
  graph->nodes = std::move(new_nodes);
  return static_cast<int>(matches.size());
}

}  // namespace passes
}  // namespace inferc
