#include "ir/passes/constant_fold.h"

#include <utility>
#include <vector>

#include "kernels/movement.h"
#include "runtime/tensor.h"

namespace inferc {
namespace passes {

int FoldConstantTranspose(Graph* g) {
  // Graph outputs are returned from the executor's tape; don't fold a Transpose
  // that produces one (rare, and not worth the special-casing).
  std::vector<std::string> outs = g->outputs;

  auto is_graph_output = [&](const std::string& name) {
    for (const auto& o : outs) if (o == name) return true;
    return false;
  };

  int folded = 0;
  std::vector<Node> kept;
  kept.reserve(g->nodes.size());
  for (auto& node : g->nodes) {
    const bool foldable =
        node.op_type == "Transpose" && node.inputs.size() == 1 &&
        node.outputs.size() == 1 && !is_graph_output(node.outputs[0]);
    if (foldable) {
      const Tensor* in = g->GetTensor(node.inputs[0]);
      if (in != nullptr && in->IsInitializer()) {
        const std::vector<int64_t> perm = node.GetAttrInts("perm");  // empty = reverse
        rt::Tensor src =
            rt::Tensor::FromHostBytes(in->dtype, in->shape, in->raw_data.data());
        rt::Tensor dst = rt::Transpose(src, perm);

        // Materialize the transposed result as a new initializer. The original
        // input initializer is left untouched (it may have other consumers).
        Tensor& out = g->GetOrCreateTensor(node.outputs[0]);
        out.name = node.outputs[0];
        out.dtype = dst.dtype();
        out.shape = dst.shape();
        const uint8_t* p = static_cast<const uint8_t*>(dst.bytes());
        out.raw_data.assign(p, p + dst.byte_size());

        ++folded;
        continue;  // drop the Transpose node
      }
    }
    kept.push_back(std::move(node));
  }
  g->nodes = std::move(kept);
  return folded;
}

}  // namespace passes
}  // namespace inferc
