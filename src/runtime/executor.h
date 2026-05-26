#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/graph.h"
#include "runtime/tensor.h"

namespace inferc {
namespace prof { class Profiler; }
namespace rt {

// Walks an IR Graph in topological order, dispatches each node to its
// kernel, and produces the graph's outputs.
//
// Construction is one-time cost: weights are materialized from the IR's
// initializer bytes into runtime Tensors.
//
// Run() is per-inference: it builds a name → Tensor "tape", seeds it with
// inputs and initializers, executes every node, and returns the output
// tensors keyed by graph.outputs name.
class Executor {
 public:
  explicit Executor(const Graph& graph);

  // Run inference. `inputs` maps graph-input names to runtime Tensors.
  // Returns a map containing every graph output (and only those) by name.
  // If `profiler` is non-null, records one IterRecord with per-op timings.
  std::map<std::string, Tensor> Run(
      const std::map<std::string, Tensor>& inputs,
      prof::Profiler* profiler = nullptr) const;

 private:
  const Graph* graph_;
  // Weights, prepared once from initializer bytes. unordered_map: the decode
  // hot path does ~3 tape lookups per node over ~3100 nodes; O(1) hashing beats
  // std::map's O(log n) long-string comparisons.
  std::unordered_map<std::string, Tensor> initializers_;
};

}  // namespace rt
}  // namespace inferc
