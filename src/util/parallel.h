#pragma once

#include <cstdint>
#include <functional>

// Intra-op parallelism for the executor's hot kernels. inferc's executor walks
// nodes serially; this splits a single kernel's outer loop (rows / output tiles)
// across cores via Grand Central Dispatch. Each block writes a disjoint output
// region, so there are no data races.
//
// This is an engineering optimization (parallelism), not a novel algorithm —
// see the paper positioning: the research contribution is the AMX
// characterization, not the parallelization.
namespace inferc {
namespace par {

// Process [0, n) in `n_blocks`-ish chunks, calling fn(begin, end) per chunk,
// possibly concurrently. Runs serially when parallelism is disabled, n is
// below `grain` (not worth the dispatch overhead), or only one block results.
void ParallelFor(int64_t n, int64_t grain,
                 const std::function<void(int64_t begin, int64_t end)>& fn);

// Global toggle (default on). Off == serial — used for the single-threaded
// baseline and the multi-thread ablation.
void SetParallelEnabled(bool on);
bool ParallelEnabled();

// Max worker blocks to split into (defaults to the machine's physical core
// count, queried once). Bounds dispatch granularity.
int MaxThreads();

}  // namespace par
}  // namespace inferc
