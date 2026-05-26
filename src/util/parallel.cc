#include "util/parallel.h"

#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace inferc {
namespace par {

namespace {
// Default on; `INFERC_PARALLEL=0` forces serial (for the parallel-vs-serial
// ablation and the single-threaded baseline).
bool InitEnabled() {
  const char* e = std::getenv("INFERC_PARALLEL");
  return !(e != nullptr && std::strcmp(e, "0") == 0);
}
bool g_enabled = InitEnabled();

int QueryPhysicalCores() {
  int n = 0;
  size_t sz = sizeof(n);
  // Use all physical cores (P + E). Apple Silicon is heterogeneous, so the E-
  // cores are slower, but with many ops parallelized GCD's dynamic dispatch
  // balances the load (faster cores grab more blocks) — measured faster and
  // lower-variance than P-cores-only once the whole pipeline is parallel.
  // Override with INFERC_THREADS.
  if (sysctlbyname("hw.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0) return n;
  return 1;
}
}  // namespace

void SetParallelEnabled(bool on) { g_enabled = on; }
bool ParallelEnabled() { return g_enabled; }

int MaxThreads() {
  static const int cores = [] {
    if (const char* e = std::getenv("INFERC_THREADS")) {
      int n = std::atoi(e);
      if (n > 0) return n;
    }
    return QueryPhysicalCores();
  }();
  return cores;
}

void ParallelFor(int64_t n, int64_t grain,
                 const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  // Decide block count: at most MaxThreads, at least 1, and no smaller than
  // `grain` elements per block (so we don't pay dispatch overhead on tiny work).
  int64_t max_blocks = std::max<int64_t>(1, MaxThreads());
  int64_t by_grain = grain > 0 ? (n + grain - 1) / grain : max_blocks;
  int64_t blocks = std::min(max_blocks, std::max<int64_t>(1, by_grain));

  if (!g_enabled || blocks <= 1) {
    fn(0, n);
    return;
  }

  const int64_t chunk = (n + blocks - 1) / blocks;
  dispatch_apply(static_cast<size_t>(blocks),
                 dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                 ^(size_t b) {
                   int64_t begin = static_cast<int64_t>(b) * chunk;
                   int64_t end = std::min(n, begin + chunk);
                   if (begin < end) fn(begin, end);
                 });
}

}  // namespace par
}  // namespace inferc
