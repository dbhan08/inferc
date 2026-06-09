// The single-thread loop pins at ~670 GFLOPS the moment it contains loads
// (amx_microbench.cc). But there is ONE shared AMX block per P-cluster, and AMX
// register state is per-thread -- so multiple threads issuing to the same block
// can interleave thread A's loads with thread B's FMAs, filling the issue slots
// one thread leaves idle. This sweeps thread count to find the SHARED-BLOCK
// ceiling, with loads (the real GEMM regime) vs pure FMA (the absolute cap).
//
//   - if loaded-MT climbs toward pure-FMA-MT as T grows -> loads ARE hidden
//     cross-thread; the true ceiling is the pure-FMA rate, and a smarter MT
//     schedule (not a better single-thread microkernel) is the lever.
//   - if loaded-MT plateaus below it -> the shared block is load-issue-capped
//     even across threads.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}
static const uint64_t PAIR = 1ULL << 62;

// one worker: own L1-resident A/B, own AMX register state; runs the chosen body.
static void worker(int64_t iters, bool with_loads, std::atomic<int>* ready,
                   std::atomic<bool>* go, int nthreads) {
  float* A = nullptr; float* B = nullptr;
  posix_memalign((void**)&A, 128, 64 * sizeof(float));
  posix_memalign((void**)&B, 128, 64 * sizeof(float));
  for (int i = 0; i < 64; ++i) { A[i] = float(i) * 1e-3f; B[i] = float(i) * 1e-3f; }
  const uint64_t a = reinterpret_cast<uint64_t>(A), b = reinterpret_cast<uint64_t>(B);
  AMX_SET();
  for (int i = 0; i < 2000; ++i) AMX_FMA32(Fma32Op(0, 0, false));  // warm
  ready->fetch_add(1);
  while (!go->load()) {}                                           // release barrier
  if (with_loads) {
    for (int64_t i = 0; i < iters; ++i) {
      AMX_LDX(b | (0ULL << 56) | PAIR);
      AMX_LDX((b + 128) | (2ULL << 56) | PAIR);
      AMX_LDY(a | (0ULL << 56));
      AMX_FMA32(Fma32Op(0, 0, false)); AMX_FMA32(Fma32Op(1, 64, false));
      AMX_FMA32(Fma32Op(2, 128, false)); AMX_FMA32(Fma32Op(3, 192, false));
    }
  } else {
    for (int64_t i = 0; i < iters; ++i) {
      AMX_FMA32(Fma32Op(0, 0, false)); AMX_FMA32(Fma32Op(1, 64, false));
      AMX_FMA32(Fma32Op(2, 128, false)); AMX_FMA32(Fma32Op(3, 192, false));
    }
  }
  AMX_CLR();
  free(A); free(B);
}

static double run(int T, bool with_loads, int64_t iters) {
  std::atomic<int> ready{0}; std::atomic<bool> go{false};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) ts.emplace_back(worker, iters, with_loads, &ready, &go, T);
  while (ready.load() < T) {}                  // wait until all warmed at the barrier
  auto t0 = clk::now();
  go.store(true);
  for (auto& th : ts) th.join();
  double s = sec(clk::now() - t0);
  return (double)T * iters * 4 * 512 / s / 1e9;  // aggregate GFLOPS
}

int main() {
  const int64_t IT = 20'000'000;
  std::printf("%-2s  %-22s  %-22s  %s\n", "T", "loaded-MT GFLOPS", "pure-FMA-MT GFLOPS", "loaded % of pure");
  for (int T : {1, 2, 3, 4, 6, 8}) {
    double gl = run(T, true, IT);
    double gf = run(T, false, IT);
    std::printf("%-2d  %-22.0f  %-22.0f  %.0f%%\n", T, gl, gf, 100.0 * gl / gf);
  }
  std::printf("\nT=4 = P-cluster (4 P-cores, 1 shared AMX block). T>4 spills to E-cluster.\n"
              "Single-thread loaded floor was ~670; Accelerate large-GEMM ~924, our MT ~924-1070.\n");
  return 0;
}
