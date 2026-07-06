// Multi-thread FMA16 ceiling -- resolves the "single-thread only" caveat of
// amx_fma16_microbench.cc. The fp32 analog (amx_mt_ceiling.cc) showed the shared
// P-cluster AMX block lifts the loaded aggregate from a ~670 single-thread floor
// to ~1,480 at 8 threads (both blocks). fp16's single-thread loaded number is
// already ~1,480 (f32-acc); this asks what the multi-thread fp16 aggregate is, so
// the Paper-2 kernel target is grounded in a real ceiling, not a 1-thread one.
//
// FMA16 in fp16-in/fp32-accumulate (bit62=1) -- the kernel's mode. Each thread:
// own L1-resident fp16 A/B, own AMX state; loaded body = 1 LDX + 1 LDY + 4 FMA16.

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
static inline uint64_t Fma16(int zrow, int xoff, int yoff, bool skipZ, bool f32acc) {
  return (uint64_t(zrow) << 20) | (uint64_t(xoff) << 10) | uint64_t(yoff) |
         (skipZ ? (1ULL << 27) : 0) | (f32acc ? (1ULL << 62) : 0);
}

static void worker(int64_t iters, bool with_loads, std::atomic<int>* ready,
                   std::atomic<bool>* go) {
  __fp16* A = nullptr; __fp16* B = nullptr;
  posix_memalign((void**)&A, 128, 64 * sizeof(__fp16));
  posix_memalign((void**)&B, 128, 64 * sizeof(__fp16));
  for (int i = 0; i < 64; ++i) { A[i] = (__fp16)(i * 1e-3f); B[i] = (__fp16)(i * 1e-3f); }
  const uint64_t a = reinterpret_cast<uint64_t>(A), b = reinterpret_cast<uint64_t>(B);
  AMX_SET();
  for (int i = 0; i < 2000; ++i) AMX_FMA16(Fma16(0, 0, 0, true, true));   // warm
  ready->fetch_add(1);
  while (!go->load()) {}
  if (with_loads) {
    for (int64_t i = 0; i < iters; ++i) {
      AMX_LDX(b | (0ULL << 56)); AMX_LDY(a | (0ULL << 56));
      AMX_FMA16(Fma16(0,  0, 0, true, true)); AMX_FMA16(Fma16(8,  0, 0, true, true));
      AMX_FMA16(Fma16(16, 0, 0, true, true)); AMX_FMA16(Fma16(24, 0, 0, true, true));
    }
  } else {
    for (int64_t i = 0; i < iters; ++i) {
      AMX_FMA16(Fma16(0,  0, 0, true, true)); AMX_FMA16(Fma16(8,  0, 0, true, true));
      AMX_FMA16(Fma16(16, 0, 0, true, true)); AMX_FMA16(Fma16(24, 0, 0, true, true));
    }
  }
  AMX_CLR();
  free(A); free(B);
}

static double run(int T, bool with_loads, int64_t iters) {
  std::atomic<int> ready{0}; std::atomic<bool> go{false};
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t) ts.emplace_back(worker, iters, with_loads, &ready, &go);
  while (ready.load() < T) {}
  auto t0 = clk::now();
  go.store(true);
  for (auto& th : ts) th.join();
  double s = sec(clk::now() - t0);
  return (double)T * iters * 4 * 1024 * 2 / s / 1e9;  // aggregate GFLOPS (1024 MAC/FMA16)
}

int main() {
  const int64_t IT = 15'000'000;
  std::printf("%-2s  %-22s  %-22s  %s\n", "T", "loaded-MT GFLOPS", "pure-FMA16-MT GFLOPS", "loaded % of pure");
  for (int T : {1, 2, 3, 4, 6, 8}) {
    double gl = run(T, true, IT);
    double gf = run(T, false, IT);
    std::printf("%-2d  %-22.0f  %-22.0f  %.0f%%\n", T, gl, gf, 100.0 * gl / gf);
  }
  std::printf("\nT=4 = P-cluster shared AMX block; T>4 spills to E-cluster (fp16-in/fp32-acc).\n"
              "Compare: fp32 loaded aggregate was ~670->~1,480 (1->8 thr). fp16 single-thread\n"
              "loaded was already ~1,480; this is the real multi-thread Paper-2 target.\n");
  return 0;
}
