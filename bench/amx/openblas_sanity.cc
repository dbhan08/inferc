// SOTA sanity check: run OpenBLAS sgemm (NEON-only, single thread) at our LLM prefill
// shapes to confirm Accelerate (which uses AMX internally) is the M1 fp32 SOTA.
// If OpenBLAS lands at NEON-peak levels (~60-80 GFLOPS) at all shapes while
// Accelerate hits hundreds, the AMX advantage is real and the 1.18x BLIS beat at
// QKV is genuinely beating the fp32 SOTA on this hardware.
#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>
#include <vector>

// OpenBLAS-specific helpers (declared in openblas_config.h pulled in by cblas.h).

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

int main() {
  openblas_set_num_threads(1);
  std::printf("OpenBLAS config: %s\n", openblas_get_config());
  std::printf("OpenBLAS core:   %s\n", openblas_get_corename());
  std::printf("OpenBLAS threads:%d\n", openblas_get_num_threads());
  std::printf("---\n");

  struct Shape { int M, N, K; const char* name; };
  const Shape shapes[] = {
    { 128, 2048,  2048, "QKV"     },
    { 128, 8192,  2048, "FFN1"    },
    { 128, 2048,  8192, "FFN2"    },
    { 128, 60000, 2048, "LM-head" },
  };

  for (auto& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K);
    std::vector<float> B(size_t(s.K) * s.N);
    std::vector<float> C(size_t(s.M) * s.N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7)  * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * s.M * double(s.N) * s.K;

    auto bench = [&](auto fn) {
      fn(); fn();
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        auto t0 = clk::now();
        fn();
        best = std::min(best, ms(clk::now() - t0));
      }
      return std::pair<double, double>(best, flops / (best / 1e3) / 1e9);
    };

    auto [t, gf] = bench([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  s.M, s.N, s.K, 1.0f, A.data(), s.K, B.data(), s.N, 0.0f,
                  C.data(), s.N);
    });
    std::printf("  %-9s [M=%d N=%d K=%d, %.1f GFLOPs work]   OpenBLAS  %7.2f ms  %5.0f GFLOPS\n",
                s.name, s.M, s.N, s.K, flops / 1e9, t, gf);
  }
  return 0;
}
