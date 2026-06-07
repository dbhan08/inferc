// Dual-cluster GEMM with proportional load balancing.
//
// Part 3 of the headroom probe showed the E-cluster has a ~125 GFLOPS AMX block
// but an even P/E split loses (the slow E half bottlenecks the join). This test
// gives E a *small* column slice and sweeps the fraction, running the P portion
// across the P-cluster (dispatch_apply, USER_INITIATED) concurrently with one E
// worker (UTILITY, less throttled than BACKGROUND).
//
// Question: at a K>=N shape where Accelerate underfeeds the AMX, can
// P-cluster-MT + a proportional E slice beat Accelerate's multi-thread sgemm?

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}

// Compute C[:, jc:jc+Nc_main] for one packed panel (At is shared, read-only).
static void panel(const float* At, const float* B, float* C, int64_t M,
                  int64_t N, int64_t K, int Kc, int64_t jc, int64_t Nc_main,
                  std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Nc_main);
  AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k)
      std::memcpy(&packB[k * Nc_main], &B[(pc + k) * N + jc],
                  Nc_main * sizeof(float));
    const float* pB = packB.data();
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t jr = 0; jr < Nc_main; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 16; ++j)
              AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                      (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool first = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + kk * Nc_main + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, first));   AMX_FMA32(Fma32Op(1, 64, first));
          AMX_FMA32(Fma32Op(2, 128, first)); AMX_FMA32(Fma32Op(3, 192, first));
        }
        for (int t = 0; t < 4; ++t)
          for (int j = 0; j < 16; ++j)
            AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                    (uint64_t(4 * j + t) << 56));
      }
    }
  }
  AMX_CLR();
}

int main() {
  const char* veclib = std::getenv("VECLIB_MAXIMUM_THREADS");
  std::printf("VECLIB_MAXIMUM_THREADS = %s\n",
              veclib ? veclib : "(unset, Accelerate default multi-thread)");

  // K>=N shapes where Accelerate underfeeds the AMX (Part 1).
  struct Shape { int M, N, K; const char* tag; };
  const Shape shapes[] = {
    {128, 2048, 2048, "QKV   [128,2048,2048]"},
    {128, 4096, 4096, "QKV   [128,4096,4096]"},
    {128, 4096, 11008, "FFN2 [128,4096,11008]"},
  };
  const int Nc = 512, Kc = 512;

  for (auto& s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    std::vector<float> Cref(size_t(M) * N, 0.f);
    const double flops = 2.0 * M * double(N) * K;

    // Pre-transpose A once (shared read-only across all workers).
    std::vector<float> At(size_t(K) * M);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
    const float *atp = At.data(), *bp = B.data(); float* cp = C.data();

    const float *ap = A.data(); float* crefp = Cref.data();
    auto accel = [&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, ap,
                  K, bp, N, 0.0f, crefp, N);
    };

    // COLD Accelerate: measured before any of our GCD kernel code runs this
    // shape, so Accelerate's threadpool is uncontended. Also fills Cref for the
    // correctness check. Compare against the PAIRED Accelerate below to detect
    // GCD-threadpool starvation.
    double tcold = 1e30;
    accel(); accel();
    for (int i = 0; i < 7; ++i) {
      auto t0 = clk::now(); accel(); tcold = std::min(tcold, ms(clk::now() - t0));
    }
    double gCold = flops / (tcold / 1e3) / 1e9;

    // Build the list of jc panels (multiples of Nc).
    std::vector<int64_t> all_jc;
    for (int64_t jc = 0; jc < N; jc += Nc) all_jc.push_back(jc);
    const int nPanels = (int)all_jc.size();

    std::printf("\n=== %s ===\n", s.tag);
    std::printf("  Accelerate COLD (uncontended): %.0f GFLOPS\n", gCold);
    std::printf("  %-6s %-10s %-8s\n", "E-frac", "agg GFLOPS", "panels P/E");

    // Sweep the fraction of panels given to the E-cluster (our kernel only;
    // within-bench, so the E contribution is a clean relative measurement).
    double best_g = 0; double best_frac = 0;
    for (double efrac : {0.0, 0.10, 0.15, 0.20, 0.25}) {
      int nE = (int)std::lround(efrac * nPanels);
      nE = std::min(nE, nPanels - 1);          // keep at least one P panel
      int nP = nPanels - nE;
      // P gets the first nP panels, E the last nE.
      std::vector<int64_t> pjc(all_jc.begin(), all_jc.begin() + nP);
      std::vector<int64_t> ejc(all_jc.begin() + nP, all_jc.end());
      std::vector<std::vector<float>> packP(nP);
      std::vector<float> packE;

      auto once = [&]() -> double {
        std::fill(C.begin(), C.end(), 0.f);
        dispatch_group_t g = dispatch_group_create();
        auto t0 = clk::now();
        // E worker: one UTILITY thread sweeps its panels.
        if (nE > 0) {
          dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            std::vector<float> pk;
            for (int64_t jc : ejc) {
              int64_t Nc_main = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
              if (Nc_main > 0) panel(atp, bp, cp, M, N, K, Kc, jc, Nc_main, pk);
            }
          });
        }
        // P workers: dispatch_apply across the P-cluster.
        dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
          int64_t jc = pjc[w];
          int64_t Nc_main = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
          if (Nc_main > 0) panel(atp, bp, cp, M, N, K, Kc, jc, Nc_main, packP[w]);
        });
        dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
        return ms(clk::now() - t0);
      };

      once(); once();  // warmup
      double best = 1e30;
      for (int i = 0; i < 7; ++i) best = std::min(best, once());
      double g = flops / (best / 1e3) / 1e9;

      // Correctness vs Accelerate on a prefix.
      float maxd = 0.f;
      size_t spot = std::min<size_t>(C.size(), 4096);
      for (size_t i = 0; i < spot; ++i) maxd = std::max(maxd, std::fabs(cp[i] - crefp[i]));

      std::printf("  %-6.2f %-10.0f %d/%d%s\n", efrac, g, nP, nE,
                  maxd > 1e-3f ? "  BUG" : "");
      if (g > best_g) { best_g = g; best_frac = efrac; }
    }

    // Paired head-to-head at the best E-fraction: alternate Accelerate and our
    // dual-cluster kernel back-to-back (cancels thermal/scheduler drift), with
    // warmup, mean +/- std over the per-trial ratio. This is the trustworthy
    // comparison; the absolute Accelerate GFLOPS appears for reference.
    {
      int nE = (int)std::lround(best_frac * nPanels);
      nE = std::min(nE, nPanels - 1);
      int nP = nPanels - nE;
      std::vector<int64_t> pjc(all_jc.begin(), all_jc.begin() + nP);
      std::vector<int64_t> ejc(all_jc.begin() + nP, all_jc.end());
      std::vector<std::vector<float>> packP(nP);
      auto ours = [&] {
        std::fill(C.begin(), C.end(), 0.f);
        dispatch_group_t g = dispatch_group_create();
        if (nE > 0)
          dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            std::vector<float> pk;
            for (int64_t jc : ejc) {
              int64_t Ncm = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
              if (Ncm > 0) panel(atp, bp, cp, M, N, K, Kc, jc, Ncm, pk);
            }
          });
        dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
          int64_t jc = pjc[w];
          int64_t Ncm = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
          if (Ncm > 0) panel(atp, bp, cp, M, N, K, Kc, jc, Ncm, packP[w]);
        });
        dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
      };
      auto tg = [&](void (^f)()) { auto t0 = clk::now(); f(); return ms(clk::now() - t0); };
      for (int i = 0; i < 4; ++i) { accel(); ours(); }  // warmup both
      std::vector<double> ra, ro, rr;
      for (int i = 0; i < 20; ++i) {
        double ta = tg(^{ accel(); });
        double to = tg(^{ ours(); });
        ra.push_back(flops / (ta / 1e3) / 1e9);
        ro.push_back(flops / (to / 1e3) / 1e9);
        rr.push_back(ta / to);  // speedup ours over Accelerate
      }
      auto mean = [](std::vector<double>& v) { double m = 0; for (double x : v) m += x; return m / v.size(); };
      auto sd = [&](std::vector<double>& v, double m) { double s = 0; for (double x : v) s += (x - m) * (x - m); return std::sqrt(s / (v.size() - 1)); };
      double ma = mean(ra), mo = mean(ro), mr = mean(rr);
      std::printf("  PAIRED @ E-frac %.2f (20 trials): Accel(paired) %.0f | ours %.0f | speedup %.2f +/- %.2fx\n",
                  best_frac, ma, mo, mr, sd(rr, mr));
      std::printf("  HONEST vs COLD Accelerate (%.0f): ours/cold = %.2fx %s\n",
                  gCold, mo / gCold,
                  mo > gCold ? "<- beats cold Accelerate" : "<- does NOT beat cold Accelerate");
      if (ma < 0.85 * gCold)
        std::printf("  [!] paired Accel (%.0f) << cold Accel (%.0f): GCD threads are starving "
                    "Accelerate; the paired speedup is inflated.\n", ma, gCold);
    }
  }
  return 0;
}
