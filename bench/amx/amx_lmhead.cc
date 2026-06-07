// LM-head profiling: why do we lose to Accelerate at N-very-large shapes, and
// can the UNPACKED (streaming) variant close it?
//
//   * Accelerate cold (first in process, uncontended) as reference.
//   * packed-MT: finer-panel packed kernel, multi-thread, + a pack-only pass to
//     attribute time to B-panel packing vs AMX compute.
//   * unpacked-MT: software-pipelined unpacked kernel parallelized over column
//     blocks (no 490 MB pack write).
// Bit-exact verified vs Accelerate.

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}

// ---- packed panel (B[:, jc:jc+Nc_main] packed, then AMX) ----
static void packed_panel(const float* At, const float* B, float* C, int64_t M,
                         int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm,
                         std::vector<float>& packB, bool pack_only) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Ncm);
  if (!pack_only) AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k)
      std::memcpy(&packB[k * Ncm], &B[(pc + k) * N + jc], Ncm * sizeof(float));
    if (pack_only) continue;
    const float* pB = packB.data();
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
            AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool f = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + kk * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, f)); AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f)); AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
  }
  if (!pack_only) AMX_CLR();
}

// ---- unpacked software-pipelined, columns [jstart, jend) (At shared) ----
static void unpacked_range(const float* At, const float* B, float* C, int64_t M,
                           int64_t N, int64_t K, int64_t jstart, int64_t jend) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  static const uint64_t fA[4] = {(0ULL<<20)|(0ULL<<10), (1ULL<<20)|(64ULL<<10),
                                 (2ULL<<20)|(128ULL<<10), (3ULL<<20)|(192ULL<<10)};
  static const uint64_t fB[4] = {(0ULL<<20)|(256ULL<<10)|64ULL, (1ULL<<20)|(320ULL<<10)|64ULL,
                                 (2ULL<<20)|(384ULL<<10)|64ULL, (3ULL<<20)|(448ULL<<10)|64ULL};
  AMX_SET();
  int64_t j0 = jstart;
  for (; j0 + 64 <= jend; j0 += 64) {
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      AMX_LDY(reinterpret_cast<uint64_t>(&At[0 * M + i0]) | (0ULL << 56));
      const float* b0 = B + 0 * N + j0;
      AMX_LDX(reinterpret_cast<uint64_t>(b0)      | (0ULL << 56) | LDX_PAIR);
      AMX_LDX(reinterpret_cast<uint64_t>(b0 + 32) | (2ULL << 56) | LDX_PAIR);
      const uint64_t fmask = (1ULL << 27);
      int64_t k = 0;
      for (; k + 2 <= K - 1; k += 2) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[(k + 1) * M + i0]) | (1ULL << 56));
        const float* b1 = B + (k + 1) * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(b1)      | (4ULL << 56) | LDX_PAIR);
        AMX_LDX(reinterpret_cast<uint64_t>(b1 + 32) | (6ULL << 56) | LDX_PAIR);
        uint64_t fm = (k == 0) ? fmask : 0;
        AMX_FMA32(fA[0]|fm); AMX_FMA32(fA[1]|fm); AMX_FMA32(fA[2]|fm); AMX_FMA32(fA[3]|fm);
        AMX_LDY(reinterpret_cast<uint64_t>(&At[(k + 2) * M + i0]) | (0ULL << 56));
        const float* b2 = B + (k + 2) * N + j0;
        AMX_LDX(reinterpret_cast<uint64_t>(b2)      | (0ULL << 56) | LDX_PAIR);
        AMX_LDX(reinterpret_cast<uint64_t>(b2 + 32) | (2ULL << 56) | LDX_PAIR);
        AMX_FMA32(fB[0]); AMX_FMA32(fB[1]); AMX_FMA32(fB[2]); AMX_FMA32(fB[3]);
      }
      while (k < K) {
        bool curB = (k & 1); const uint64_t* f = curB ? fB : fA;
        uint64_t fm = (k == 0) ? fmask : 0;
        if (k + 1 < K) {
          bool nB = !curB; int xr = nB ? 4 : 0;
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(k + 1) * M + i0]) | (uint64_t(nB) << 56));
          const float* bn = B + (k + 1) * N + j0;
          AMX_LDX(reinterpret_cast<uint64_t>(bn)      | (uint64_t(xr) << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(bn + 32) | (uint64_t(xr + 2) << 56) | LDX_PAIR);
        }
        AMX_FMA32(f[0]|fm); AMX_FMA32(f[1]|fm); AMX_FMA32(f[2]|fm); AMX_FMA32(f[3]|fm);
        ++k;
      }
      for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0 + 16 * t) | (uint64_t(4 * j + t) << 56));
    }
  }
  for (; j0 + 16 <= jend; j0 += 16)
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) | (uint64_t(4 * j) << 56));
    }
  AMX_CLR();
  for (; j0 < jend; ++j0)
    for (int64_t i = 0; i < M; ++i) {
      float s = 0; for (int64_t k = 0; k < K; ++k) s += At[k * M + i] * B[k * N + j0];
      C[i * N + j0] = s;
    }
}

// Correctness tail for residual columns [j0, N).
static void tail_cols(const float* At, const float* B, float* C, int64_t M,
                      int64_t N, int64_t K, int64_t j0) {
  AMX_SET();
  for (; j0 + 16 <= N; j0 += 16)
    for (int64_t i0 = 0; i0 < M; i0 += 16) {
      for (int64_t k = 0; k < K; ++k) {
        AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
        AMX_LDX(reinterpret_cast<uint64_t>(B + k * N + j0));
        AMX_FMA32(Fma32Op(0, 0, k == 0));
      }
      for (int j = 0; j < 16; ++j)
        AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + j0) | (uint64_t(4 * j) << 56));
    }
  AMX_CLR();
  for (; j0 < N; ++j0)
    for (int64_t i = 0; i < M; ++i) {
      float s = 0; for (int64_t k = 0; k < K; ++k) s += At[k * M + i] * B[k * N + j0];
      C[i * N + j0] = s;
    }
}

// Double-buffered pack/compute overlap: while the AMX (this thread) computes
// panel p from one buffer, a worker core packs panel p+1 into the other. Hides
// the memory-bound packing behind the AMX-bound compute. Panels carry full K
// (packB is L2-resident), so Z accumulates over the whole k with no Kc blocking.
static void packed_overlap(const float* At, const float* B, float* C, int64_t M,
                           int64_t N, int64_t K, int Nc) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  std::vector<int64_t> jcs;
  for (int64_t jc = 0; jc < N; jc += Nc) jcs.push_back(jc);
  const int nP = (int)jcs.size();
  auto ncm = [&](int p) { return (std::min<int64_t>(Nc, N - jcs[p]) / 64) * 64; };
  std::vector<float> buf0(size_t(K) * Nc), buf1(size_t(K) * Nc);
  float* bufs[2] = {buf0.data(), buf1.data()};
  dispatch_queue_t pq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

  if (nP > 0) {  // pack panel 0
    int64_t Ncm = ncm(0), jc = jcs[0];
    for (int64_t k = 0; k < K; ++k)
      std::memcpy(&bufs[0][k * Ncm], &B[k * N + jc], Ncm * sizeof(float));
  }
  AMX_SET();
  for (int p = 0; p < nP; ++p) {
    int64_t jc = jcs[p], Ncm = ncm(p);
    dispatch_semaphore_t done = nullptr;
    if (p + 1 < nP) {  // async-pack panel p+1 into the other buffer
      done = dispatch_semaphore_create(0);
      float* nb = bufs[(p + 1) % 2];
      int64_t njc = jcs[p + 1], nNcm = ncm(p + 1);
      dispatch_async(pq, ^{
        for (int64_t k = 0; k < K; ++k)
          std::memcpy(&nb[k * nNcm], &B[k * N + njc], nNcm * sizeof(float));
        dispatch_semaphore_signal(done);
      });
    }
    const float* pb = bufs[p % 2];  // compute panel p (AMX) from its buffer
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        for (int64_t k = 0; k < K; ++k) {
          const bool f = (k == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[k * M + i0]));
          const float* br = pb + k * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(br)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(br + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, f)); AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f)); AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
    if (done) dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  }
  AMX_CLR();
  int64_t covered = nP ? jcs[nP - 1] + ncm(nP - 1) : 0;
  if (covered < N) tail_cols(At, B, C, M, N, K, covered);
}

static double bestms(int iters, void (^f)()) {
  f(); f();
  double b = 1e30;
  for (int i = 0; i < iters; ++i) { auto t0 = clk::now(); f(); b = std::min(b, ms(clk::now() - t0)); }
  return b;
}

int main() {
  struct Shape { int M, N, K; const char* tag; };
  const Shape shapes[] = {
    {128, 60000, 2048, "GPT-2-style LM-head [60000,2048]"},
    {128, 32000, 2048, "TinyLlama   LM-head [32000,2048]"},
    {128, 32000, 4096, "Llama-7B    LM-head [32000,4096]"},
  };
  const int Nc = 512, Kc = 512;

  for (auto& s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f),
        Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * double(N) * K;
    const float *ap = A.data(), *bp = B.data(); float* cp = C.data(); float* crefp = Cref.data();

    // Accelerate cold (first thing in process for this shape).
    double tA = bestms(7, ^{ cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                                         1.0f, ap, K, bp, N, 0.0f, crefp, N); });
    double gA = flops / (tA / 1e3) / 1e9;

    std::vector<float> At(size_t(K) * M);
    for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
    const float* atp = At.data();

    std::vector<int64_t> all;
    for (int64_t jc = 0; jc < N; jc += Nc) all.push_back(jc);
    const int nP = (int)all.size();
    std::vector<std::vector<float>> packP(nP);
    int64_t covered = 0;
    for (int64_t jc : all) covered = std::max<int64_t>(covered, jc + (std::min<int64_t>(Nc, N - jc) / 64) * 64);

    // packed-MT (full) and pack-only (memcpy, no AMX) to attribute time.
    auto packed = [&](bool pack_only) {
      dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t jc = all[w]; int64_t Ncm = (std::min<int64_t>(Nc, N - jc) / 64) * 64;
        if (Ncm > 0) packed_panel(atp, bp, cp, M, N, K, Kc, jc, Ncm, packP[w], pack_only);
      });
      if (!pack_only && covered < N) unpacked_range(atp, bp, cp, M, N, K, covered, N);
    };
    double tPk = bestms(7, ^{ packed(false); });
    double tPackOnly = bestms(7, ^{ packed(true); });
    double gPk = flops / (tPk / 1e3) / 1e9;

    // unpacked-MT: parallelize over column blocks.
    const int nBlk = 24;
    int64_t blk = ((N + nBlk - 1) / nBlk + 63) / 64 * 64;
    auto unpacked = [&] {
      dispatch_apply(nBlk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t js = (int64_t)w * blk, je = std::min<int64_t>(js + blk, N);
        if (js < je) unpacked_range(atp, bp, cp, M, N, K, js, je);
      });
    };
    std::fill(C.begin(), C.end(), 0.f);
    double tUn = bestms(7, ^{ unpacked(); });
    double gUn = flops / (tUn / 1e3) / 1e9;
    float maxd = 0.f;
    for (size_t i = 0; i < C.size(); i += 997) maxd = std::max(maxd, std::fabs(cp[i] - crefp[i]));

    // double-buffered pack/compute overlap
    std::fill(C.begin(), C.end(), 0.f);
    double tOv = bestms(7, ^{ packed_overlap(atp, bp, cp, M, N, K, Nc); });
    double gOv = flops / (tOv / 1e3) / 1e9;
    float maxdOv = 0.f;
    for (size_t i = 0; i < C.size(); i += 997) maxdOv = std::max(maxdOv, std::fabs(cp[i] - crefp[i]));

    std::printf("\n=== %s ===\n", s.tag);
    std::printf("  Accelerate:        %5.0f GFLOPS  (%.2f ms)\n", gA, tA);
    std::printf("  packed-MT:         %5.0f GFLOPS  (%.2f ms)  pack-only %.2f ms (%.0f%% of packed)\n",
                gPk, tPk, tPackOnly, 100.0 * tPackOnly / tPk);
    std::printf("  unpacked-MT:       %5.0f GFLOPS  (%.2f ms)  %s\n",
                gUn, tUn, maxd < 1e-3f ? "bit-exact" : "BUG");
    std::printf("  overlap (2-buf):   %5.0f GFLOPS  (%.2f ms)  %s  vs Accel %.2fx %s\n",
                gOv, tOv, maxdOv < 1e-3f ? "bit-exact" : "BUG", gOv / gA,
                gOv > gA ? "<- BEATS Accelerate" : "");
  }
  return 0;
}
