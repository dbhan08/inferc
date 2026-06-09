// Per-call profiling: how do we beat BNNS WITHOUT pre-packing (a true kernel
// win, the MpGEMM bar), and why is Llama LM-head unstable?
//
// Per-call reality: A changes every call, so transpose-A is paid each call; B is
// the weight. Two ways to feed B to the AMX:
//   packed   -- copy B panels into a contiguous buffer, then stream (current).
//   unpacked -- software-pipelined kernel that reads B in place, no pack write.
// The pack is pure overhead BNNS hides; skipping it (unpacked) may win where B's
// access pattern is already AMX-friendly. We time both, MT, vs BNNS, and report
// mean+/-std over many trials so the Llama LM-head variance is visible, not hidden
// by min-of-N. Transpose-A is INSIDE the timed region (per-call honesty).

#include <Accelerate/Accelerate.h>
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
static BNNSNDArrayDescriptor desc(void* data, BNNSDataType dt, int64_t rows, int64_t cols) {
  BNNSNDArrayDescriptor d; std::memset(&d, 0, sizeof(d));
  d.layout = BNNSDataLayoutRowMajorMatrix; d.size[0] = (size_t)cols; d.size[1] = (size_t)rows;
  d.data = data; d.data_type = dt; return d;
}

// packed: copy B[:, jc:jc+Ncm] into packB (Kc-blocked), then AMX-compute it.
static void packed_panel(const float* At, const float* B, float* C, int64_t M,
                         int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm,
                         std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Ncm);
  AMX_SET();
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    for (int64_t k = 0; k < Kc_eff; ++k)
      std::memcpy(&packB[k * Ncm], &B[(pc + k) * N + jc], Ncm * sizeof(float));
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
  AMX_CLR();
}

// unpacked software-pipelined, columns [jstart,jend), reads B in place (no pack).
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

static void tail_cols(const float* At, const float* B, float* C, int64_t M, int64_t N, int64_t K, int64_t j0) {
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

// mean and sample-std of GFLOPS over `runs` timed trials of f().
static void stat(int runs, double flops, void (^f)(), double& g_mean, double& g_std, double& cov) {
  f(); f();
  std::vector<double> t;
  for (int i = 0; i < runs; ++i) { auto t0 = clk::now(); f(); t.push_back(ms(clk::now() - t0)); }
  double sum = 0; for (double x : t) sum += x; double mt = sum / t.size();
  double v = 0; for (double x : t) v += (x - mt) * (x - mt); double st = std::sqrt(v / t.size());
  g_mean = flops / (mt / 1e3) / 1e9;
  g_std = g_mean - flops / ((mt + st) / 1e3) / 1e9;  // GFLOPS spread from +1 std slower
  cov = 100.0 * st / mt;                              // coefficient of variation, %
}

int main() {
  struct S { int M, N, K; const char* t; };
  const S sh[] = {
    {128, 2048, 2048, "GPT-2  QKV "}, {128, 8192, 2048, "GPT-2  FFN1"},
    {128, 2048, 8192, "GPT-2  FFN2"}, {128, 60000, 2048, "GPT-2  LMh "},
    {128, 2048, 2048, "Tiny   QKV "}, {128, 5632, 2048, "Tiny   FFN1"},
    {128, 2048, 5632, "Tiny   FFN2"}, {128, 32000, 2048, "Tiny   LMh "},
    {128, 4096, 4096, "Llama  QKV "}, {128, 11008, 4096, "Llama  FFN1"},
    {128, 4096, 11008, "Llama FFN2"}, {128, 32000, 4096, "Llama  LMh "},
  };
  const int RUNS = 25;
  std::printf("%-12s %-16s %-16s %-16s | beats BNNS?\n",
              "shape", "BNNS (cov%)", "packed (cov%)", "unpacked (cov%)");
  int pk_w = 0, un_w = 0, any_w = 0;
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f),
        Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float *ap = A.data(), *bp = B.data(); float* cp = C.data();

    // BNNS reference + baseline (workspace + descriptors reused across calls).
    std::vector<float> Cb(size_t(M) * N, 0.f);
    auto da = desc(A.data(), BNNSDataTypeFloat32, M, K);
    auto db = desc(B.data(), BNNSDataTypeFloat32, K, N);
    auto dc = desc(Cb.data(), BNNSDataTypeFloat32, M, N);
    size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
    std::vector<char> ws(wsz ? wsz : 1);
    const BNNSNDArrayDescriptor* dap = &da; const BNNSNDArrayDescriptor* dbp = &db;
    BNNSNDArrayDescriptor* dcp = &dc; void* wsp = wsz ? ws.data() : nullptr;
    double gB, sB, cB;
    stat(RUNS, flops, ^{ BNNSMatMul(false, false, 1.0f, dap, dbp, dcp, wsp, nullptr); }, gB, sB, cB);

    const int Nc = (N > K) ? 256 : 512, Kc = (N > K) ? 256 : 512;
    std::vector<float> At(size_t(K) * M); float* atw = At.data(); const float* atp = At.data();
    std::vector<int64_t> jcs; for (int64_t jc = 0; jc < N; jc += Nc) jcs.push_back(jc);
    const int nP = (int)jcs.size();
    auto ncm = [&](int p) { return (std::min<int64_t>(Nc, N - jcs[p]) / 64) * 64; };
    int64_t covered = 0; for (int p = 0; p < nP; ++p) covered = std::max<int64_t>(covered, jcs[p] + ncm(p));
    std::vector<std::vector<float>> scratch(nP);
    const int64_t* jcp = jcs.data(); auto* scp = &scratch;

    // packed per-call: transpose A (timed) + pack+compute each panel.
    auto packed = ^{
      for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) atw[k * M + i] = ap[i * K + k];
      dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t Ncm = (std::min<int64_t>(Nc, N - jcp[w]) / 64) * 64;
        if (Ncm > 0) packed_panel(atp, bp, cp, M, N, K, Kc, jcp[w], Ncm, (*scp)[w]);
      });
      if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
    };
    double gP, sP, cP; stat(RUNS, flops, packed, gP, sP, cP);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, ap, K, bp, N, 0.0f, Cref.data(), N);
    float dPk = 0.f; for (size_t i = 0; i < C.size(); i += 1023) dPk = std::max(dPk, std::fabs(cp[i] - Cref[i]));

    // unpacked per-call: transpose A (timed) + streaming compute over col blocks.
    const int nBlk = 24;
    int64_t blk = ((N + nBlk - 1) / nBlk + 63) / 64 * 64;
    std::fill(C.begin(), C.end(), 0.f);
    auto unpacked = ^{
      for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) atw[k * M + i] = ap[i * K + k];
      dispatch_apply(nBlk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t js = (int64_t)w * blk, je = std::min<int64_t>(js + blk, N);
        if (js < je) unpacked_range(atp, bp, cp, M, N, K, js, je);
      });
    };
    double gU, sU, cU; stat(RUNS, flops, unpacked, gU, sU, cU);
    float dUn = 0.f; for (size_t i = 0; i < C.size(); i += 1023) dUn = std::max(dUn, std::fabs(cp[i] - Cref[i]));

    bool pw = gP > gB, uw = gU > gB;
    pk_w += pw; un_w += uw; any_w += (pw || uw);
    std::printf("%-12s %5.0f+-%-3.0f(%3.0f%%) %5.0f+-%-3.0f(%3.0f%%) %5.0f+-%-3.0f(%3.0f%%) | pk %s un %s %s%s\n",
                s.t, gB, sB, cB, gP, sP, cP, gU, sU, cU,
                pw ? "Y" : "-", uw ? "Y" : "-",
                (pw || uw) ? "<=WIN" : "",
                (dPk < 1e-2f && dUn < 1e-2f) ? "" : " BUG");
    std::printf("CSV,%s,%.1f,%.1f\n", s.t, gB, gP);  // BNNS, packed-per-call
  }
  std::printf("\nper-call vs BNNS: packed %d/12, unpacked %d/12, best-of-two %d/12\n", pk_w, un_w, any_w);
  std::printf("cov%% = coefficient of variation (std/mean) over %d trials; high cov%% = unstable shape.\n", RUNS);
  return 0;
}
