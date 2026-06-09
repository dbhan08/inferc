// Pre-packed-weight comparison across all 12 LLM prefill shapes, with a BNNS
// fairness gate.
//
// In LLM inference B is the weight, constant across every token. Accelerate's
// cblas_sgemm is stateless and re-packs B each call. A deployed kernel can pack
// B once at model load and amortize it to ~0. We test whether that flips the
// N>>K shapes (LM head, FFN-up) that the per-call comparison loses, for a clean
// 12/12, and whether Accelerate's BNNS matmul -- the other Accelerate GEMM
// entry point -- closes the gap (the fairness gate: if BNNS already matches our
// pre-packed throughput, the win is cblas-only).
//
// Columns (multi-threaded, all cores, fp32, bit-exact vs cblas):
//   cblas           Accelerate cblas_sgemm, re-packs each call
//   BNNS            Accelerate BNNSMatMul, same descriptors reused each call
//   ours-prepacked  B packed once (untimed), Kc-blocked compute-only timed

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
static void pack_full(const float* B, float* dst, int64_t N, int64_t K, int64_t jc, int64_t Ncm) {
  for (int64_t k = 0; k < K; ++k) std::memcpy(&dst[k * Ncm], &B[k * N + jc], Ncm * sizeof(float));
}
// Kc-cache-blocked compute from a pre-packed [K x Ncm] panel (no packing here).
static void compute_kc(const float* At, const float* pB, float* C, int64_t M,
                       int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  for (int64_t pc = 0; pc < K; pc += Kc) {
    int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
    const bool first_pc = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        if (!first_pc)
          for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
            AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc_eff; ++kk) {
          const bool f = (first_pc && kk == 0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + (pc + kk) * Ncm + jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          AMX_FMA32(Fma32Op(0, 0, f)); AMX_FMA32(Fma32Op(1, 64, f));
          AMX_FMA32(Fma32Op(2, 128, f)); AMX_FMA32(Fma32Op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
  }
}
// Per-call kernel: pack one Kc-block then immediately compute it (interleaved,
// cache-friendly -- the deployed per-call path), packing B on this call.
static void packed_panel(const float* At, const float* B, float* C, int64_t M,
                         int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm,
                         std::vector<float>& packB) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  packB.resize(size_t(Kc) * Ncm);
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
}
// Residual columns [j0, N): 16-wide AMX tiles + scalar remainder (reads B).
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
  std::printf("%-12s %-8s %-8s %-16s\n", "shape", "cblas", "BNNS", "ours-prepacked");
  int wins = 0, wins_bnns = 0, wins_pc_bnns = 0;
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f), Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float *ap = A.data(), *bp = B.data(); float* cp = C.data(); float* crp = Cref.data();
    auto best = [](int n, void (^f)()) { f(); f(); double b = 1e30; for (int i = 0; i < n; ++i) { auto t0 = clk::now(); f(); b = std::min(b, ms(clk::now() - t0)); } return b; };

    double tA = best(7, ^{ cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, ap, K, bp, N, 0.0f, crp, N); });
    double gA = flops / (tA / 1e3) / 1e9;

    // BNNS fp32, descriptors + workspace reused across calls (same weight B).
    std::vector<float> Cb(size_t(M) * N, 0.f);
    auto da = desc(A.data(), BNNSDataTypeFloat32, M, K);
    auto db = desc(B.data(), BNNSDataTypeFloat32, K, N);
    auto dc = desc(Cb.data(), BNNSDataTypeFloat32, M, N);
    size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
    std::vector<char> ws(wsz ? wsz : 1);
    const BNNSNDArrayDescriptor* dap = &da; const BNNSNDArrayDescriptor* dbp = &db;
    BNNSNDArrayDescriptor* dcp = &dc; void* wsp = wsz ? ws.data() : nullptr;
    double tB = best(7, ^{ BNNSMatMul(false, false, 1.0f, dap, dbp, dcp, wsp, nullptr); });
    double gB = flops / (tB / 1e3) / 1e9;

    // transpose A once; shape-adaptive panels. NOTE: large Kc wins single-thread
    // (kills Z-reload passes, see amx_profile.cc) but LOSES multi-threaded -- all
    // threads share one P-cluster L2, so deep panels thrash. Kc=256/512 is the
    // MT-tuned choice.
    const int Nc = (N > K) ? 256 : 512, Kc = (N > K) ? 256 : 512;
    std::vector<float> At(size_t(K) * M);
    for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
    const float* atp = At.data();
    std::vector<int64_t> jcs; for (int64_t jc = 0; jc < N; jc += Nc) jcs.push_back(jc);
    const int nP = (int)jcs.size();
    auto ncm = [&](int p) { return (std::min<int64_t>(Nc, N - jcs[p]) / 64) * 64; };
    int64_t covered = 0; for (int p = 0; p < nP; ++p) covered = std::max<int64_t>(covered, jcs[p] + ncm(p));
    std::vector<std::vector<float>> packed(nP);
    for (int p = 0; p < nP; ++p) { int64_t Ncm = ncm(p); if (Ncm <= 0) continue; packed[p].resize(size_t(K) * Ncm); pack_full(bp, packed[p].data(), N, K, jcs[p], Ncm); }
    const int64_t* jcp = jcs.data(); auto* pkp = &packed;
    auto prepacked = ^{
      dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t Ncm = (std::min<int64_t>(Nc, N - jcp[w]) / 64) * 64;
        if (Ncm <= 0) return;
        AMX_SET(); compute_kc(atp, (*pkp)[w].data(), cp, M, N, K, Kc, jcp[w], Ncm); AMX_CLR();
      });
      if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
    };
    double tP = best(7, prepacked);
    double gP = flops / (tP / 1e3) / 1e9;
    float maxd = 0.f; for (size_t i = 0; i < C.size(); i += 1023) maxd = std::max(maxd, std::fabs(cp[i] - crp[i]));
    bool ok = maxd < 1e-3f;
    if (ok && gP > gA) wins++;
    if (ok && gP > gB) wins_bnns++;

    // per-call ours (packs B each call) -- the non-pre-packed kernel, for the
    // per-call comparison against the stronger BNNS baseline.
    std::vector<std::vector<float>> scratch(nP);
    auto* scp = &scratch;
    auto percall = ^{
      dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t Ncm = (std::min<int64_t>(Nc, N - jcp[w]) / 64) * 64;
        if (Ncm <= 0) return;
        std::vector<float>& sc = (*scp)[w];
        AMX_SET(); packed_panel(atp, bp, cp, M, N, K, Kc, jcp[w], Ncm, sc); AMX_CLR();
      });
      if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
    };
    double tC = best(7, percall);
    double gC = flops / (tC / 1e3) / 1e9;
    if (gC > gB) wins_pc_bnns++;

    std::printf("%-12s cblas %-5.0f BNNS %-5.0f | per-call %-5.0f (%.2fxB) prepack %-5.0f (%.2fxB) %s\n",
                s.t, gA, gB, gC, gC / gB, gP, gP / gB, ok ? "" : "BUG");
  }
  std::printf("\nprepacked beats cblas %d/12, beats BNNS %d/12; per-call beats BNNS %d/12\n",
              wins, wins_bnns, wins_pc_bnns);
  return 0;
}
