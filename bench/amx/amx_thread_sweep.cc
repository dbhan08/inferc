// Does tuning thread count beat the default GCD dispatch on the REAL pre-packed
// kernel? The MT microbench (amx_mt_ceiling.cc) showed the shared P-cluster AMX
// block peaks at T=3 (~1200 GFLOPS, L1-hot) while dispatch_apply oversubscribes.
// Here we partition the actual pre-packed panels across EXACTLY T threads (QoS
// USER_INITIATED -> P-cluster bias), sweep T per shape, and compare to BNNS and
// to the GCD default. If a fixed T beats both on real (cache-resident) shapes,
// the ~25% headroom is real; if not, it was an L1 artifact.

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <pthread/qos.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
static inline uint64_t Fma32Op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}
static const uint64_t LDX_PAIR = 1ULL << 62;
static BNNSNDArrayDescriptor desc(void* data, BNNSDataType dt, int64_t rows, int64_t cols) {
  BNNSNDArrayDescriptor d; std::memset(&d, 0, sizeof(d));
  d.layout = BNNSDataLayoutRowMajorMatrix; d.size[0] = (size_t)cols; d.size[1] = (size_t)rows;
  d.data = data; d.data_type = dt; return d;
}
static void pack_full(const float* B, float* dst, int64_t N, int64_t K, int64_t jc, int64_t Ncm) {
  for (int64_t k = 0; k < K; ++k) std::memcpy(&dst[k * Ncm], &B[k * N + jc], Ncm * sizeof(float));
}
static void compute_kc(const float* At, const float* pB, float* C, int64_t M,
                       int64_t N, int64_t K, int Kc, int64_t jc, int64_t Ncm) {
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
    {128, 2048, 2048, "GPT2 QKV "}, {128, 8192, 2048, "GPT2 FFN1"},
    {128, 2048, 8192, "GPT2 FFN2"}, {128, 60000, 2048, "GPT2 LMh "},
    {128, 4096, 4096, "Llama QKV"}, {128, 11008, 4096, "Llama FN1"},
    {128, 4096, 11008, "LlamaFN2"}, {128, 32000, 4096, "Llama LMh"},
  };
  const int Ts[] = {1, 2, 3, 4, 6, 8};
  std::printf("%-9s %6s | T=1   T=2   T=3   T=4   T=6   T=8  | bestT  GCD   BNNS | beats?\n", "shape", "");
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f), Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float* bp = B.data(); float* cp = C.data();
    const int Nc = (N > K) ? 256 : 512, Kc = (N > K) ? 256 : 512;

    // BNNS reference
    auto da = desc(A.data(), BNNSDataTypeFloat32, M, K);
    auto db = desc(B.data(), BNNSDataTypeFloat32, K, N);
    auto dc = desc(Cref.data(), BNNSDataTypeFloat32, M, N);
    size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
    std::vector<char> ws(wsz ? wsz : 1);
    const BNNSNDArrayDescriptor *dap = &da, *dbp = &db; BNNSNDArrayDescriptor* dcp = &dc;
    void* wsp = wsz ? ws.data() : nullptr;
    auto bnns = [&]{ BNNSMatMul(false, false, 1.0f, dap, dbp, dcp, wsp, nullptr); };
    bnns(); bnns(); double tB = 1e30; for (int i = 0; i < 9; ++i){ auto t0=clk::now(); bnns(); tB=std::min(tB,ms(clk::now()-t0)); }
    double gB = flops / (tB / 1e3) / 1e9;

    // pre-pack B and transpose A once (amortized)
    std::vector<float> At(size_t(K) * M);
    for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
    const float* atp = At.data();
    std::vector<int64_t> jcs; for (int64_t jc = 0; jc < N; jc += Nc) jcs.push_back(jc);
    const int nP = (int)jcs.size();
    auto ncm = [&](int p){ return (std::min<int64_t>(Nc, N - jcs[p]) / 64) * 64; };
    int64_t covered = 0; for (int p = 0; p < nP; ++p) covered = std::max<int64_t>(covered, jcs[p] + ncm(p));
    std::vector<std::vector<float>> packed(nP);
    for (int p = 0; p < nP; ++p){ int64_t Ncm = ncm(p); if (Ncm>0){ packed[p].resize(size_t(K)*Ncm); pack_full(bp, packed[p].data(), N, K, jcs[p], Ncm); } }

    // fixed-T panel partition (round-robin for tail balance), QoS = P-cluster bias.
    auto run_T = [&](int T){
      std::atomic<int> ready{0}; std::atomic<bool> go{false};
      auto worker = [&](int tid){
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        AMX_SET();
        ready.fetch_add(1); while(!go.load()){}
        for (int p = tid; p < nP; p += T){ int64_t Ncm = ncm(p); if (Ncm>0) compute_kc(atp, packed[p].data(), cp, M, N, K, Kc, jcs[p], Ncm); }
        AMX_CLR();
      };
      double best = 1e30;
      for (int rep = 0; rep < 7; ++rep){
        std::vector<std::thread> th; ready=0; go=false;
        for (int t=0;t<T;++t) th.emplace_back(worker, t);
        while(ready.load()<T){}
        auto t0=clk::now(); go.store(true);
        for (auto& x:th) x.join();
        if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
        best = std::min(best, ms(clk::now()-t0));
      }
      return flops / (best/1e3) / 1e9;
    };

    double g[6]; double bestT = 0; int bestTi = 0;
    for (int i = 0; i < 6; ++i){ g[i] = run_T(Ts[i]); if (g[i] > bestT){ bestT = g[i]; bestTi = Ts[i]; } }

    // GCD default (dispatch_apply) for reference
    const int64_t* jcp = jcs.data(); auto* pkp = &packed;
    auto gcd = ^{
      dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w){
        int64_t Ncm = (std::min<int64_t>(Nc, N - jcp[w]) / 64) * 64;
        if (Ncm>0){ AMX_SET(); compute_kc(atp, (*pkp)[w].data(), cp, M, N, K, Kc, jcp[w], Ncm); AMX_CLR(); }
      });
      if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
    };
    gcd(); gcd(); double tG=1e30; for(int i=0;i<7;++i){auto t0=clk::now(); gcd(); tG=std::min(tG,ms(clk::now()-t0));}
    double gG = flops/(tG/1e3)/1e9;
    float md=0.f; for(size_t i=0;i<C.size(); i+=1023) md=std::max(md,std::fabs(cp[i]-Cref[i]));

    std::printf("%-9s %6s | %4.0f  %4.0f  %4.0f  %4.0f  %4.0f  %4.0f | %4.0f(T%d) %4.0f %4.0f | bT %s GCD %s%s\n",
                s.t, K>N?"K>N":(N>K?"N>K":"K=N"), g[0],g[1],g[2],g[3],g[4],g[5],
                bestT, bestTi, gG, gB,
                bestT>gB?"Y":"-", gG>gB?"Y":"-", md<1e-2f?"":" BUG");
  }
  std::printf("\nbestT = best fixed thread count; GCD = default dispatch_apply; BNNS = Accelerate.\n"
              "If bestT > BNNS where GCD <= BNNS, thread-count tuning is the (real-shape) lever.\n");
  return 0;
}
