// Combine both AMX blocks. M1 has TWO matrix units: one shared by the P-cluster,
// a weaker one shared by the E-cluster. macOS places QOS_CLASS_USER_INITIATED
// threads on P-cores and QOS_CLASS_BACKGROUND threads on E-cores -> they drive
// DIFFERENT physical AMX blocks, so work split across them is additive (no shared
// issue port). Accelerate appears to use only the P-block (~924 ceiling); if a
// clean panel partition P-block + E-block sums above it, that's a real win on the
// pre-packed kernel. We sweep the E-cluster's panel share and compare to P-only
// and BNNS. Pre-packed weight (amortized), bit-exact vs Accelerate.

#include <Accelerate/Accelerate.h>
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
static BNNSNDArrayDescriptor desc(void* d2, BNNSDataType dt, int64_t rows, int64_t cols) {
  BNNSNDArrayDescriptor d; std::memset(&d, 0, sizeof(d));
  d.layout = BNNSDataLayoutRowMajorMatrix; d.size[0] = (size_t)cols; d.size[1] = (size_t)rows;
  d.data = d2; d.data_type = dt; return d;
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
    {128, 4096, 4096, "Llama QKV"}, {128, 11008, 4096, "Llama FN1"},
    {128, 60000, 2048, "GPT2 LMh "}, {128, 32000, 4096, "Llama LMh"},
  };
  const int Tp = 4, Te = 4;                 // P-cluster + E-cluster threads
  const int eShares[] = {0, 1, 2, 3, 4};    // E-cluster panel share out of 10
  std::printf("%-9s %5s | E-share 0    .1    .2    .3    .4 | best  BNNS | beat?\n", "shape", "");
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f), Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const double flops = 2.0 * M * (double)N * K;
    const float* bp = B.data(); float* cp = C.data();
    const int Nc = (N > K) ? 256 : 512, Kc = (N > K) ? 256 : 512;

    auto da = desc(A.data(), BNNSDataTypeFloat32, M, K);
    auto db = desc(B.data(), BNNSDataTypeFloat32, K, N);
    auto dc = desc(Cref.data(), BNNSDataTypeFloat32, M, N);
    size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
    std::vector<char> ws(wsz ? wsz : 1);
    const BNNSNDArrayDescriptor *dap=&da,*dbp=&db; BNNSNDArrayDescriptor* dcp=&dc; void* wsp = wsz?ws.data():nullptr;
    auto bnns=[&]{ BNNSMatMul(false,false,1.0f,dap,dbp,dcp,wsp,nullptr); }; bnns();bnns();
    double tB=1e30; for(int i=0;i<9;++i){auto t0=clk::now();bnns();tB=std::min(tB,ms(clk::now()-t0));}
    double gB = flops/(tB/1e3)/1e9;

    std::vector<float> At(size_t(K) * M);
    for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];
    const float* atp = At.data();
    std::vector<int64_t> jcs; for (int64_t jc=0;jc<N;jc+=Nc) jcs.push_back(jc);
    const int nP=(int)jcs.size();
    auto ncm=[&](int p){ return (std::min<int64_t>(Nc,N-jcs[p])/64)*64; };
    int64_t covered=0; for(int p=0;p<nP;++p) covered=std::max<int64_t>(covered,jcs[p]+ncm(p));
    std::vector<std::vector<float>> packed(nP);
    for(int p=0;p<nP;++p){int64_t Ncm=ncm(p); if(Ncm>0){packed[p].resize(size_t(K)*Ncm); pack_full(bp,packed[p].data(),N,K,jcs[p],Ncm);}}

    auto run=[&](int eShare){
      // partition panels: round-robin, every panel with (p%10)<eShare -> E-cluster.
      std::vector<int> Pp, Pe;
      for(int p=0;p<nP;++p) ((p%10)<eShare ? Pe : Pp).push_back(p);
      std::atomic<int> ready{0}; std::atomic<bool> go{false};
      auto work=[&](bool eCluster,int tid,int nth,const std::vector<int>& list){
        pthread_set_qos_class_self_np(eCluster?QOS_CLASS_BACKGROUND:QOS_CLASS_USER_INITIATED,0);
        AMX_SET(); ready.fetch_add(1); while(!go.load()){}
        for(size_t i=tid;i<list.size();i+=nth){int p=list[i]; int64_t Ncm=ncm(p); if(Ncm>0) compute_kc(atp,packed[p].data(),cp,M,N,K,Kc,jcs[p],Ncm);}
        AMX_CLR();
      };
      int nThreads = Tp + (eShare>0?Te:0);
      double best=1e30;
      for(int rep=0;rep<7;++rep){
        std::vector<std::thread> th; ready=0; go=false;
        for(int t=0;t<Tp;++t) th.emplace_back(work,false,t,Tp,std::cref(Pp));
        if(eShare>0) for(int t=0;t<Te;++t) th.emplace_back(work,true,t,Te,std::cref(Pe));
        while(ready.load()<nThreads){}
        auto t0=clk::now(); go.store(true);
        for(auto& x:th) x.join();
        if(covered<N) tail_cols(atp,bp,cp,M,N,K,covered);
        best=std::min(best,ms(clk::now()-t0));
      }
      return flops/(best/1e3)/1e9;
    };

    double g[5]; double bestv=0;
    for(int i=0;i<5;++i){ g[i]=run(eShares[i]); bestv=std::max(bestv,g[i]); }
    float md=0.f; for(size_t i=0;i<C.size();i+=1023) md=std::max(md,std::fabs(cp[i]-Cref[i]));
    std::printf("%-9s %5s | %5.0f %5.0f %5.0f %5.0f %5.0f | %4.0f %4.0f | %s%s\n",
                s.t, K>N?"K>N":(N>K?"N>K":"K=N"), g[0],g[1],g[2],g[3],g[4], bestv, gB,
                bestv>gB?"Y":"-", md<1e-2f?"":" BUG");
  }
  std::printf("\nE-share = fraction of panels routed to the E-cluster's AMX block (0 = P-only).\n"
              "P-only is column .0; if a nonzero E-share beats both .0 and BNNS, the 2nd block is a real lever.\n");
  return 0;
}
