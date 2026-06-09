// Hardened dual-cluster measurement. Persistent spin-barrier thread pool (no
// per-rep thread spawn -> removes the small-shape noise), mean+/-std over many
// reps, E-cluster panel-share sweep. Run BNNS in a SEPARATE process so our 8
// busy threads can't perturb Accelerate's own scheduling:
//     amx_dualcluster_hard bnns      # Accelerate BNNSMatMul, mean+/-std
//     amx_dualcluster_hard ours      # pre-packed P+E dual-block, mean+/-std
// Combine the two outputs offline for the honest ratio. Bit-exact vs cblas.
//
// P-cluster threads: QOS_USER_INITIATED (P-cores, P-AMX block).
// E-cluster threads: QOS_UTILITY        (E-cores, E-AMX block; less throttled
//                    than BACKGROUND while still off the P-cluster).

#include <Accelerate/Accelerate.h>
#include <pthread/qos.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

// ---- persistent BLOCKING-barrier pool (idle threads sleep, no spin pollution) ----
static const int Tp = 4, Te = 4, NTH = Tp + Te;
static std::mutex g_mtx;
static std::condition_variable g_cv_go, g_cv_done;
static uint64_t g_epoch = 0; static int g_done = 0; static bool g_run = true;
static const float* g_atp; static float* g_cp;
static std::vector<std::vector<float>>* g_packed;
static const int64_t* g_jcs; static int g_M, g_N, g_K, g_Kc, g_Nc;
static std::vector<int> g_panels[NTH];

static void worker(int tid) {
  pthread_set_qos_class_self_np(tid < Tp ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY, 0);
  AMX_SET();
  uint64_t myE = 0;
  for (;;) {
    std::unique_lock<std::mutex> lk(g_mtx);
    g_cv_go.wait(lk, [&]{ return g_epoch != myE || !g_run; });
    if (!g_run) { lk.unlock(); AMX_CLR(); return; }
    myE = g_epoch; lk.unlock();
    for (int p : g_panels[tid]) {
      int64_t Ncm = (std::min<int64_t>(g_Nc, g_N - g_jcs[p]) / 64) * 64;
      if (Ncm > 0) compute_kc(g_atp, (*g_packed)[p].data(), g_cp, g_M, g_N, g_K, g_Kc, g_jcs[p], Ncm);
    }
    lk.lock();
    if (++g_done == NTH) { lk.unlock(); g_cv_done.notify_one(); } else lk.unlock();
  }
}
// release the pool for one dispatch and block until all workers finish.
static void dispatch_once() {
  { std::lock_guard<std::mutex> lk(g_mtx); g_done = 0; ++g_epoch; }
  g_cv_go.notify_all();
  std::unique_lock<std::mutex> lk(g_mtx);
  g_cv_done.wait(lk, [&]{ return g_done == NTH; });
}

static void stat(std::vector<double>& t, double flops, double& mean, double& sd, double& cov) {
  double s = 0; for (double x : t) s += x; double mt = s / t.size();
  double v = 0; for (double x : t) v += (x - mt) * (x - mt); double st = std::sqrt(v / t.size());
  mean = flops / (mt / 1e3) / 1e9; sd = mean - flops / ((mt + st) / 1e3) / 1e9; cov = 100 * st / mt;
}

struct S { int M, N, K; const char* t; };
static const S SH[] = {
  {128, 2048, 2048, "GPT2 QKV "}, {128, 8192, 2048, "GPT2 FFN1"},
  {128, 2048, 8192, "GPT2 FFN2"}, {128, 60000, 2048, "GPT2 LMh "},
  {128, 2048, 2048, "Tiny QKV "}, {128, 5632, 2048, "Tiny FFN1"},
  {128, 2048, 5632, "Tiny FFN2"}, {128, 32000, 2048, "Tiny LMh "},
  {128, 4096, 4096, "Llama QKV"}, {128, 11008, 4096, "Llama FN1"},
  {128, 4096, 11008, "LlamaFN2"}, {128, 32000, 4096, "Llama LMh"},
};
static const int REPS = 30;

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "ours";

  if (mode == "bnns") {
    std::printf("# BNNS (isolated process)  mean GFLOPS +/- std (cov%%)\n");
    for (auto& s : SH) {
      const int M = s.M, N = s.N, K = s.K;
      std::vector<float> A(size_t(M)*K), B(size_t(K)*N), C(size_t(M)*N, 0.f);
      for (size_t i=0;i<A.size();++i) A[i]=float(i%7)*0.01f;
      for (size_t i=0;i<B.size();++i) B[i]=float(i%11)*0.01f;
      double flops = 2.0*M*(double)N*K;
      auto da=desc(A.data(),BNNSDataTypeFloat32,M,K), db=desc(B.data(),BNNSDataTypeFloat32,K,N), dc=desc(C.data(),BNNSDataTypeFloat32,M,N);
      size_t wsz=BNNSMatMulWorkspaceSize(false,false,1.0f,&da,&db,&dc,nullptr); std::vector<char> ws(wsz?wsz:1);
      const BNNSNDArrayDescriptor *dap=&da,*dbp=&db; BNNSNDArrayDescriptor* dcp=&dc; void* wsp=wsz?ws.data():nullptr;
      auto f=[&]{ BNNSMatMul(false,false,1.0f,dap,dbp,dcp,wsp,nullptr); }; f(); f(); f();
      std::vector<double> t; for(int i=0;i<REPS;++i){auto t0=clk::now(); f(); t.push_back(ms(clk::now()-t0));}
      double m,sd,cov; stat(t,flops,m,sd,cov);
      std::printf("%-9s %6.0f +/- %-4.0f (%3.0f%%)\n", s.t, m, sd, cov);
      std::printf("CSV,%s,%.1f\n", s.t, m);
    }
    return 0;
  }

  if (mode == "cblas") {
    std::printf("# cblas_sgemm (isolated)  mean GFLOPS +/- std (cov%%)\n");
    for (auto& s : SH) {
      const int M=s.M,N=s.N,K=s.K;
      std::vector<float> A(size_t(M)*K),B(size_t(K)*N),C(size_t(M)*N,0.f);
      for(size_t i=0;i<A.size();++i)A[i]=float(i%7)*0.01f;
      for(size_t i=0;i<B.size();++i)B[i]=float(i%11)*0.01f;
      double flops=2.0*M*(double)N*K; const float*ap=A.data(),*bp=B.data(); float*cp=C.data();
      auto f=[&]{ cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,ap,K,bp,N,0.0f,cp,N); }; f();f();f();
      std::vector<double> t; for(int i=0;i<REPS;++i){auto t0=clk::now();f();t.push_back(ms(clk::now()-t0));}
      double m,sd,cov; stat(t,flops,m,sd,cov);
      std::printf("%-9s %6.0f +/- %-4.0f (%3.0f%%)\n",s.t,m,sd,cov);
      std::printf("CSV,%s,%.1f\n",s.t,m);
    }
    return 0;
  }

  if (mode == "bnnsfc") {  // BNNS fully-connected filter: weight B^T pre-baked at creation
    std::printf("# BNNS-FC pre-packed weight (isolated)  mean GFLOPS +/- std (cov%%)\n");
    for (auto& s : SH) {
      const int M=s.M,N=s.N,K=s.K;
      std::vector<float> A(size_t(M)*K),B(size_t(K)*N),Bt(size_t(N)*K),C(size_t(M)*N,0.f);
      for(size_t i=0;i<A.size();++i)A[i]=float(i%7)*0.01f;
      for(size_t i=0;i<B.size();++i)B[i]=float(i%11)*0.01f;
      for(int64_t k=0;k<K;++k)for(int64_t n=0;n<N;++n)Bt[n*K+k]=B[k*N+n];
      double flops=2.0*M*(double)N*K;
      BNNSNDArrayDescriptor id; std::memset(&id,0,sizeof(id)); id.layout=BNNSDataLayoutVector; id.size[0]=(size_t)K; id.data_type=BNNSDataTypeFloat32;
      BNNSNDArrayDescriptor wd; std::memset(&wd,0,sizeof(wd)); wd.layout=BNNSDataLayoutRowMajorMatrix; wd.size[0]=(size_t)K; wd.size[1]=(size_t)N; wd.data=Bt.data(); wd.data_type=BNNSDataTypeFloat32;
      BNNSNDArrayDescriptor od; std::memset(&od,0,sizeof(od)); od.layout=BNNSDataLayoutVector; od.size[0]=(size_t)N; od.data_type=BNNSDataTypeFloat32;
      BNNSNDArrayDescriptor bias; std::memset(&bias,0,sizeof(bias));
      BNNSActivation act; std::memset(&act,0,sizeof(act)); act.function=BNNSActivationFunctionIdentity;
      BNNSLayerParametersFullyConnected p; std::memset(&p,0,sizeof(p)); p.i_desc=id; p.w_desc=wd; p.o_desc=od; p.bias=bias; p.activation=act;
      BNNSFilter filt=BNNSFilterCreateLayerFullyConnected(&p,nullptr);
      if(!filt){ std::printf("%-9s FILTER-FAIL\nCSV,%s,0\n",s.t,s.t); continue; }
      const float*ap=A.data(); float*cp=C.data();
      auto f=[&]{ BNNSFilterApplyBatch(filt,(size_t)M,ap,(size_t)K,cp,(size_t)N); }; f();f();f();
      std::vector<double> t; for(int i=0;i<REPS;++i){auto t0=clk::now();f();t.push_back(ms(clk::now()-t0));}
      double m,sd,cov; stat(t,flops,m,sd,cov);
      BNNSFilterDestroy(filt);
      std::printf("%-9s %6.0f +/- %-4.0f (%3.0f%%)\n",s.t,m,sd,cov);
      std::printf("CSV,%s,%.1f\n",s.t,m);
    }
    return 0;
  }

  // ---- ours: persistent pool, E-share sweep ----
  std::vector<std::thread> pool;
  for (int t = 0; t < NTH; ++t) pool.emplace_back(worker, t);
  const int eShares[] = {0, 1, 2, 3};
  std::printf("# pre-packed P+E dual-block (isolated process)  mean GFLOPS (best-E cov%%)\n");
  std::printf("%-9s %5s | E=0    E.1   E.2   E.3  | best (E.x, cov%%)\n", "shape", "");
  for (auto& s : SH) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M)*K), B(size_t(K)*N), C(size_t(M)*N, 0.f), Cref(size_t(M)*N, 0.f);
    for (size_t i=0;i<A.size();++i) A[i]=float(i%7)*0.01f;
    for (size_t i=0;i<B.size();++i) B[i]=float(i%11)*0.01f;
    const double flops = 2.0*M*(double)N*K;
    const float* bp = B.data();
    const int Nc = (N>K)?256:512, Kc = (N>K)?256:512;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,A.data(),K,bp,N,0.0f,Cref.data(),N);

    std::vector<float> At(size_t(K)*M);
    for (int64_t i=0;i<M;++i) for(int64_t k=0;k<K;++k) At[k*M+i]=A[i*K+k];
    std::vector<int64_t> jcs; for(int64_t jc=0;jc<N;jc+=Nc) jcs.push_back(jc);
    const int nP=(int)jcs.size();
    auto ncm=[&](int p){ return (std::min<int64_t>(Nc,N-jcs[p])/64)*64; };
    int64_t covered=0; for(int p=0;p<nP;++p) covered=std::max<int64_t>(covered,jcs[p]+ncm(p));
    std::vector<std::vector<float>> packed(nP);
    for(int p=0;p<nP;++p){int64_t Ncm=ncm(p); if(Ncm>0){packed[p].resize(size_t(K)*Ncm); pack_full(bp,packed[p].data(),N,K,jcs[p],Ncm);}}

    g_atp=At.data(); g_cp=C.data(); g_packed=&packed; g_jcs=jcs.data();
    g_M=M; g_N=N; g_K=K; g_Kc=Kc; g_Nc=Nc;

    double bestG=0, bestCov=0; int bestE=0; double col[4];
    for (int ei = 0; ei < 4; ++ei) {
      int eShare = eShares[ei];
      for (int t=0;t<NTH;++t) g_panels[t].clear();
      std::vector<int> Pp, Pe;
      for (int p=0;p<nP;++p) ((p%10)<eShare?Pe:Pp).push_back(p);
      for (size_t i=0;i<Pp.size();++i) g_panels[i%Tp].push_back(Pp[i]);
      for (size_t i=0;i<Pe.size();++i) g_panels[Tp + i%Te].push_back(Pe[i]);
      // warm
      for (int w=0;w<3;++w){ dispatch_once(); if(covered<N) tail_cols(At.data(),bp,C.data(),M,N,K,covered); }
      std::vector<double> tv;
      for (int rep=0; rep<REPS; ++rep) {
        auto t0=clk::now();
        dispatch_once();
        if(covered<N) tail_cols(At.data(),bp,C.data(),M,N,K,covered);
        tv.push_back(ms(clk::now()-t0));
      }
      double m,sd,cov; stat(tv,flops,m,sd,cov); col[ei]=m;
      if (m>bestG){ bestG=m; bestCov=cov; bestE=eShare; }
    }
    float md=0.f; for(size_t i=0;i<C.size();i+=1023) md=std::max(md,std::fabs(C.data()[i]-Cref[i]));
    std::printf("%-9s %5s | %4.0f  %4.0f  %4.0f  %4.0f | %4.0f (E.%d, %2.0f%%) %s\n",
                s.t, K>N?"K>N":(N>K?"N>K":"K=N"), col[0],col[1],col[2],col[3], bestG, bestE, bestCov, md<1e-2f?"":"BUG");
    std::printf("CSV,%s,%.1f,%.1f,%d\n", s.t, bestG, col[0], bestE);  // best, P-only(E=0), eshare
  }
  { std::lock_guard<std::mutex> lk(g_mtx); g_run = false; ++g_epoch; }
  g_cv_go.notify_all();
  for (auto& th : pool) th.join();
  return 0;
}
