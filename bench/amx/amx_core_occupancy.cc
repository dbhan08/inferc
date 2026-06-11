// Direct measurement of how much of the M1 each GEMM backend lights up at an
// LLM-prefill shape -- the evidence for the paper's otherwise-inferential claim
// that Accelerate under-parallelizes these shapes and leaves the second on-chip
// AMX block (the E-cluster's) idle.
//
// The M1 has two clusters (4 P-cores + 4 E-cores), one AMX block each. An AMX
// block can only be driven by an active core in its cluster, and an AMX-issuing
// thread keeps its core busy, so per-core CPU active residency is a sound proxy
// for AMX-block usage (AMX has no public PMU on M1). We sample per-core ticks
// (host_processor_info / PROCESSOR_CPU_LOAD_INFO) across a sustained loop of
// each backend at the QKV and FFN1 shapes and report, baseline-subtracted, the
// active-core-equivalents in each cluster. Cluster membership is identified by
// the standard QOS trick: QOS_CLASS_BACKGROUND threads are confined to the
// E-cores by the scheduler.
//
// Build: clang++ -O3 -std=c++17 -I third_party -framework Accelerate \
//          bench/amx/amx_core_occupancy.cc -o /tmp/amx_core_occupancy
//
// Reads no inputs; prints a per-cluster occupancy table per backend per shape.

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
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

// ---- per-core CPU active-residency sampling -------------------------------
struct Snap { std::vector<uint64_t> busy, total; };
static Snap snap_cpus() {
  natural_t ncpu = 0; processor_info_array_t info = nullptr; mach_msg_type_number_t cnt = 0;
  host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &ncpu, &info, &cnt);
  processor_cpu_load_info_t cpu = (processor_cpu_load_info_t)info;
  Snap s; s.busy.resize(ncpu); s.total.resize(ncpu);
  for (natural_t i = 0; i < ncpu; ++i) {
    uint64_t u = cpu[i].cpu_ticks[CPU_STATE_USER], sy = cpu[i].cpu_ticks[CPU_STATE_SYSTEM],
             ni = cpu[i].cpu_ticks[CPU_STATE_NICE], id = cpu[i].cpu_ticks[CPU_STATE_IDLE];
    s.busy[i] = u + sy + ni; s.total[i] = u + sy + ni + id;
  }
  vm_deallocate(mach_task_self(), (vm_address_t)info, cnt * sizeof(int));
  return s;
}
// per-core active fraction in [0,1] between two snapshots
static std::vector<double> active_frac(const Snap& a, const Snap& b) {
  std::vector<double> f(a.busy.size(), 0.0);
  for (size_t i = 0; i < f.size(); ++i) {
    double db = double(b.busy[i] - a.busy[i]), dt = double(b.total[i] - a.total[i]);
    f[i] = dt > 0 ? db / dt : 0.0;
  }
  return f;
}

// Identify the 4 E-core CPU indices: spin 4 background-QOS threads (the
// scheduler confines QOS_CLASS_BACKGROUND to the E-cluster) and take the 4
// hottest cores.
static std::vector<int> find_ecores(int ncpu) {
  std::atomic<bool> stop{false};
  std::vector<std::thread> ts;
  Snap a = snap_cpus();
  for (int t = 0; t < 4; ++t) ts.emplace_back([&] {
    pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
    volatile double x = 1.0; while (!stop.load()) { for (int i = 0; i < 100000; ++i) x = x * 1.0000001 + 1.0; }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  Snap b = snap_cpus(); stop = true; for (auto& th : ts) th.join();
  auto f = active_frac(a, b);
  std::vector<int> idx(ncpu); for (int i = 0; i < ncpu; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int p, int q) { return f[p] > f[q]; });
  std::vector<int> e(idx.begin(), idx.begin() + std::min(4, ncpu));
  std::sort(e.begin(), e.end());
  return e;
}

// Run `work` in a tight loop for `seconds`, sampling per-core occupancy across it.
template <class F>
static std::vector<double> occupancy(F&& work, double seconds) {
  Snap a = snap_cpus();
  auto t0 = clk::now();
  while (ms(clk::now() - t0) < seconds * 1e3) work();
  Snap b = snap_cpus();
  return active_frac(a, b);
}

int main() {
  const int ncpu = 8;
  std::vector<int> ecore = find_ecores(ncpu);
  bool isE[8] = {false};
  for (int e : ecore) if (e < 8) isE[e] = true;
  std::printf("E-cluster CPU indices: ");
  for (int e : ecore) std::printf("%d ", e);
  std::printf("(remaining are P-cores)\n\n");

  // baseline (idle) occupancy over 0.6 s, to subtract background OS activity
  auto base = occupancy([] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }, 0.6);

  struct Sh { int M, N, K; const char* t; };
  const Sh shapes[] = {
    {128, 2048, 2048, "QKV  (N=K=2048, K>=N)"},
    {128, 8192, 2048, "FFN1 (N=8192,  N>K) "},
  };

  const int Nc = 64, Kc = 1024;
  for (auto& s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f), Cb(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    const float *ap = A.data(), *bp = B.data(); float* cp = C.data();

    // cblas backend
    auto cblas_work = [&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, ap, K, bp, N, 0.0f, cp, N);
    };
    // BNNSMatMul backend (descriptors reused)
    auto da = desc(A.data(), BNNSDataTypeFloat32, M, K);
    auto db = desc(B.data(), BNNSDataTypeFloat32, K, N);
    auto dc = desc(Cb.data(), BNNSDataTypeFloat32, M, N);
    size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
    std::vector<char> ws(wsz ? wsz : 1);
    const BNNSNDArrayDescriptor* dap = &da; const BNNSNDArrayDescriptor* dbp = &db;
    BNNSNDArrayDescriptor* dcp = &dc; void* wsp = wsz ? ws.data() : nullptr;
    auto bnns_work = [&] { BNNSMatMul(false, false, 1.0f, dap, dbp, dcp, wsp, nullptr); };

    // our pre-packed kernel: transpose A, pack B into Nc panels (once), dispatch
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
    auto ours_work = [&] {
      dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
        int64_t Ncm = (std::min<int64_t>(Nc, N - jcp[w]) / 64) * 64;
        if (Ncm <= 0) return;
        AMX_SET(); compute_kc(atp, (*pkp)[w].data(), cp, M, N, K, Kc, jcp[w], Ncm); AMX_CLR();
      });
      if (covered < N) tail_cols(atp, bp, cp, M, N, K, covered);
    };

    struct Backend { const char* name; std::function<void()> work; };
    std::vector<Backend> backs = {
      {"cblas_sgemm   ", cblas_work}, {"BNNSMatMul    ", bnns_work}, {"ours-prepacked", ours_work},
    };

    std::printf("=== %s ===\n", s.t);
    std::printf("%-16s %-12s %-12s %-8s\n",
                "backend", "P-cluster", "E-cluster", "total");
    for (auto& bk : backs) {
      // warm up, then take the median P/E active-core-equivalents over 3 windows
      for (int i = 0; i < 3; ++i) bk.work();
      std::vector<double> ps, es;
      for (int rep = 0; rep < 3; ++rep) {
        auto f = occupancy(bk.work, 1.5);
        double psum = 0, esum = 0;
        for (int i = 0; i < ncpu; ++i) {
          double a = std::max(0.0, f[i] - base[i]);   // baseline-subtracted
          if (isE[i]) esum += a; else psum += a;
        }
        ps.push_back(psum); es.push_back(esum);
      }
      std::sort(ps.begin(), ps.end()); std::sort(es.begin(), es.end());
      double p = ps[1], e = es[1];   // median of 3
      // active-core-equivalents per cluster (max 4.0 each)
      std::printf("%-16s %-12.2f %-12.2f %-8.2f\n", bk.name, p, e, p + e);
    }
    std::printf("\n");
  }
  std::printf("Reading: P-busy/E-busy are baseline-subtracted active-core-equivalents\n"
              "(max 4.0 per cluster). E-busy ~0 means the E-cluster's AMX block is idle.\n");
  return 0;
}
