// amx_shim.h -- pre-packed direct-AMX fp32 GEMM drop-in for ggml-blas prefill
// matmuls (research shim for the inferc M1-AMX paper end-to-end measurement).
//
// ggml-blas computes C[M,N] = A[M,K] . W[N,K]^T via cblas_sgemm(NoTrans A,
// Trans B) where src0 (the weight) is stored row-major [N,K]. This shim
// intercepts that call for constant-weight fp32 matmuls: each weight is
// transposed-and-packed ONCE (keyed on its data pointer) into [K, Nc] panels,
// then every call runs only the activation transpose + the multi-threaded
// Nc=64/Kc=2048 compute loop -- the kernel of the paper. Non-qualifying
// matmuls (small, non-fp32, or non-leaf src0 such as attention scores) return
// false and fall back to cblas.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <dispatch/dispatch.h>

// ---- vendored Apple AMX encodings (corsix/amx) ----
#define AMXSHIM_NOP_OP_IMM5(op, imm5) \
  __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
#define AMXSHIM_OP_GPR(op, gpr) \
  __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")
#define SHIM_LDX(g)   AMXSHIM_OP_GPR(0, g)
#define SHIM_LDY(g)   AMXSHIM_OP_GPR(1, g)
#define SHIM_LDZ(g)   AMXSHIM_OP_GPR(4, g)
#define SHIM_STZ(g)   AMXSHIM_OP_GPR(5, g)
#define SHIM_FMA32(g) AMXSHIM_OP_GPR(12, g)
#define SHIM_SET()    AMXSHIM_NOP_OP_IMM5(17, 0)
#define SHIM_CLR()    AMXSHIM_NOP_OP_IMM5(17, 1)

namespace amxshim {

static const int NC = 64, KC = 2048;

static inline uint64_t fma32op(int z, int xo, bool f) {
  return (uint64_t(z) << 20) | (uint64_t(xo) << 10) | (f ? (1ULL << 27) : 0);
}

struct Packed {
  int64_t N = 0, K = 0, covered = 0;
  std::vector<int64_t> jcs;
  std::vector<std::vector<float>> panels;  // panels[p] = [K x Ncm], row-major in k
};
static std::unordered_map<const void*, Packed> g_cache;
static std::mutex g_mtx;

// diagnostics (GGML_AMX_SHIM_DIAG=1 prints a summary at exit)
static std::atomic<long> g_calls{0}, g_packs{0};
static std::atomic<double> g_kernel_ms{0}, g_pack_ms{0};
struct DiagPrinter {
  ~DiagPrinter() {
    if (const char* e = getenv("GGML_AMX_SHIM_DIAG"); e && e[0] == '1')
      fprintf(stderr, "[amx_shim] calls=%ld packs=%ld kernel=%.1fms pack=%.1fms\n",
              g_calls.load(), g_packs.load(), g_kernel_ms.load(), g_pack_ms.load());
  }
};
static DiagPrinter g_diag;

// Build [K, Ncm] panels of B = W^T from the stored weight W[N,K] (row stride K).
// B[k][jc+c] = W[jc+c][k]  ->  panel[k*Ncm + c] = W[(jc+c)*K + k].
static void build_pack(Packed& P, const float* W, int64_t N, int64_t K) {
  P.N = N; P.K = K; P.covered = 0;
  for (int64_t jc = 0; jc < N; jc += NC) P.jcs.push_back(jc);
  P.panels.resize(P.jcs.size());
  for (size_t p = 0; p < P.jcs.size(); ++p) {
    int64_t jc = P.jcs[p];
    int64_t Ncm = (std::min<int64_t>(NC, N - jc) / 64) * 64;
    P.covered = std::max<int64_t>(P.covered, jc + Ncm);
    if (Ncm <= 0) continue;
    auto& buf = P.panels[p];
    buf.resize(size_t(K) * Ncm);
    for (int64_t k = 0; k < K; ++k)
      for (int64_t c = 0; c < Ncm; ++c) buf[k * Ncm + c] = W[(jc + c) * K + k];
  }
}

static void compute_kc(const float* At, const float* pB, float* C, int64_t M,
                       int64_t N, int64_t K, int64_t jc, int64_t Ncm) {
  const uint64_t LDX_PAIR = 1ULL << 62;
  for (int64_t pc = 0; pc < K; pc += KC) {
    int64_t Kc = std::min<int64_t>(KC, K - pc);
    const bool fp = (pc == 0);
    for (int64_t i0 = 0; i0 < M; i0 += 16)
      for (int64_t jr = 0; jr < Ncm; jr += 64) {
        if (!fp)
          for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
            SHIM_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
        for (int64_t kk = 0; kk < Kc; ++kk) {
          const bool f = (fp && kk == 0);
          SHIM_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));
          const float* brow = pB + (pc + kk) * Ncm + jr;
          SHIM_LDX(reinterpret_cast<uint64_t>(brow)      | (0ULL << 56) | LDX_PAIR);
          SHIM_LDX(reinterpret_cast<uint64_t>(brow + 32) | (2ULL << 56) | LDX_PAIR);
          SHIM_FMA32(fma32op(0, 0, f)); SHIM_FMA32(fma32op(1, 64, f));
          SHIM_FMA32(fma32op(2, 128, f)); SHIM_FMA32(fma32op(3, 192, f));
        }
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 16; ++j)
          SHIM_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) | (uint64_t(4 * j + t) << 56));
      }
  }
}

// Residual columns [j0, N) -- scalar, never hit when N is a multiple of 64.
static void tail_scalar(const float* At, const float* W, float* C, int64_t M,
                        int64_t N, int64_t K, int64_t j0) {
  for (; j0 < N; ++j0)
    for (int64_t i = 0; i < M; ++i) {
      float s = 0;
      for (int64_t k = 0; k < K; ++k) s += At[k * M + i] * W[j0 * K + k];
      C[i * N + j0] = s;
    }
}

static thread_local std::vector<float> tls_At;

// W = src0->data [N,K]; A = src1->data [M,K]; C = dst->data [M,N].
// Returns false (decline) for shapes/types we do not handle.
inline bool try_mul_mat(const void* wkey, const float* W, const float* A,
                        float* C, int64_t M, int64_t N, int64_t K) {
  if (M < 32 || N < 1024 || K < 1024) return false;  // prefill, large GEMM only
  g_calls.fetch_add(1, std::memory_order_relaxed);
  Packed* P;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_cache.find(wkey);
    if (it == g_cache.end()) {
      auto p0 = std::chrono::steady_clock::now();
      Packed np; build_pack(np, W, N, K);
      it = g_cache.emplace(wkey, std::move(np)).first;
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - p0).count();
      g_packs.fetch_add(1, std::memory_order_relaxed);
      for (double cur = g_pack_ms.load();
           !g_pack_ms.compare_exchange_weak(cur, cur + ms););
    }
    P = &it->second;
  }
  tls_At.resize(size_t(K) * M);
  float* At = tls_At.data();
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At[k * M + i] = A[i * K + k];

  const int64_t* jcp = P->jcs.data();
  const int nP = (int)P->jcs.size();
  const std::vector<std::vector<float>>* pan = &P->panels;
  const int64_t NN = N, KK = K, MM = M;
  const float* Atp = At; float* Cp = C;
  auto k0 = std::chrono::steady_clock::now();
  dispatch_apply(nP, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
    int64_t jc = jcp[w];
    int64_t Ncm = (std::min<int64_t>(NC, NN - jc) / 64) * 64;
    if (Ncm <= 0) return;
    SHIM_SET(); compute_kc(Atp, (*pan)[w].data(), Cp, MM, NN, KK, jc, Ncm); SHIM_CLR();
  });
  double kms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - k0).count();
  for (double cur = g_kernel_ms.load();
       !g_kernel_ms.compare_exchange_weak(cur, cur + kms););
  if (P->covered < N) tail_scalar(At, W, C, M, N, K, P->covered);
  return true;
}

}  // namespace amxshim
