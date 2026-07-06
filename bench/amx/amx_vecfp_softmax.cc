// Profile whether AMX VECFP can do in-register softmax (the gate for a fused
// attention kernel enabled by the cheap-EXTRH finding). Softmax needs: subtract-max
// (VECFP mode1), exp (polynomial via mode0 fmadd + 2^k range reduction via VECINT),
// rowsum (AMX matmul-with-ones), rowmax (NO horizontal op -> obstacle). This measures
// the throughput of the VECFP building blocks (fmadd, max) and verifies they compute,
// then estimates the per-row softmax cost vs the attention GEMMs.
//
// vecfp encoding: (alu<<47)|(lane<<42)|(zrow<<20)|(xoff<<10)|yoff. lane 4 = f32 one row.
//   alu 0 = z+x*y (fmadd), alu 7 = max(x,z), alu 1 = z-x*y, alu 5 = min(x,z).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double sec(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline uint64_t Vfp(int alu, int zrow, int xoff, int yoff) {
  return (uint64_t(alu) << 47) | (4ULL << 42) | (uint64_t(zrow) << 20) | (uint64_t(xoff) << 10) | uint64_t(yoff);
}

int main() {
  float* X = nullptr; float* Y = nullptr; void* Z = nullptr;
  posix_memalign((void**)&X, 128, 64); posix_memalign((void**)&Y, 128, 64); posix_memalign(&Z, 128, 64 * 64);
  for (int i = 0; i < 16; ++i) { X[i] = float(i) - 8.f; Y[i] = 1.0f; }
  const uint64_t x = (uint64_t)X, y = (uint64_t)Y, z = (uint64_t)Z;
  const int64_t ITERS = 30'000'000; const double GHZ = 3.2;
  auto rep = [&](const char* t, double s, int64_t ops) { std::printf("%-26s %5.2f cyc/op\n", t, s * GHZ * 1e9 / ops); };

  AMX_SET();
  AMX_LDX(x | (0ULL << 56)); AMX_LDY(y | (0ULL << 56));
  for (int i = 0; i < 2000; ++i) AMX_VECFP(Vfp(0, 0, 0, 0));

  // throughput: fmadd (exp polynomial building block), 4 independent Z rows for ILP
  { auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) { AMX_VECFP(Vfp(0,0,0,0)); AMX_VECFP(Vfp(0,1,0,0)); AMX_VECFP(Vfp(0,2,0,0)); AMX_VECFP(Vfp(0,3,0,0)); }
    rep("VECFP fmadd (mode0)", sec(clk::now()-t0), ITERS*4); }
  // throughput: max (mode7) -- running-max for online softmax
  { auto t0 = clk::now();
    for (int64_t i = 0; i < ITERS; ++i) { AMX_VECFP(Vfp(7,0,0,0)); AMX_VECFP(Vfp(7,1,0,0)); AMX_VECFP(Vfp(7,2,0,0)); AMX_VECFP(Vfp(7,3,0,0)); }
    rep("VECFP max (mode7)", sec(clk::now()-t0), ITERS*4); }

  // CORRECTNESS: max(x,z) with z=0 -> relu; fmadd z+x*y; verify a 1-step poll.
  std::memset(Z, 0, 64 * 64);
  AMX_LDZ(z | (0ULL << 56)); AMX_LDZ((z + 64) | (1ULL << 56));   // zero the Z REGISTER rows 0,1
  AMX_VECFP(Vfp(7, 0, 0, 0));                      // Z[0] = max(X, Z=0) = relu(X)
  AMX_VECFP(Vfp(0, 1, 0, 0));                      // Z[1] = 0 + X*Y(=1) = X
  AMX_STZ(z | (0ULL << 56)); AMX_STZ((z + 64) | (1ULL << 56));
  AMX_CLR();
  const float* z0 = (const float*)Z; const float* z1 = (const float*)((char*)Z + 64);
  bool relu_ok = true, fmadd_ok = true;
  for (int i = 0; i < 16; ++i) { if (z0[i] != std::fmax(X[i], 0.f)) relu_ok = false; if (z1[i] != X[i]) fmadd_ok = false; }
  std::printf("correctness: max/relu %s, fmadd %s\n", relu_ok ? "OK" : "FAIL", fmadd_ok ? "OK" : "FAIL");

  std::printf("\nSoftmax feasibility on M1 AMX VECFP:\n"
              "  subtract-max : mode1 z-x*y          -> lane-wise, OK\n"
              "  exp(r) poly  : ~5-6 mode0 fmadd      -> OK (cost = 5-6 x fmadd cyc)\n"
              "  exp 2^k      : VECINT shift+reinterp -> feasible, extra ops\n"
              "  rowSUM       : matmul-with-ones (FMA)-> AMX-native, OK\n"
              "  rowMAX       : NO horizontal op       -> OBSTACLE (transpose or skip-max)\n"
              "Estimate ~8-12 VECFP/VECINT ops per score element for exp+scale. Compare\n"
              "that cost (x cyc/op above) against the Q.K^T and .V GEMM cost per row to\n"
              "judge whether in-register softmax keeps the fused-attention win.\n");
  free(X); free(Y); free(Z); return 0;
}
