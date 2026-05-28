// Gate 3: the AMX int4 dequant-GEMV kernel.
// y[N] = W[N,K] * x[K], with W in signed int4 (i - 8), per-tensor scale.
// Pipeline (one row at a time):
//   build dequant table (16 fp32) once, LDX -> X[1]
//   zero Z[0] via LDZ
//   per K-block of 128 weights = 1 LDX of int4 (64B) + 8x { LDY x-chunk, genlut, vecfp }
//   STZ Z[0] -> sum-reduce 16 fp32 -> y[r]
//
// Validates byte-exact-ish vs a scalar reference on a small shape, then measures
// throughput at GPT-2-small scale (N=60000, K=2048, 123M weights) — directly
// comparable to bench/amx/decode_floor.cc (fp32 9.6ms, NEON int4 12.66ms, ORT 11ms).

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

// genlut mode 11 (u4 -> 32-bit lookup, 16 lanes). Encoding pinned from corsix/amx genlut.c.
static inline constexpr uint64_t genlut_op(int mode, int tbl_reg, bool tbl_y,
                                           bool dest_z, bool dest_y, int dest_reg,
                                           bool src_y, int src_off) {
  return (uint64_t(mode) << 53) | (uint64_t(tbl_reg) << 60) |
         (uint64_t(tbl_y) << 59) | (uint64_t(dest_z) << 26) |
         (uint64_t(dest_y) << 25) | (uint64_t(dest_reg) << 20) |
         (uint64_t(src_y) << 10) | uint64_t(src_off & 0x1FF);
}

// vecfp Z[zrow] += X[xreg] * Y[yreg]; fp32 single row; lane mode 4, ALU mode 0.
// x_off / y_off are byte offsets within the X / Y register file (xreg * 64).
static inline constexpr uint64_t vecfp_fma_fp32(int zrow, int x_off, int y_off) {
  return (4ULL << 42)             // lane width mode: f32 one row
       | (uint64_t(zrow) << 20)   // Z row to accumulate into
       | (uint64_t(x_off) << 10)
       | uint64_t(y_off);
  // ALU mode = 0 (z + x*y), indexed = 0, M1: no broadcast / shuffle — all-zero bits suffice.
}

// B=4 batched kernel: process 4 rows together per K-chunk so the 8 LDYs are
// shared across all 4 rows (cuts LDY count 4x). 16 Z accumulators (4 rows × 4
// chains). Reduction at the end of each batch.
void amx_int4_gemv_b4(const uint8_t* W, float scale,
                      const float* x, float* y,
                      int64_t N, int64_t K) {
  alignas(64) float table[16];
  for (int i = 0; i < 16; ++i) table[i] = scale * float(i - 8);
  alignas(64) float zero64[16] = {0};
  alignas(64) float z_buf[16][16];   // 16 accumulator rows × 16 lanes

  static const uint64_t g_ops[8] = {
    genlut_op(11, 1, false, false, false, 2, false,  0),
    genlut_op(11, 1, false, false, false, 3, false,  8),
    genlut_op(11, 1, false, false, false, 4, false, 16),
    genlut_op(11, 1, false, false, false, 5, false, 24),
    genlut_op(11, 1, false, false, false, 2, false, 32),
    genlut_op(11, 1, false, false, false, 3, false, 40),
    genlut_op(11, 1, false, false, false, 4, false, 48),
    genlut_op(11, 1, false, false, false, 5, false, 56),
  };
  // v_ops[row_in_batch][segment]: Z[row*4 + (s%4)] += X[2 + (s%4)] * Y[s].
  uint64_t v_ops[4][8];
  for (int r = 0; r < 4; ++r) {
    for (int s = 0; s < 8; ++s) {
      v_ops[r][s] = vecfp_fma_fp32(r * 4 + (s & 3),
                                   128 + (s & 3) * 64,
                                   s * 64);
    }
  }

  AMX_SET();
  AMX_LDX(reinterpret_cast<uint64_t>(table) | (1ULL << 56));

  const int64_t K_PER_LDX = 128;
  const int64_t Kchunks = K / K_PER_LDX;

  for (int64_t r0 = 0; r0 < N; r0 += 4) {
    // Clear 16 Z accumulator rows.
    for (int z = 0; z < 16; ++z)
      AMX_LDZ(reinterpret_cast<uint64_t>(zero64) | (uint64_t(z) << 56));

    for (int64_t c = 0; c < Kchunks; ++c) {
      const float* xb = x + c * K_PER_LDX;
      // 8 LDYs SHARED across all 4 rows of this batch (the win).
      AMX_LDY(reinterpret_cast<uint64_t>(xb +   0) | (0ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  16) | (1ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  32) | (2ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  48) | (3ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  64) | (4ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  80) | (5ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  96) | (6ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb + 112) | (7ULL << 56));
      for (int r = 0; r < 4; ++r) {
        const uint8_t* wptr = W + (r0 + r) * (K / 2) + c * 64;
        AMX_LDX(reinterpret_cast<uint64_t>(wptr) | (0ULL << 56));
        for (int s = 0; s < 8; ++s) {
          AMX_GENLUT(g_ops[s]);
          AMX_VECFP(v_ops[r][s]);
        }
      }
    }

    // STZ all 16 accumulator rows, reduce per output row.
    for (int z = 0; z < 16; ++z)
      AMX_STZ(reinterpret_cast<uint64_t>(z_buf[z]) | (uint64_t(z) << 56));
    for (int r = 0; r < 4; ++r) {
      float32x4_t acc = vdupq_n_f32(0);
      for (int z = 0; z < 4; ++z) {
        const float* zp = z_buf[r * 4 + z];
        acc = vaddq_f32(acc, vld1q_f32(zp));
        acc = vaddq_f32(acc, vld1q_f32(zp + 4));
        acc = vaddq_f32(acc, vld1q_f32(zp + 8));
        acc = vaddq_f32(acc, vld1q_f32(zp + 12));
      }
      y[r0 + r] = vaddvq_f32(acc);
    }
  }

  AMX_CLR();
}

void amx_int4_gemv(const uint8_t* W, float scale,
                   const float* x, float* y,
                   int64_t N, int64_t K) {
  alignas(64) float table[16];
  for (int i = 0; i < 16; ++i) table[i] = scale * float(i - 8);
  alignas(64) float zero64[16] = {0};
  alignas(64) float z_rows[4][16];

  // 4-way ILP: 4 independent chains, each owning one X-dest reg, one Y reg, one Z row.
  //   chain c ∈ {0..3}:  genlut dest = X[2+c],  LDY into Y[c],  vecfp into Z[c]
  // The 8 segments of one X[0] LDX map to chains s%4, so each chain handles 2 segs.
  // (g_ops[s] and v_ops[s] are the pre-baked operands for segment s.)
  static const uint64_t g_ops[8] = {
    genlut_op(11, 1, false, false, false, 2, false,  0),  // chain 0
    genlut_op(11, 1, false, false, false, 3, false,  8),  // chain 1
    genlut_op(11, 1, false, false, false, 4, false, 16),  // chain 2
    genlut_op(11, 1, false, false, false, 5, false, 24),  // chain 3
    genlut_op(11, 1, false, false, false, 2, false, 32),  // chain 0 again
    genlut_op(11, 1, false, false, false, 3, false, 40),  // chain 1
    genlut_op(11, 1, false, false, false, 4, false, 48),
    genlut_op(11, 1, false, false, false, 5, false, 56),
  };
  // Each segment owns a distinct Y reg (8 total) so the LDYs can all issue first,
  // independent of the genlut/vecfp dependency chains.
  static const uint64_t v_ops[8] = {
    vecfp_fma_fp32(0, 128,   0),  // Z[0] += X[2] * Y[0]
    vecfp_fma_fp32(1, 192,  64),  // Z[1] += X[3] * Y[1]
    vecfp_fma_fp32(2, 256, 128),  // Z[2] += X[4] * Y[2]
    vecfp_fma_fp32(3, 320, 192),  // Z[3] += X[5] * Y[3]
    vecfp_fma_fp32(0, 128, 256),  // Z[0] += X[2] * Y[4]   (X[2] reused after Z[0]-vecfp drained it)
    vecfp_fma_fp32(1, 192, 320),
    vecfp_fma_fp32(2, 256, 384),
    vecfp_fma_fp32(3, 320, 448),
  };

  AMX_SET();
  AMX_LDX(reinterpret_cast<uint64_t>(table) | (1ULL << 56));  // table -> X[1]

  const int64_t K_PER_LDX = 128;
  const int64_t Kchunks = K / K_PER_LDX;

  for (int64_t r = 0; r < N; ++r) {
    // Clear Z[0..3] (one LDZ per row).
    AMX_LDZ(reinterpret_cast<uint64_t>(zero64) | (0ULL << 56));
    AMX_LDZ(reinterpret_cast<uint64_t>(zero64) | (1ULL << 56));
    AMX_LDZ(reinterpret_cast<uint64_t>(zero64) | (2ULL << 56));
    AMX_LDZ(reinterpret_cast<uint64_t>(zero64) | (3ULL << 56));

    const uint8_t* wrow = W + r * (K / 2);
    for (int64_t c = 0; c < Kchunks; ++c) {
      AMX_LDX(reinterpret_cast<uint64_t>(wrow + c * 64) | (0ULL << 56));
      const float* xb = x + c * K_PER_LDX;
      // Hoist: issue all 8 LDYs first (each into a distinct Y reg). They have no
      // dependency on the genlut/vecfp chain — letting the AMX issue queue pipeline
      // them in parallel with the genlut→vecfp dependency chain.
      AMX_LDY(reinterpret_cast<uint64_t>(xb +   0) | (0ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  16) | (1ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  32) | (2ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  48) | (3ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  64) | (4ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  80) | (5ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb +  96) | (6ULL << 56));
      AMX_LDY(reinterpret_cast<uint64_t>(xb + 112) | (7ULL << 56));
      for (int s = 0; s < 8; ++s) {
        AMX_GENLUT(g_ops[s]);
        AMX_VECFP(v_ops[s]);
      }
    }

    // Reduce the 4 Z rows → scalar y[r].
    AMX_STZ(reinterpret_cast<uint64_t>(z_rows[0]) | (0ULL << 56));
    AMX_STZ(reinterpret_cast<uint64_t>(z_rows[1]) | (1ULL << 56));
    AMX_STZ(reinterpret_cast<uint64_t>(z_rows[2]) | (2ULL << 56));
    AMX_STZ(reinterpret_cast<uint64_t>(z_rows[3]) | (3ULL << 56));
    float32x4_t acc = vdupq_n_f32(0);
    for (int z = 0; z < 4; ++z) {
      acc = vaddq_f32(acc, vld1q_f32(z_rows[z]));
      acc = vaddq_f32(acc, vld1q_f32(z_rows[z] + 4));
      acc = vaddq_f32(acc, vld1q_f32(z_rows[z] + 8));
      acc = vaddq_f32(acc, vld1q_f32(z_rows[z] + 12));
    }
    y[r] = vaddvq_f32(acc);
  }

  AMX_CLR();
}

// Scalar reference (correctness ground truth)
static void scalar_int4_gemv(const uint8_t* W, float scale,
                             const float* x, float* y,
                             int64_t N, int64_t K) {
  for (int64_t r = 0; r < N; ++r) {
    double acc = 0;
    for (int64_t k = 0; k < K; ++k) {
      uint8_t byte = W[r * (K / 2) + k / 2];
      int nibble = (k & 1) ? (byte >> 4) : (byte & 0xf);
      acc += double(scale * float(nibble - 8)) * double(x[k]);
    }
    y[r] = float(acc);
  }
}

int main() {
  // (1) correctness vs scalar ref
  {
    const int64_t N = 16, K = 256;
    std::vector<uint8_t> W(N * K / 2);
    std::vector<float> x(K), y_amx(N), y_ref(N);
    for (size_t i = 0; i < W.size(); ++i) W[i] = uint8_t((i * 31) & 0xff);
    for (int64_t i = 0; i < K; ++i) x[i] = std::sin(0.07 * i);
    const float scale = 0.125f;
    amx_int4_gemv(W.data(), scale, x.data(), y_amx.data(), N, K);
    scalar_int4_gemv(W.data(), scale, x.data(), y_ref.data(), N, K);
    float maxd = 0;
    for (int i = 0; i < N; ++i) maxd = std::max(maxd, std::abs(y_amx[i] - y_ref[i]));
    std::printf("[correctness 4-way] N=%lld K=%lld  max-abs-diff vs scalar = %.2e\n",
                (long long)N, (long long)K, maxd);
    // Also test the B=4 batched kernel
    std::vector<float> y_b4(N);
    amx_int4_gemv_b4(W.data(), scale, x.data(), y_b4.data(), N, K);
    float maxd_b4 = 0;
    for (int i = 0; i < N; ++i) maxd_b4 = std::max(maxd_b4, std::abs(y_b4[i] - y_ref[i]));
    std::printf("[correctness B=4 ] N=%lld K=%lld  max-abs-diff vs scalar = %.2e\n",
                (long long)N, (long long)K, maxd_b4);
    if (maxd_b4 > 1e-4f) {
      std::printf("  b4[0..7]: "); for (int i = 0; i < 8; ++i) std::printf("%.4f ", y_b4[i]);
      std::printf("\n  ref[0..7]: "); for (int i = 0; i < 8; ++i) std::printf("%.4f ", y_ref[i]);
      std::printf("\n");
      return 1;
    }
    if (maxd > 1e-4f) {
      std::printf("  amx[0..7]: "); for (int i = 0; i < 8; ++i) std::printf("%.4f ", y_amx[i]);
      std::printf("\n  ref[0..7]: "); for (int i = 0; i < 8; ++i) std::printf("%.4f ", y_ref[i]);
      std::printf("\n");
      return 1;
    }
  }

  // (2) perf at GPT-2-small scale (same shape as decode_floor.cc -> direct compare)
  const int64_t N = 60000, K = 2048;
  std::vector<uint8_t> W(N * K / 2);
  std::vector<float> x(K), y(N);
  for (size_t i = 0; i < W.size(); ++i) W[i] = uint8_t(i & 0xff);
  for (int64_t i = 0; i < K; ++i) x[i] = float(i % 7) * 0.1f;
  const float scale = 0.125f;
  const double bytes_in = double(N) * K * 0.5;  // int4 weight bytes streamed per call

  auto bench = [&](const char* name, auto fn) {
    fn(); fn();
    double best = 1e30;
    for (int it = 0; it < 6; ++it) {
      auto t0 = clk::now();
      fn();
      best = std::min(best, ms(clk::now() - t0));
    }
    std::printf("  %-32s %6.2f ms   %6.1f GB/s int4 in\n", name, best,
                bytes_in / (best / 1e3) / 1e9);
    return best;
  };

  double t_amx = bench("AMX int4 GEMV 4-way ILP", [&] {
    amx_int4_gemv(W.data(), scale, x.data(), y.data(), N, K);
  });
  double t_b4 = bench("AMX int4 GEMV B=4 batched", [&] {
    amx_int4_gemv_b4(W.data(), scale, x.data(), y.data(), N, K);
  });
  (void)t_b4;

  std::printf("\nfirst 4 outputs: %.3f %.3f %.3f %.3f\n", y[0], y[1], y[2], y[3]);
  std::printf("\nbars to beat at N=60000 K=2048 (from decode_floor.cc):\n"
              "  fp32 NEON GEMV       9.61 ms\n"
              "  int8 NEON GEMV       9.26 ms\n"
              "  int4 NEON GEMV      12.66 ms\n"
              "  Accelerate sgemv    14.67 ms\n"
              "  ORT GPT-2 decode    ~11   ms\n"
              "  int4 memory floor    2.41 ms (AMX LDX-only)\n"
              "this kernel:          %6.2f ms\n", t_amx);
  return 0;
}
