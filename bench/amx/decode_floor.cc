// De-risk microbench for the AMX low-bit decode bet.
// Simulates one batch-1 decode token's weight reads at GPT-2-small scale
// (~123M weights) as a big GEMV, in fp32 / int8 / int4, plus raw read bandwidth.
// Question 1 (prize): how much does int4 cut the per-token decode floor vs fp32?
// Question 2 (AMX value): is int4 memory-bound (AMX~=NEON) or dequant-bound
//   (where an AMX genlut dequant could beat NEON)?  We compare int4-GEMV time to
//   the raw int4 read time: if GEMV >> read, dequant is the bottleneck -> AMX wins.
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

int main() {
  const int64_t K = 2048;
  const int64_t ROWS = 60000;            // ROWS*K = 122.88M ~ GPT-2-small params
  const int64_t N = ROWS * K;
  std::printf("weights ~%.1fM (GPT-2-small scale): fp32 %.0fMB  int8 %.0fMB  int4 %.0fMB\n",
              N / 1e6, N * 4 / 1e6, N / 1e6, N * 0.5 / 1e6);

  std::vector<float> Wf(N);
  std::vector<int8_t> Wi8(N);
  std::vector<uint8_t> Wi4(N / 2);
  std::vector<float> x(K), y(ROWS);
  for (int64_t i = 0; i < K; ++i) x[i] = (float)(i % 7) * 0.1f;
  for (int64_t i = 0; i < N; ++i) { Wf[i] = (float)(i % 15 - 7) * 0.1f; Wi8[i] = (int8_t)(i % 15 - 7); }
  for (size_t i = 0; i < Wi4.size(); ++i) Wi4[i] = (uint8_t)((i & 0x0f) | ((i & 0x0f) << 4));

  volatile float sink = 0;
  auto bench = [&](const char* name, double bytes, auto fn) {
    fn(); fn();
    double best = 1e30;
    for (int it = 0; it < 6; ++it) { auto t0 = clk::now(); fn(); best = std::min(best, ms(clk::now() - t0)); }
    std::printf("  %-26s %6.2f ms   %6.1f GB/s\n", name, best, bytes / (best / 1e3) / 1e9);
    return best;
  };

  // (0) raw read bandwidth: stream-sum the int4 buffer (the int4 memory floor)
  double t_rd4 = bench("raw read int4 (61MB)", N * 0.5, [&] {
    uint8x16_t acc = vdupq_n_u8(0);
    for (size_t i = 0; i < Wi4.size(); i += 16) acc = vaddq_u8(acc, vld1q_u8(&Wi4[i]));
    sink = vaddvq_u8(acc);
  });

  // (1) fp32 GEMV — 4 independent accumulators (break the fma-latency chain)
  bench("fp32 GEMV (491MB)", N * 4.0, [&] {
    for (int64_t r = 0; r < ROWS; ++r) {
      const float* w = &Wf[r * K];
      float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
      for (int64_t k = 0; k < K; k += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + k),      vld1q_f32(&x[k]));
        a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4),  vld1q_f32(&x[k + 4]));
        a2 = vfmaq_f32(a2, vld1q_f32(w + k + 8),  vld1q_f32(&x[k + 8]));
        a3 = vfmaq_f32(a3, vld1q_f32(w + k + 12), vld1q_f32(&x[k + 12]));
      }
      y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
    sink = y[0];
  });

  // (2) int8 dequant-GEMV — 4 independent accumulators
  bench("int8 GEMV (123MB)", N * 1.0, [&] {
    for (int64_t r = 0; r < ROWS; ++r) {
      const int8_t* w = &Wi8[r * K];
      float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
      for (int64_t k = 0; k < K; k += 16) {
        int8x16_t v = vld1q_s8(w + k);
        int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vld1q_f32(&x[k]));
        a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(&x[k + 4]));
        a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vld1q_f32(&x[k + 8]));
        a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(&x[k + 12]));
      }
      y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
    sink = y[0];
  });

  // (3) int4 dequant-GEMV — NEON nibble unpack -> signed -> fp32, 4 accumulators
  bench("int4 GEMV (61MB)", N * 0.5, [&] {
    for (int64_t r = 0; r < ROWS; ++r) {
      const uint8_t* w = &Wi4[r * K / 2];
      float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
      for (int64_t k = 0; k < K; k += 16) {
        uint8x8_t p = vld1_u8(w + k / 2);
        uint8x8_t lo = vand_u8(p, vdup_n_u8(0x0f)), hi = vshr_n_u8(p, 4);
        uint8x8x2_t z = vzip_u8(lo, hi);
        int8x16_t v = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(z.val[0], z.val[1])), vdupq_n_s8(8));
        int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_s8(vget_high_s8(v));
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))),  vld1q_f32(&x[k]));
        a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))), vld1q_f32(&x[k + 4]));
        a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),  vld1q_f32(&x[k + 8]));
        a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))), vld1q_f32(&x[k + 12]));
      }
      y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
    sink = y[0];
  });

  // (4) Accelerate (AMX) fp32 sgemv reference
  bench("Accelerate sgemv fp32", N * 4.0, [&] {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)ROWS, (int)K, 1.0f, Wf.data(), (int)K,
                x.data(), 1, 0.0f, y.data(), 1);
    sink = y[0];
  });

  std::printf("\nint4 read floor = %.2f ms; if int4-GEMV is far above this, dequant is the\n"
              "bottleneck -> AMX genlut dequant has headroom. sink=%.1f\n", t_rd4, (float)sink);
  return 0;
}
