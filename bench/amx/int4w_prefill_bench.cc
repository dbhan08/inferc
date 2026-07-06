// Tier B: int4-weight, fp32-compute prefill GEMM kernel.
//
// Hypothesis: at LM head shapes (large N, memory-bound) where the fp32 kernel
// loses to Accelerate at 0.77-0.85×, cutting B memory traffic 8× via int4
// weight packing should plausibly flip the win.
//
// Kernel structure (same BLIS+Kc as fp32 kernel, but with int4 B):
//   for jc, pc: pack B[pc:pc+Kc, jc:jc+Nc] as int4 nibbles (Kc*Nc/2 bytes)
//   for i0, jr inner:
//     for kk: LDX 32B int4, GENLUT×4 (u4→f32, mode 11), LDY 16 fp32 of A,
//             FMA32×4 across Z banks 0..3
//
// Per-tensor symmetric int4 quant: scale = max(|W|)/7. Table = scale × (i-8)
// preloaded into X[1]. Genlut consumes from X[7] (LDX target).
//
// Compared against Accelerate fp32 sgemm (since Accelerate cblas has no int4
// GEMM, this is the apples-to-apples comparison: fp32 weights baseline vs
// int4-packed weights with fp32 compute).

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

static inline uint64_t Fma32Op(int zbase, int x_off_bytes, bool first) {
  return (uint64_t(zbase) << 20) | (uint64_t(x_off_bytes) << 10) |
         (first ? (1ULL << 27) : 0);
}

// GENLUT mode 11 (u4 → f32, 16 lanes). table_reg=tbl, dest_reg=dst, src offset.
static inline constexpr uint64_t genlut_u4_f32(int tbl, int dst, int src_off) {
  return (uint64_t(11) << 53) | (uint64_t(tbl) << 60) |
         (uint64_t(dst) << 20) | uint64_t(src_off & 0x1FF);
}

// Quantize B[K, N] fp32 → int4 packed (nibble pairs in bytes), per-tensor scale.
// Returns scale; out has K*N/2 bytes. Even k uses low nibble, odd k high nibble.
static float quantize_int4_per_tensor(const float* B, int64_t K, int64_t N,
                                      std::vector<uint8_t>& out) {
  float maxabs = 0;
  for (int64_t i = 0; i < K * N; ++i) maxabs = std::fmax(maxabs, std::fabs(B[i]));
  const float scale = maxabs / 7.f;          // signed range [-8..7], use 7 for symmetry
  const float inv = (scale > 0.f) ? 1.f / scale : 0.f;
  out.assign(size_t(K) * N / 2, 0);
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      int q = int(std::round(B[k * N + n] * inv));
      if (q >  7) q =  7;
      if (q < -8) q = -8;
      uint8_t u = uint8_t(q + 8);              // encode as 0..15 (so genlut table = scale*(i-8))
      size_t byte_idx = (size_t(k) * N + n) / 2;
      if (((k * N + n) & 1) == 0) out[byte_idx] |= u;
      else                         out[byte_idx] |= u << 4;
    }
  }
  return scale;
}

// Pack the (pc-row, jc-col panel) of int4 weights into a contiguous Kc × Nc
// nibble buffer matching the microkernel's read pattern.
static void pack_panel_int4(const uint8_t* Wq, int64_t K, int64_t N,
                            int64_t pc, int64_t jc, int64_t Kc_eff,
                            int64_t Nc_main, uint8_t* packW) {
  // For each row k of the panel, copy Nc_main int4 values (= Nc_main/2 bytes
  // if pc+k and jc are both even — they are, by construction).
  for (int64_t kk = 0; kk < Kc_eff; ++kk) {
    // source: row pc+kk of W, cols jc..jc+Nc_main
    // each W row has N nibbles = N/2 bytes. Column jc starts at byte jc/2.
    const uint8_t* src = Wq + (size_t(pc + kk) * N + jc) / 2;
    std::memcpy(packW + kk * (Nc_main / 2), src, Nc_main / 2);
  }
}

// The int4-weight, fp32-compute BLIS+Kc microkernel for one (i0, jr) tile.
// Layout (matches int4_gemv_amx.cc convention so genlut src_off works):
//   X[0] = LDX target for 64 bytes of int4 weights (128 nibbles)
//   X[1] = scaled dequant table (16 fp32 = 64 bytes)
//   X[2..5] = genlut outputs feeding FMA32 banks 0..3
// src_off in genlut is a byte offset into the X file (0..511), so src_off=0,8,
// 16,24 reads from X[0] bytes [0,8), [8,16), [16,24), [24,32).
// FMA32 reads X[2..5] at byte offsets 128, 192, 256, 320 within the X file.
static void int4w_microkernel(const uint8_t* packW, const float* At,
                              int64_t Kc_eff, int64_t pc, int64_t Nc_main,
                              int64_t i0, int64_t jr, int64_t M, int64_t N,
                              int64_t jc, float* C, bool is_first_pc) {
  // pc>0 case: load partial Z from C
  if (!is_first_pc) {
    for (int t = 0; t < 4; ++t)
      for (int j = 0; j < 16; ++j)
        AMX_LDZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
                (uint64_t(4 * j + t) << 56));
  }

  // GENLUT operands: read 8-byte chunks of X[0] (the LDX-loaded weights) at
  // byte offsets 0/8/16/24 (within the X file, which puts them in X[0]),
  // table at X[1], dequant 16 fp32 → X[2..5] in turn.
  static const uint64_t g0 = genlut_u4_f32(/*tbl=*/1, /*dst=*/2, /*src_off=*/ 0);
  static const uint64_t g1 = genlut_u4_f32(/*tbl=*/1, /*dst=*/3, /*src_off=*/ 8);
  static const uint64_t g2 = genlut_u4_f32(/*tbl=*/1, /*dst=*/4, /*src_off=*/16);
  static const uint64_t g3 = genlut_u4_f32(/*tbl=*/1, /*dst=*/5, /*src_off=*/24);

  for (int64_t kk = 0; kk < Kc_eff; ++kk) {
    const bool first = (is_first_pc && kk == 0);
    // LDX 64 bytes of int4 weights at the current jr column offset.
    // We only need 32 bytes (64 nibbles for one ILP cycle) but LDX always
    // brings 64; the unused 32 bytes are ignored by the 4 genluts below.
    const uint8_t* wptr = packW + kk * (Nc_main / 2) + jr / 2;
    AMX_LDX(reinterpret_cast<uint64_t>(wptr) | (0ULL << 56));
    AMX_GENLUT(g0);
    AMX_GENLUT(g1);
    AMX_GENLUT(g2);
    AMX_GENLUT(g3);

    // LDY one column of A (16 fp32 values from At[pc+kk, i0:i0+16])
    AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc + kk) * M + i0]));

    // FMA32 banks 0..3 read X[2..5] at byte offsets 128, 192, 256, 320.
    AMX_FMA32(Fma32Op(0, 128, first));
    AMX_FMA32(Fma32Op(1, 192, first));
    AMX_FMA32(Fma32Op(2, 256, first));
    AMX_FMA32(Fma32Op(3, 320, first));
  }

  // Always STZ — pc>0 has running sum, pc=last writes final
  for (int t = 0; t < 4; ++t)
    for (int j = 0; j < 16; ++j)
      AMX_STZ(reinterpret_cast<uint64_t>(C + (i0 + j) * N + jc + jr + 16 * t) |
              (uint64_t(4 * j + t) << 56));
}

// BLIS+Kc int4-weight fp32-compute kernel.
static void amx_sgemm_int4w(const float* A, const uint8_t* Wq, float scale,
                            float* C, int64_t M, int64_t N, int64_t K,
                            int Nc, int Kc,
                            std::vector<float>& At_scratch,
                            std::vector<uint8_t>& packW_scratch) {
  At_scratch.resize(size_t(K) * M);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k) At_scratch[k * M + i] = A[i * K + k];
  const float* At = At_scratch.data();

  // Build the per-tensor scaled dequant table: table[i] = scale * (i - 8)
  alignas(64) float table[16];
  for (int i = 0; i < 16; ++i) table[i] = scale * float(i - 8);

  AMX_SET();
  // Load scaled dequant table into X[1] (persistent across all microkernel calls)
  AMX_LDX(reinterpret_cast<uint64_t>(table) | (1ULL << 56));

  for (int64_t jc = 0; jc < N; jc += Nc) {
    int64_t Nc_eff  = std::min<int64_t>(Nc, N - jc);
    int64_t Nc_main = (Nc_eff / 64) * 64;     // round down to multiple of 64
    if (Nc_main == 0) continue;               // tail not handled in this proof-of-concept

    packW_scratch.resize(size_t(Kc) * Nc_main / 2 + 64);  // +64B pad: LDX always reads 64B
    for (int64_t pc = 0; pc < K; pc += Kc) {
      int64_t Kc_eff = std::min<int64_t>(Kc, K - pc);
      pack_panel_int4(Wq, K, N, pc, jc, Kc_eff, Nc_main, packW_scratch.data());
      const bool is_first_pc = (pc == 0);

      for (int64_t i0 = 0; i0 < M; i0 += 16) {
        for (int64_t jr = 0; jr < Nc_main; jr += 64) {
          int4w_microkernel(packW_scratch.data(), At, Kc_eff, pc, Nc_main,
                            i0, jr, M, N, jc, C, is_first_pc);
        }
      }
    }
  }
  AMX_CLR();
}

int main() {
  // Force single-thread Accelerate for apples-to-apples comparison.
  // (Tested via VECLIB_MAXIMUM_THREADS=1 in env; library may also use OMP.)

  struct Shape { int M, N, K; const char* model; const char* op; };
  // Focus: shapes where the fp32 kernel currently loses (LM head + FFN1 large).
  // Plus QKV for sanity.
  const Shape shapes[] = {
    { 128,  2048,  2048, "GPT-2-small",    "QKV     (sanity)" },
    { 128, 60000,  2048, "GPT-2-small",    "LM head" },
    { 128, 32000,  2048, "TinyLlama-1.1B", "LM head" },
    { 128, 32000,  4096, "Llama-7B",       "LM head" },
    { 128, 11008,  4096, "Llama-7B",       "FFN1   " },
  };

  std::vector<float> At_scratch;
  std::vector<uint8_t> packW_scratch, Wq;

  std::printf("Tier B: int4-weight, fp32-compute prefill GEMM at LM head shapes.\n");
  std::printf("Per-tensor symmetric int4 quant on B; A and C remain fp32.\n");
  std::printf("Memory traffic: B fp32 = K*N*4 bytes; B int4 = K*N/2 bytes (8x cut).\n\n");

  std::printf("  %-32s %-8s | %-15s | %-15s | accuracy\n",
              "model / op", "shape", "Accelerate fp32", "ours int4w fp32c");

  double sum_log_ratio = 0; int count = 0;
  for (auto& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K);
    std::vector<float> B(size_t(s.K) * s.N);
    std::vector<float> C_ref(size_t(s.M) * s.N, 0.f);
    std::vector<float> C(size_t(s.M) * s.N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7)  * 0.01f;
    // Use a non-pathological B so int4 quant has meaningful dynamic range
    for (size_t i = 0; i < B.size(); ++i) B[i] = (float(i % 13) - 6.f) * 0.05f;
    const double flops = 2.0 * s.M * double(s.N) * s.K;

    // Quantize B once outside the bench loop
    float scale = quantize_int4_per_tensor(B.data(), s.K, s.N, Wq);

    auto bench = [&](auto fn) {
      fn(); fn();
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        auto t0 = clk::now(); fn(); best = std::min(best, ms(clk::now() - t0));
      }
      return std::pair<double, double>(best, flops / (best / 1e3) / 1e9);
    };

    // Accelerate baseline (fp32 weights, fp32 sgemm)
    auto [accel_ms, accel_gf] = bench([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s.M, s.N, s.K,
                  1.0f, A.data(), s.K, B.data(), s.N, 0.0f, C_ref.data(), s.N);
    });

    // Ours: int4 weights via genlut, fp32 compute
    // Adaptive Nc/Kc: at very large N, packing the int4 panel is cheap so try larger Nc.
    int Nc = (s.N >= 32000) ? 4096 : (s.N >= 8192 ? 4096 : std::min(s.N, 2048));
    int Kc = (s.K >= 4096) ? 1024 : 2048;
    auto [our_ms, our_gf] = bench([&] {
      amx_sgemm_int4w(A.data(), Wq.data(), scale, C.data(),
                      s.M, s.N, s.K, Nc, Kc, At_scratch, packW_scratch);
    });

    // Accuracy: int4 has limited representational precision so we expect
    // a relative error proportional to the quantization scale, not bit-exact.
    double sum_sq_err = 0, sum_sq_ref = 0;
    size_t spot = std::min<size_t>(C.size(), 65536);
    for (size_t i = 0; i < spot; ++i) {
      double e = double(C[i]) - double(C_ref[i]);
      sum_sq_err += e * e;
      sum_sq_ref += double(C_ref[i]) * double(C_ref[i]);
    }
    double rel_rmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0;
    double ratio = our_gf / accel_gf;
    sum_log_ratio += std::log(ratio);  count++;

    char tag[40]; std::snprintf(tag, sizeof(tag), "[%d,%d,%d]", s.M, s.N, s.K);
    std::printf("  %-12s %-19s %-8s | %5.2fms %4.0fGF | %5.2fms %4.0fGF | rel-rmse %.1e  %.2fx %s\n",
                s.model, s.op, tag, accel_ms, accel_gf, our_ms, our_gf,
                rel_rmse, ratio, ratio >= 1.0 ? "<--BEATS" : "");
  }
  std::printf("\n  geomean ratio across %d shapes: %.2fx\n",
              count, std::exp(sum_log_ratio / count));
  return 0;
}
