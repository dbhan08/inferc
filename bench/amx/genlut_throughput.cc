// Gate 2 of the AMX low-bit decode bet: measure AMX `genlut` throughput.
//
// The bet rests on: AMX genlut (mode 11: u4 -> 32-bit lookup, 16 lanes) can
// dequant int4 weights to fp32 fast enough to reach the int4 memory floor
// (~25 GB/s on M1, per decode_floor.cc). If it can't, the bet is dead.
//
// Method: stream a 61 MB int4 buffer through (A) LDX-only and (B) LDX + 8x
// genlut per 64B chunk. If B ~ A, genlut keeps up with memory -> PASS.
// If B >> A, genlut is the new bottleneck -> FAIL.
//
// Encoding pinned from corsix/amx genlut.md + genlut.c (canonical):
//   mode  @ bit 53 (4)   src_off  @ bit 0 (9)    src_from_y  @ bit 10
//   dest_reg @ bit 20 (3)   dest_y @ bit 25      dest_z @ bit 26
//   tbl_from_y @ bit 59     tbl_reg @ bit 60 (3)

#include <Accelerate/Accelerate.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

static inline constexpr uint64_t genlut_op(int mode, int tbl_reg, bool tbl_y,
                                           bool dest_z, bool dest_y, int dest_reg,
                                           bool src_y, int src_off) {
  return (uint64_t(mode) << 53) | (uint64_t(tbl_reg) << 60) |
         (uint64_t(tbl_y) << 59) | (uint64_t(dest_z) << 26) |
         (uint64_t(dest_y) << 25) | (uint64_t(dest_reg) << 20) |
         (uint64_t(src_y) << 10) | uint64_t(src_off & 0x1FF);
}

int main() {
  const int64_t BYTES = 61'440'000;  // 61 MB int4 = 122.88M weights (GPT-2-small scale)
  std::vector<uint8_t> Wi4(BYTES);
  for (size_t i = 0; i < BYTES; ++i) Wi4[i] = (uint8_t)(i & 0xff);

  // 16-entry fp32 dequant table: u4 index 0..15 -> -8..+7 (signed int4 interp).
  alignas(64) float table[16];
  for (int i = 0; i < 16; ++i) table[i] = float(i - 8);
  alignas(64) float sink_buf[16] = {0};

  auto bench = [&](const char* name, double bytes, auto fn) {
    for (int w = 0; w < 2; ++w) fn();
    double best = 1e30;
    for (int it = 0; it < 6; ++it) {
      auto t0 = clk::now();
      fn();
      best = std::min(best, ms(clk::now() - t0));
    }
    std::printf("  %-32s %6.2f ms   %6.1f GB/s\n", name, best,
                bytes / (best / 1e3) / 1e9);
    return best;
  };

  // (A) LDX-only stream: AMX memory-read floor for this path.
  double t_a = bench("LDX stream 61MB (no genlut)", BYTES, [&] {
    AMX_SET();
    for (size_t i = 0; i + 64 <= (size_t)BYTES; i += 64) {
      AMX_LDX(reinterpret_cast<uint64_t>(Wi4.data() + i) | (0ULL << 56));
    }
    AMX_STX(reinterpret_cast<uint64_t>(sink_buf) | (0ULL << 56));
    AMX_CLR();
  });

  // (B) LDX + 8 genluts/chunk: every 64B int4 loaded -> 128 fp32 dequanted
  //     (8 ops at offsets 0,8,...,56; each consumes 16 u4 nibbles, produces 16 f32).
  bench("LDX + genlut x8 stream 61MB", BYTES, [&] {
    AMX_SET();
    AMX_LDX(reinterpret_cast<uint64_t>(table) | (1ULL << 56));  // table -> X[1]
    constexpr uint64_t g0 = genlut_op(11, 1, false, false, false, 2, false, 0);
    constexpr uint64_t g1 = genlut_op(11, 1, false, false, false, 2, false, 8);
    constexpr uint64_t g2 = genlut_op(11, 1, false, false, false, 2, false, 16);
    constexpr uint64_t g3 = genlut_op(11, 1, false, false, false, 2, false, 24);
    constexpr uint64_t g4 = genlut_op(11, 1, false, false, false, 2, false, 32);
    constexpr uint64_t g5 = genlut_op(11, 1, false, false, false, 2, false, 40);
    constexpr uint64_t g6 = genlut_op(11, 1, false, false, false, 2, false, 48);
    constexpr uint64_t g7 = genlut_op(11, 1, false, false, false, 2, false, 56);
    for (size_t i = 0; i + 64 <= (size_t)BYTES; i += 64) {
      AMX_LDX(reinterpret_cast<uint64_t>(Wi4.data() + i) | (0ULL << 56));
      AMX_GENLUT(g0); AMX_GENLUT(g1); AMX_GENLUT(g2); AMX_GENLUT(g3);
      AMX_GENLUT(g4); AMX_GENLUT(g5); AMX_GENLUT(g6); AMX_GENLUT(g7);
    }
    AMX_STX(reinterpret_cast<uint64_t>(sink_buf) | (2ULL << 56));  // observe X[2]
    AMX_CLR();
  });

  std::printf("\nfirst 8 dequanted fp32 from last chunk (sanity): ");
  for (int i = 0; i < 8; ++i) std::printf("%.1f ", sink_buf[i]);
  std::printf("\nGate 2 PASS if (B) is at or near (A); FAIL if (B) >> (A).\n");
  return 0;
}
