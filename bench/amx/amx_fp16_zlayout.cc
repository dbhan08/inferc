// STEP 1: fp16 FMA16 Z-layout probe. corsix says matrix fma16 writes "8 Z
// registers"; Zhou/meekolab say "32". The drain cost of the blocked-fp16 kernel
// depends entirely on HOW MANY Z rows one fp16 outer product occupies and their
// layout -- so resolve it empirically. Issue ONE known 32x32 fp16 outer product
// (skip-Z = fresh), then STZ all 64 Z rows and inspect which are written.
//
// Inputs chosen to identify the (i,j)->Z[row][col] map: X[i]=i+1, Y[j]=1, so the
// true outer product is C[i][j] = i+1 (each output ROW i is the constant i+1).
// Run BOTH accumulate modes: bit62=0 (fp16 Z) and bit62=1 (fp32 Z, widening).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"

static inline uint64_t Fma16(int zrow, int xoff, int yoff, bool skipZ, bool f32acc) {
  return (uint64_t(zrow) << 20) | (uint64_t(xoff) << 10) | uint64_t(yoff) |
         (skipZ ? (1ULL << 27) : 0) | (f32acc ? (1ULL << 62) : 0);
}

int main() {
  __fp16* X = nullptr; __fp16* Y = nullptr; void* Zbuf = nullptr;
  posix_memalign((void**)&X, 128, 64);            // 32 fp16
  posix_memalign((void**)&Y, 128, 64);
  posix_memalign(&Zbuf, 128, 64 * 64);            // 64 Z rows x 64 bytes
  for (int i = 0; i < 32; ++i) { X[i] = (__fp16)(i + 1); Y[i] = (__fp16)1.0f; }
  const uint64_t x = (uint64_t)X, y = (uint64_t)Y, z = (uint64_t)Zbuf;

  for (int f32acc = 0; f32acc <= 1; ++f32acc) {
    std::printf("==== fma16 matrix outer product, bit62(f32acc)=%d  (X[i]=i+1, Y[j]=1) ====\n", f32acc);
    AMX_SET();
    std::memset(Zbuf, 0, 64 * 64);
    // load Z=0 first by storing zeros is not enough; use skip-Z on the FMA to overwrite.
    AMX_LDX(x | (0ULL << 56));                     // X reg0 = 32 fp16
    AMX_LDY(y | (0ULL << 56));                     // Y reg0 = 32 fp16
    AMX_FMA16(Fma16(/*zrow=*/0, /*xoff=*/0, /*yoff=*/0, /*skipZ=*/true, (bool)f32acc));
    for (int r = 0; r < 64; ++r) AMX_STZ((z + (uint64_t)r * 64) | ((uint64_t)r << 56));
    AMX_CLR();

    int nonzero_rows = 0;
    for (int r = 0; r < 64; ++r) {
      const uint8_t* row = (const uint8_t*)Zbuf + r * 64;
      bool nz = false; for (int b = 0; b < 64; ++b) if (row[b]) { nz = true; break; }
      if (!nz) continue;
      ++nonzero_rows;
      if (f32acc) {
        const float* v = (const float*)row;        // 16 fp32 per row
        std::printf("Z[%2d] f32: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f ...\n",
                    r, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
      } else {
        const __fp16* v = (const __fp16*)row;       // 32 fp16 per row
        std::printf("Z[%2d] f16: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f ...\n",
                    r, (float)v[0], (float)v[1], (float)v[2], (float)v[3],
                    (float)v[4], (float)v[5], (float)v[6], (float)v[7]);
      }
    }
    std::printf(">>> %d of 64 Z rows written.\n\n", nonzero_rows);
  }
  std::printf("Reading: the count of written rows = drain cost per tile. The value\n"
              "pattern (constant-per-row => row indexes output-i; ramp-within-row =>\n"
              "col indexes output-j) reveals the (i,j) layout and whether f32acc\n"
              "interleaves pairs (2x the rows) vs f16 (half).\n");
  free(X); free(Y); free(Zbuf);
  return 0;
}
