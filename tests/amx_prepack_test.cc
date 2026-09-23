// Bit-exactness gate for the pre-packed AMX GEMM of docs/PAPER_DRAFT.md.
//
// The paper claims every output is bit-identical to Accelerate's cblas_sgemm
// at all twelve LLM prefill shapes. The benchmark harness only samples every
// 1023rd element; this test compares the FULL output matrix with a tolerance
// of exactly zero, at shapes that exercise every code path:
//   - the K-block carry (K > Kc, so LDZ/STZ partial-sum reload runs),
//   - a non-multiple-of-64 N (the 16-wide AMX tail + scalar remainder),
//   - the paper's LM-head shape at reduced M so the test stays fast.

#include <gtest/gtest.h>

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "kernels/amx_prepack_gemm.h"

namespace {

struct Shape { int64_t M, N, K; const char* tag; };

// Deterministic, non-trivial operands. Mixed signs and magnitudes so that
// accumulation order actually matters for the low bits.
void Fill(std::vector<float>& v, double freq, double scale) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = float(std::sin(freq * double(i)) * scale);
}

// Returns the number of elements whose bit pattern differs from cblas_sgemm.
int64_t CountMismatches(const Shape& s, float* max_diff_out) {
  const int64_t M = s.M, N = s.N, K = s.K;
  std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N, 0.f),
      R(size_t(M) * N, 0.f);
  Fill(A, 0.013, 1.0);
  Fill(B, 0.0071, 0.5);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K, 1.0f,
              A.data(), (int)K, B.data(), (int)N, 0.0f, R.data(), (int)N);
  auto W = inferc::rt::AmxPackWeight(B.data(), N, K);
  inferc::rt::AmxPrepackedSgemm(A.data(), W, C.data(), M);
  int64_t bad = 0; float max_diff = 0.f;
  for (size_t i = 0; i < C.size(); ++i) {
    const float d = std::fabs(C[i] - R[i]);
    if (d > max_diff) max_diff = d;
    if (d != 0.0f) ++bad;
  }
  std::printf("[AmxPrepack] %-28s M=%lld N=%lld K=%lld  mismatches=%lld/%zu max_abs_diff=%.3e\n",
              s.tag, (long long)M, (long long)N, (long long)K, (long long)bad, C.size(),
              max_diff);
  *max_diff_out = max_diff;
  return bad;
}

}  // namespace

TEST(AmxPrepack, BitExactQkvSquare) {
  float md; EXPECT_EQ(CountMismatches({128, 2048, 2048, "QKV (Table 3 row)"}, &md), 0);
}

TEST(AmxPrepack, BitExactWithKBlockCarry) {
  // K = 5632 > Kc = 2048 exercises the LDZ/STZ partial-accumulator carry.
  float md; EXPECT_EQ(CountMismatches({128, 2048, 5632, "TinyLlama FFN2 (K>Kc)"}, &md), 0);
}

TEST(AmxPrepack, BitExactLmHeadTail) {
  // N = 60000 = 937*64 + 32: the last 32 columns go through the 16-wide tail.
  // M reduced to 32 so the 60000-wide output stays cheap.
  float md; EXPECT_EQ(CountMismatches({32, 60000, 2048, "GPT-2 LM head (N%64!=0)"}, &md), 0);
}

TEST(AmxPrepack, BitExactScalarRemainder) {
  // N = 100 = 64 + 16 + 16 + 4: panel + two 16-wide tiles + 4 scalar columns.
  float md; EXPECT_EQ(CountMismatches({32, 100, 300, "odd N (scalar tail)"}, &md), 0);
}

TEST(AmxPrepack, RejectsMNotMultipleOf16) {
  std::vector<float> A(20 * 64), B(64 * 64), C(20 * 64);
  auto W = inferc::rt::AmxPackWeight(B.data(), 64, 64);
  EXPECT_THROW(inferc::rt::AmxPrepackedSgemm(A.data(), W, C.data(), 20), std::invalid_argument);
}
