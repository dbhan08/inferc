// fp16 ceiling probe: does the AMX fp16 path clear the fp32 ~900 GFLOPS ceiling
// at the N-very-large shapes (LM-head, FFN1-large) we can't beat in fp32?
// Compares Accelerate fp32 sgemm vs Accelerate/BNNS fp16 matmul. fp16 here uses
// fp16 accumulate (throughput ceiling only); a real LM-head kernel would use
// fp16 inputs + fp32 accumulate for accuracy.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

static BNNSNDArrayDescriptor desc(void* data, BNNSDataType dt, int64_t rows, int64_t cols) {
  BNNSNDArrayDescriptor d; std::memset(&d, 0, sizeof(d));
  d.layout = BNNSDataLayoutRowMajorMatrix;
  d.size[0] = (size_t)cols; d.size[1] = (size_t)rows;
  d.data = data; d.data_type = dt;
  return d;
}

int main() {
  struct S { int M, N, K; const char* t; };
  const S sh[] = {
    {128, 2048, 2048, "QKV-2048 (K=N ref)"},
    {128, 4096, 4096, "QKV-4096 (K=N ref)"},
    {128, 60000, 2048, "LM-head 60000/2048"},
    {128, 32000, 2048, "LM-head 32000/2048"},
    {128, 32000, 4096, "LM-head 32000/4096"},
    {128, 11008, 4096, "FFN1 11008/4096 (N>K)"},
  };
  std::printf("%-24s %-12s %-12s %-8s\n", "shape", "fp32 GFLOPS", "fp16 GFLOPS", "f16/f32");
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    const double flops = 2.0 * M * (double)N * K;

    std::vector<float> A(size_t(M) * K, 0.5f), B(size_t(K) * N, 0.5f), C(size_t(M) * N, 0.f);
    auto sgemm = [&] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                                   1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N); };
    sgemm(); sgemm();
    double t32 = 1e30;
    for (int i = 0; i < 7; ++i) { auto t0 = clk::now(); sgemm(); t32 = std::min(t32, ms(clk::now() - t0)); }
    double g32 = flops / (t32 / 1e3) / 1e9;

    std::vector<__fp16> Ah(size_t(M) * K, (__fp16)0.5f), Bh(size_t(K) * N, (__fp16)0.5f),
        Ch(size_t(M) * N, (__fp16)0.f);
    auto da = desc(Ah.data(), BNNSDataTypeFloat16, M, K);
    auto db = desc(Bh.data(), BNNSDataTypeFloat16, K, N);
    auto dc = desc(Ch.data(), BNNSDataTypeFloat16, M, N);
    size_t wsz = BNNSMatMulWorkspaceSize(false, false, 1.0f, &da, &db, &dc, nullptr);
    std::vector<char> ws(wsz ? wsz : 1);
    auto mm16 = [&] { BNNSMatMul(false, false, 1.0f, &da, &db, &dc, wsz ? ws.data() : nullptr, nullptr); };
    mm16(); mm16();
    double t16 = 1e30;
    for (int i = 0; i < 7; ++i) { auto t0 = clk::now(); mm16(); t16 = std::min(t16, ms(clk::now() - t0)); }
    double g16 = flops / (t16 / 1e3) / 1e9;

    std::printf("%-24s %-12.0f %-12.0f %.2fx%s\n", s.t, g32, g16, g16 / g32,
                g16 > 950 ? "  <- clears fp32 ceiling" : "");
  }
  return 0;
}
