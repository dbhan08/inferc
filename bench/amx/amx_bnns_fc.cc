// Fairness gate: does Accelerate's BNNS fully-connected filter (which bakes the
// weight in at filter creation -- i.e. PRE-PACKS it) match our pre-packed
// direct-AMX kernel at the N>>K shapes? If BNNS-FC pre-packed >= our pre-packed,
// the pre-pack win narrows to "vs the GEMM APIs (cblas, BNNSMatMul)"; if it
// stays at BNNS's per-call rate, our pre-pack win holds against all of BNNS too.
//
// Map C[M,N] = A[M,K]*B[K,N] to FC: out = W*in, in_size=K, out_size=N,
// weight W = B^T [N,K] (baked in at creation), input = A rows, batch = M.

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main() {
  struct S { int M, N, K; const char* t; };
  const S sh[] = {
    {128, 2048, 2048, "QKV-2048 (K=N ref)"},
    {128, 60000, 2048, "LM-head 60000/2048"},
    {128, 32000, 2048, "LM-head 32000/2048"},
    {128, 32000, 4096, "LM-head 32000/4096"},
    {128, 11008, 4096, "FFN1 11008/4096"},
  };
  std::printf("%-22s %-10s %-18s\n", "shape", "cblas", "BNNS-FC prepacked");
  for (auto& s : sh) {
    const int M = s.M, N = s.N, K = s.K;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), Bt(size_t(N) * K),
        C(size_t(M) * N, 0.f), Cref(size_t(M) * N, 0.f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = float(i % 7) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = float(i % 11) * 0.01f;
    for (int64_t k = 0; k < K; ++k) for (int64_t n = 0; n < N; ++n) Bt[n * K + k] = B[k * N + n];
    const double flops = 2.0 * M * (double)N * K;
    const float* ap = A.data();

    auto accel = [&] { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, Cref.data(), N); };
    accel(); accel();
    double tA = 1e30; for (int i = 0; i < 7; ++i) { auto t0 = clk::now(); accel(); tA = std::min(tA, ms(clk::now() - t0)); }
    double gA = flops / (tA / 1e3) / 1e9;

    // BNNS FC filter: weight W = B^T [N,K] baked in at creation (pre-packed).
    BNNSNDArrayDescriptor idesc; std::memset(&idesc, 0, sizeof(idesc));
    idesc.layout = BNNSDataLayoutVector; idesc.size[0] = (size_t)K; idesc.data_type = BNNSDataTypeFloat32;
    BNNSNDArrayDescriptor wdesc; std::memset(&wdesc, 0, sizeof(wdesc));
    wdesc.layout = BNNSDataLayoutRowMajorMatrix; wdesc.size[0] = (size_t)K; wdesc.size[1] = (size_t)N;
    wdesc.data = Bt.data(); wdesc.data_type = BNNSDataTypeFloat32;
    BNNSNDArrayDescriptor odesc; std::memset(&odesc, 0, sizeof(odesc));
    odesc.layout = BNNSDataLayoutVector; odesc.size[0] = (size_t)N; odesc.data_type = BNNSDataTypeFloat32;
    BNNSNDArrayDescriptor bias; std::memset(&bias, 0, sizeof(bias));
    BNNSActivation act; std::memset(&act, 0, sizeof(act)); act.function = BNNSActivationFunctionIdentity;
    BNNSLayerParametersFullyConnected p; std::memset(&p, 0, sizeof(p));
    p.i_desc = idesc; p.w_desc = wdesc; p.o_desc = odesc; p.bias = bias; p.activation = act;
    BNNSFilter filt = BNNSFilterCreateLayerFullyConnected(&p, nullptr);  // pre-packs W
    if (!filt) { std::printf("%-22s %-10.0f  (FC filter create FAILED)\n", s.t, gA); continue; }
    float* cp = C.data();
    auto fc = [&] { BNNSFilterApplyBatch(filt, (size_t)M, ap, (size_t)K, cp, (size_t)N); };
    fc(); fc();
    double tF = 1e30; for (int i = 0; i < 7; ++i) { auto t0 = clk::now(); fc(); tF = std::min(tF, ms(clk::now() - t0)); }
    double gF = flops / (tF / 1e3) / 1e9;
    float maxd = 0.f; for (size_t i = 0; i < C.size(); i += 1023) maxd = std::max(maxd, std::fabs(cp[i] - Cref.data()[i]));
    BNNSFilterDestroy(filt);
    std::printf("%-22s %-10.0f %4.0f %s (vs cblas %.2fx; ours-prepacked ~1040)\n",
                s.t, gA, gF, maxd < 1e-2f ? "" : "MISMATCH", gF / gA);
  }
  return 0;
}
