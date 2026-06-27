// BUILD (needs ggml from llama.cpp):
//   git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
//   cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
//   cmake --build /tmp/llama.cpp/build --target ggml ggml-cpu -j8
//   clang++ -std=c++17 -O3 -I/tmp/llama.cpp/ggml/include bench/amx/ggml_q4_bench.cpp \
//     -L/tmp/llama.cpp/build/bin -lggml -lggml-cpu -lggml-base -Wl,-rpath,/tmp/llama.cpp/build/bin -o /tmp/ggq4
// ggml Q4_0 matmul (the production M1 4-bit baseline: NEON DOTPROD, no AMX/Accelerate)
// at our shapes, M-sweep, single- and multi-thread. To compare vs our AMX codebook kernel.
// ggml_mul_mat(a,b): a=[K,N] Q4_0 weight, b=[K,M] f32 act -> [N,M]. Same FLOPs as our GEMM.
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}

static double run(int M,int K,int N,int nth){
  size_t mem = (size_t)512<<20;
  ggml_init_params p{mem, nullptr, false};
  ggml_context* ctx = ggml_init(p);
  // weight [K,N] f32 -> Q4_0
  std::vector<float> Wf((size_t)K*N); for(auto&x:Wf) x=(float)(rand()/(double)RAND_MAX-0.5);
  ggml_tensor* Wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
  ggml_quantize_chunk(GGML_TYPE_Q4_0, Wf.data(), Wq->data, 0, N, K, nullptr);
  ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
  float* a=(float*)A->data; for(size_t i=0;i<(size_t)K*M;++i) a[i]=(float)(rand()/(double)RAND_MAX-0.5);
  ggml_tensor* C = ggml_mul_mat(ctx, Wq, A);              // [N, M]
  ggml_cgraph* g = ggml_new_graph(ctx); ggml_build_forward_expand(g, C);
  for(int i=0;i<3;++i) ggml_graph_compute_with_ctx(ctx, g, nth);   // warm
  double best=1e30;
  for(int i=0;i<6;++i){ auto t0=clk::now(); ggml_graph_compute_with_ctx(ctx, g, nth); best=std::min(best,ms(clk::now()-t0)); }
  ggml_free(ctx);
  return best;
}

int main(){
  const int K=2048,N=8192;
  std::printf("ggml Q4_0 matmul (NEON DOTPROD), K=%d N=%d, our AMX kernel for reference:\n",K,N);
  std::printf("%-5s %-16s %-16s %s\n","M","ggml 1-thread ms","ggml 8-thread ms","[our AMX: M16 ST 1.66 / MT 0.93]");
  for(int M:{1,4,16,64,128}){
    double t1=run(M,K,N,1), t8=run(M,K,N,8);
    std::printf("%-5d %-16.2f %-16.2f\n",M,t1,t8);
  }
  return 0;
}
