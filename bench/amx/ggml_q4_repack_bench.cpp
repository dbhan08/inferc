// FAIR ggml Q4 baseline: weight in the CPU_REPACK buffer so ggml uses its OPTIMIZED
// q4_0_4x4 blocked-DOTPROD GEMM (its best M1 path), not the slow per-row vec_dot.
// Was: my earlier bench used plain CPU buffer -> per-row path (linear in M = strawman).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}

static double run(int M,int K,int N,int nth,bool* repacked){
  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, nth);
  ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  ggml_backend_reg_t reg = ggml_backend_cpu_reg();
  typedef ggml_backend_buffer_type_t* (*getb_t)(ggml_backend_dev_t);
  getb_t getb = (getb_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
  ggml_backend_buffer_type_t repack=nullptr;
  if(getb){ auto* b=getb(dev); for(int i=0;b&&b[i];++i) if(std::string(ggml_backend_buft_name(b[i]))=="CPU_REPACK") repack=b[i]; }
  *repacked = (repack!=nullptr);
  ggml_backend_buffer_type_t wbuft = repack? repack : ggml_backend_cpu_buffer_type();

  // weight ctx (no_alloc) -> alloc in repack buft -> set Q4_0 bytes (repacks)
  ggml_init_params wp{ ggml_tensor_overhead()*2, nullptr, true };
  ggml_context* cw = ggml_init(wp);
  ggml_tensor* Wq = ggml_new_tensor_2d(cw, GGML_TYPE_Q4_0, K, N);
  ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
  std::vector<float> Wf((size_t)K*N); for(auto&x:Wf) x=(float)(rand()/(double)RAND_MAX-0.5);
  size_t qsz = ggml_row_size(GGML_TYPE_Q4_0, K)*N;
  std::vector<uint8_t> q4(qsz); ggml_quantize_chunk(GGML_TYPE_Q4_0, Wf.data(), q4.data(), 0, N, K, nullptr);
  ggml_backend_tensor_set(Wq, q4.data(), 0, qsz);                 // <- repacks into CPU_REPACK layout

  // compute ctx
  ggml_init_params cp{ ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
  ggml_context* cc = ggml_init(cp);
  ggml_tensor* A = ggml_new_tensor_2d(cc, GGML_TYPE_F32, K, M);
  ggml_tensor* C = ggml_mul_mat(cc, Wq, A);
  ggml_cgraph* g = ggml_new_graph(cc); ggml_build_forward_expand(g, C);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  ggml_gallocr_alloc_graph(ga, g);
  std::vector<float> Af((size_t)K*M); for(auto&x:Af) x=(float)(rand()/(double)RAND_MAX-0.5);
  ggml_backend_tensor_set(A, Af.data(), 0, Af.size()*4);

  for(int i=0;i<4;++i) ggml_backend_graph_compute(backend, g);
  double best=1e30;
  for(int i=0;i<8;++i){ auto t0=clk::now(); ggml_backend_graph_compute(backend, g); best=std::min(best,ms(clk::now()-t0)); }
  ggml_gallocr_free(ga); ggml_free(cc); ggml_backend_buffer_free(bw); ggml_free(cw); ggml_backend_free(backend);
  return best;
}

int main(){
  const int K=2048,N=8192;
  bool rp=false; run(16,K,N,1,&rp);
  std::printf("ggml Q4_0 FAIR (repacked=%s), K=%d N=%d, vs our AMX kernel:\n", rp?"YES (q4_0_4x4)":"NO!", K,N);
  std::printf("%-5s %-16s %-16s %s\n","M","ggml-repack 1T ms","ggml-repack 8T ms","[AMX: M16 ST1.56 MT0.93]");
  for(int M:{1,4,16,64,128}){ double t1=run(M,K,N,1,&rp), t8=run(M,K,N,8,&rp); std::printf("%-5d %-16.2f %-16.2f\n",M,t1,t8); }
  return 0;
}
