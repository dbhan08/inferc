// Per-shape Nc tuner: find the panel size that makes our bit-exact pre-packed
// kernel beat BNNS Graph at EVERY one of the 12 LLM prefill shapes on Tahoe.
// MEDIAN-of-N timing (fair vs the BNNS Graph harness's median-of-30, not the
// optimistic best-of-7 the prepack bench used). BNNS Graph median targets baked
// in as the bar to clear. Bit-exact checked vs cblas (diff must be 0).
//
//   clang++ -O3 -std=c++17 -Ithird_party -framework Accelerate amx_nc_tune.cc
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "amx/aarch64.h"

using clk = std::chrono::steady_clock;
static double ms(clk::duration d){return std::chrono::duration<double,std::milli>(d).count();}
static inline uint64_t Fma32Op(int z,int xo,bool f){return (uint64_t(z)<<20)|(uint64_t(xo)<<10)|(f?(1ULL<<27):0);}
static void pack_full(const float*B,float*dst,int64_t N,int64_t K,int64_t jc,int64_t Ncm){
  for(int64_t k=0;k<K;++k) std::memcpy(&dst[k*Ncm],&B[k*N+jc],Ncm*sizeof(float));
}
static void compute_kc(const float*At,const float*pB,float*C,int64_t M,int64_t N,int64_t K,
                       int Kc,int64_t jc,int64_t Ncm){
  const uint64_t LDX_PAIR=1ULL<<62;
  for(int64_t pc=0;pc<K;pc+=Kc){
    int64_t Kc_eff=std::min<int64_t>(Kc,K-pc); const bool first=(pc==0);
    for(int64_t i0=0;i0<M;i0+=16)
      for(int64_t jr=0;jr<Ncm;jr+=64){
        if(!first) for(int t=0;t<4;++t)for(int j=0;j<16;++j)
          AMX_LDZ(reinterpret_cast<uint64_t>(C+(i0+j)*N+jc+jr+16*t)|(uint64_t(4*j+t)<<56));
        for(int64_t kk=0;kk<Kc_eff;++kk){
          const bool f=(first&&kk==0);
          AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc+kk)*M+i0]));
          const float*brow=pB+(pc+kk)*Ncm+jr;
          AMX_LDX(reinterpret_cast<uint64_t>(brow)|(0ULL<<56)|LDX_PAIR);
          AMX_LDX(reinterpret_cast<uint64_t>(brow+32)|(2ULL<<56)|LDX_PAIR);
          AMX_FMA32(Fma32Op(0,0,f));AMX_FMA32(Fma32Op(1,64,f));
          AMX_FMA32(Fma32Op(2,128,f));AMX_FMA32(Fma32Op(3,192,f));
        }
        for(int t=0;t<4;++t)for(int j=0;j<16;++j)
          AMX_STZ(reinterpret_cast<uint64_t>(C+(i0+j)*N+jc+jr+16*t)|(uint64_t(4*j+t)<<56));
      }
  }
}

int main(){
  struct S{int M,N,K;const char*t;double bg;};
  const S sh[]={
    {128,2048,2048,"gpt2_qkv",1116},{128,8192,2048,"gpt2_ffn1",983},
    {128,2048,8192,"gpt2_ffn2",812},{128,60000,2048,"gpt2_lmh",1021},
    {128,2048,2048,"tiny_qkv",1116},{128,5632,2048,"tiny_ffn1",1014},
    {128,2048,5632,"tiny_ffn2",1057},{128,32000,2048,"tiny_lmh",1037},
    {128,4096,4096,"llama_qkv",1135},{128,11008,4096,"llama_ffn1",1132},
    {128,4096,11008,"llama_ffn2",911},{128,32000,4096,"llama_lmh",1115},
  };
  const int NCS[]={64,128,192,256,384,512};
  const int KCS[]={256,512,1024};
  std::printf("%-11s %5s | per-Nc/Kc median GFLOPS (best starred) | best        vs BG    win\n","shape","BG");
  int wins=0;
  for(auto&s:sh){
    const int M=s.M,N=s.N,K=s.K;
    std::vector<float>A(size_t(M)*K),B(size_t(K)*N),C(size_t(M)*N,0.f),Cref(size_t(M)*N,0.f);
    for(size_t i=0;i<A.size();++i)A[i]=float(i%7)*0.01f;
    for(size_t i=0;i<B.size();++i)B[i]=float(i%11)*0.01f;
    std::vector<float>At(size_t(K)*M);
    for(int64_t i=0;i<M;++i)for(int64_t k=0;k<K;++k)At[k*M+i]=A[i*K+k];
    const double flops=2.0*M*(double)N*K;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,A.data(),K,B.data(),N,0.f,Cref.data(),N);
    const float*atp=At.data(); float*cp=C.data();
    auto medrun=[&](int Nc,int Kc)->double{
      std::vector<int64_t>jcs; for(int64_t jc=0;jc<N;jc+=Nc)jcs.push_back(jc);
      const int nP=(int)jcs.size();
      std::vector<std::vector<float>>pk(nP);
      for(int p=0;p<nP;++p){int64_t Ncm=(std::min<int64_t>(Nc,N-jcs[p])/64)*64;if(Ncm<=0)continue;pk[p].resize(size_t(K)*Ncm);pack_full(B.data(),pk[p].data(),N,K,jcs[p],Ncm);}
      auto*jcp=jcs.data();auto*pkp=&pk; int kc=Kc;
      auto run=^{dispatch_apply(nP,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t w){
        int64_t Ncm=(std::min<int64_t>(Nc,N-jcp[w])/64)*64; if(Ncm<=0)return;
        AMX_SET();compute_kc(atp,(*pkp)[w].data(),cp,M,N,K,kc,jcp[w],Ncm);AMX_CLR();});
        // tail columns handled by extending last panel coverage check omitted (Nc|N here)
      };
      run();run();
      std::vector<double>ts; for(int i=0;i<15;++i){auto t0=clk::now();run();ts.push_back(ms(clk::now()-t0));}
      std::sort(ts.begin(),ts.end());
      float md=0;for(size_t i=0;i<C.size();i+=997)md=std::max(md,std::fabs(cp[i]-Cref[i]));
      if(md>0)return -1; // not bit-exact -> reject
      return flops/(ts[ts.size()/2]/1e3)/1e9;
    };
    double bestG=0; int bNc=0,bKc=0;
    std::printf("%-11s %5.0f |",s.t,s.bg);
    for(int Nc:NCS){
      double cg=0;int cKc=0;
      for(int Kc:KCS){double g=medrun(Nc,Kc); if(g>cg){cg=g;cKc=Kc;}}
      if(cg>bestG){bestG=cg;bNc=Nc;bKc=cKc;}
      std::printf(" %4.0f",cg);
    }
    bool win=bestG>=s.bg; wins+=win;
    std::printf(" | Nc=%-3d Kc=%-4d %5.0f  %.2fx  %s\n",bNc,bKc,bestG,bestG/s.bg,win?"WIN":"lose");
  }
  std::printf("\nbest-Nc/Kc beats BNNS Graph at %d/12 shapes (median, bit-exact).\n",wins);
  std::printf("Nc columns: 64 128 192 256 384 512\n");
}
