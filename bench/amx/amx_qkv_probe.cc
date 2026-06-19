// Why does our pre-packed kernel cap ~840 GFLOPS at QKV on Tahoe while BNNS
// Graph hits 1116 bit-exact? Hypothesis: the deployed kernel parallelizes over
// N-panels ONLY (nP = N/Nc); at QKV (N=2048, Nc=512) that is just 4 panels -> 4
// concurrent work items -> P-cluster only, the E-cluster AMX block idle.
//
// This probes two levers on the EXACT compute_kc microkernel:
//   (1) Nc sweep         -> more N-panels = more work items (finer parallelism)
//   (2) 2D (M x N) tiling -> split the 128 M-rows into 16-row blocks too, so
//        work items = (M/Mb) * (N/Nc); fills 8 cores without shrinking Nc
// dispatch_apply on the default concurrent queue lets GCD place work on BOTH
// clusters when there are enough items. Bit-exact checked vs cblas.
//
//   clang++ -O3 -std=c++17 -Ithird_party -framework Accelerate amx_qkv_probe.cc
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
// compute_kc over an M sub-range [m0,m0+Mb): packed [K x Ncm] panel, no packing.
static void compute_kc_m(const float*At,const float*pB,float*C,int64_t M,int64_t N,int64_t K,
                         int Kc,int64_t jc,int64_t Ncm,int64_t m0,int64_t Mb){
  const uint64_t LDX_PAIR=1ULL<<62;
  for(int64_t pc=0;pc<K;pc+=Kc){
    int64_t Kc_eff=std::min<int64_t>(Kc,K-pc); const bool first=(pc==0);
    for(int64_t i0=m0;i0<m0+Mb;i0+=16)
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
  struct S{int M,N,K;const char*t;};
  const S sh[]={{128,2048,2048,"QKV-2048"},{128,4096,4096,"QKV-4096"},{128,2048,8192,"FFN2-K8192"}};
  for(auto&s:sh){
    const int M=s.M,N=s.N,K=s.K; const int Kc=512;
    std::vector<float>A(size_t(M)*K),B(size_t(K)*N),C(size_t(M)*N,0.f),Cref(size_t(M)*N,0.f);
    for(size_t i=0;i<A.size();++i)A[i]=float(i%7)*0.01f;
    for(size_t i=0;i<B.size();++i)B[i]=float(i%11)*0.01f;
    std::vector<float>At(size_t(K)*M);
    for(int64_t i=0;i<M;++i)for(int64_t k=0;k<K;++k)At[k*M+i]=A[i*K+k];
    const double flops=2.0*M*(double)N*K;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,A.data(),K,B.data(),N,0.f,Cref.data(),N);
    const float*atp=At.data(); float*cp=C.data();
    auto best=[](int n,void(^f)()){f();f();double b=1e30;for(int i=0;i<n;++i){auto t0=clk::now();f();b=std::min(b,ms(clk::now()-t0));}return b;};
    auto check=[&](){float md=0;for(size_t i=0;i<C.size();i+=997)md=std::max(md,std::fabs(cp[i]-Cref[i]));return md;};

    std::printf("\n== %s (M=%d N=%d K=%d) ==  flops=%.2e\n",s.t,M,N,K,flops);
    // (1) 1D N-panel parallelism, Nc sweep
    for(int Nc: {512,256,128,64}){
      if(N%64)break;
      std::vector<int64_t>jcs; for(int64_t jc=0;jc<N;jc+=Nc)jcs.push_back(jc);
      const int nP=(int)jcs.size();
      std::vector<std::vector<float>>pk(nP);
      for(int p=0;p<nP;++p){int64_t Ncm=std::min<int64_t>(Nc,N-jcs[p]);pk[p].resize(size_t(K)*Ncm);pack_full(B.data(),pk[p].data(),N,K,jcs[p],Ncm);}
      auto*jcp=jcs.data();auto*pkp=&pk;
      auto run=^{dispatch_apply(nP,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t w){
        int64_t Ncm=std::min<int64_t>(Nc,N-jcp[w]);AMX_SET();compute_kc_m(atp,(*pkp)[w].data(),cp,M,N,K,Kc,jcp[w],Ncm,0,M);AMX_CLR();});};
      double t=best(7,run);std::printf("  1D Nc=%-4d panels=%-3d  %6.0f GFLOPS  diff %.1e\n",Nc,nP,flops/(t/1e3)/1e9,check());
    }
    // (2) 2D (M-block x N-panel) parallelism: more items, keep Nc=256
    for(int Nc: {512,256}){
      for(int Mb: {64,32,16}){
        std::vector<int64_t>jcs; for(int64_t jc=0;jc<N;jc+=Nc)jcs.push_back(jc);
        const int nP=(int)jcs.size(), nM=(M+Mb-1)/Mb, nW=nP*nM;
        std::vector<std::vector<float>>pk(nP);
        for(int p=0;p<nP;++p){int64_t Ncm=std::min<int64_t>(Nc,N-jcs[p]);pk[p].resize(size_t(K)*Ncm);pack_full(B.data(),pk[p].data(),N,K,jcs[p],Ncm);}
        auto*jcp=jcs.data();auto*pkp=&pk;
        auto run=^{dispatch_apply(nW,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t w){
          int p=(int)(w%nP),mb=(int)(w/nP);int64_t m0=mb*Mb,Mbe=std::min<int64_t>(Mb,M-m0);
          int64_t Ncm=std::min<int64_t>(Nc,N-jcp[p]);AMX_SET();compute_kc_m(atp,(*pkp)[p].data(),cp,M,N,K,Kc,jcp[p],Ncm,m0,Mbe);AMX_CLR();});};
        double t=best(7,run);std::printf("  2D Nc=%-4d Mb=%-3d items=%-3d  %6.0f GFLOPS  diff %.1e\n",Nc,Mb,nW,flops/(t/1e3)/1e9,check());
      }
    }
  }
}
