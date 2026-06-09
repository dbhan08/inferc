// Batch-size sensitivity of the per-call kernel (re-pack each call, single
// thread) at the QKV shape (N=K=2048), vs single-thread cblas_sgemm. CSV out;
// run multiple times and take medians for the 11-invocation-style protocol.
#include <Accelerate/Accelerate.h>
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
static inline uint64_t F(int z,int xo,bool f){return (uint64_t(z)<<20)|(uint64_t(xo)<<10)|(f?(1ULL<<27):0);}
static const uint64_t PAIR=1ULL<<62;
static void compute_kc(const float*At,const float*pB,float*C,int64_t M,int64_t N,int64_t K,int Kc,int64_t jc,int64_t Ncm){
  for(int64_t pc=0;pc<K;pc+=Kc){int64_t ke=std::min<int64_t>(Kc,K-pc);bool f0=(pc==0);
    for(int64_t i0=0;i0<M;i0+=16)for(int64_t jr=0;jr<Ncm;jr+=64){
      if(!f0)for(int t=0;t<4;++t)for(int j=0;j<16;++j)AMX_LDZ(reinterpret_cast<uint64_t>(C+(i0+j)*N+jc+jr+16*t)|(uint64_t(4*j+t)<<56));
      for(int64_t kk=0;kk<ke;++kk){bool f=(f0&&kk==0);AMX_LDY(reinterpret_cast<uint64_t>(&At[(pc+kk)*M+i0]));
        const float*br=pB+(pc+kk)*Ncm+jr;AMX_LDX(reinterpret_cast<uint64_t>(br)|(0ULL<<56)|PAIR);AMX_LDX(reinterpret_cast<uint64_t>(br+32)|(2ULL<<56)|PAIR);
        AMX_FMA32(F(0,0,f));AMX_FMA32(F(1,64,f));AMX_FMA32(F(2,128,f));AMX_FMA32(F(3,192,f));}
      for(int t=0;t<4;++t)for(int j=0;j<16;++j)AMX_STZ(reinterpret_cast<uint64_t>(C+(i0+j)*N+jc+jr+16*t)|(uint64_t(4*j+t)<<56));}}
}
// per-call single-thread packed sgemm (transpose A + pack B + compute), QKV: N=K, Nc=Kc=512.
static double percall(const float*A,const float*B,float*C,int M,int N,int K,std::vector<float>&At,std::vector<float>&pB){
  const int Nc=512,Kc=512;
  for(int64_t i=0;i<M;++i)for(int64_t k=0;k<K;++k)At[k*M+i]=A[i*K+k];
  AMX_SET();
  for(int64_t jc=0;jc<N;jc+=Nc){int64_t Ncm=(std::min<int64_t>(Nc,N-jc)/64)*64; if(Ncm<=0)continue;
    for(int64_t k=0;k<K;++k)std::memcpy(&pB[k*Ncm],&B[k*N+jc],Ncm*sizeof(float));
    compute_kc(At.data(),pB.data(),C,M,N,K,Kc,jc,Ncm);}
  AMX_CLR();
  return 0;
}
int main(){
  const int Ss[]={16,32,64,128,256,512}; const int H=2048;
  for(int S:Ss){
    const int M=S,N=H,K=H; double flops=2.0*M*(double)N*K;
    std::vector<float> A(size_t(M)*K),B(size_t(K)*N),C(size_t(M)*N,0.f),Cr(size_t(M)*N,0.f);
    for(size_t i=0;i<A.size();++i)A[i]=float(i%7)*0.01f;
    for(size_t i=0;i<B.size();++i)B[i]=float(i%11)*0.01f;
    const float*ap=A.data(),*bp=B.data(); float*cp=C.data(),*crp=Cr.data();
    auto cb=[&]{cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1.0f,ap,K,bp,N,0.0f,crp,N);};
    cb();cb(); double tC=1e30; for(int i=0;i<9;++i){auto t0=clk::now();cb();tC=std::min(tC,ms(clk::now()-t0));}
    std::vector<float> At(size_t(K)*M),pB(size_t(K)*512);
    auto pc=[&]{percall(ap,bp,cp,M,N,K,At,pB);};
    pc();pc(); double tP=1e30; for(int i=0;i<9;++i){auto t0=clk::now();pc();tP=std::min(tP,ms(clk::now()-t0));}
    double gC=flops/(tC/1e3)/1e9, gP=flops/(tP/1e3)/1e9;
    float md=0; for(size_t i=0;i<C.size();i+=997) md=std::max(md,std::fabs(cp[i]-crp[i]));
    std::printf("S=%-4d cblas %5.0f  proposed %5.0f  %.2f %s\n",S,gC,gP,gP/gC,md<1e-2f?"":"BUG");
    std::printf("CSV,%d,%.1f,%.1f\n",S,gC,gP);
  }
  return 0;
}
