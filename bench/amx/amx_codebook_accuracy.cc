// ACCURACY gap-closer. The AMX kernel computes A.dequant(W) EXACTLY (verified), so the
// only accuracy question is the QUANTIZATION quality. The mechanism uniquely enables
// NON-UNIFORM (k-means / NF4) codebooks; this measures whether that actually buys accuracy
// over uniform int4 at the same 4 bits -- and whether the absolute error is usable.
// Per-group (G=64 contiguous weights) 1D k-means 16-centroid codebook vs uniform-int4
// (16 levels min..max). Report weight SQNR (dB) and GEMM output rel-error, on Gaussian
// weights (LLM weights are ~Gaussian -- which is exactly why NF4-style codebooks win).
// Host-only: this is the quant-math, independent of the (already-verified) kernel.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using std::vector;
struct Rng{unsigned long long s;double u(){s=s*6364136223846793005ULL+1442695040888963407ULL;return((s>>11)&((1ULL<<53)-1))/9007199254740992.0;}};
static double nrm(Rng&r){double u1=r.u(),u2=r.u();if(u1<1e-12)u1=1e-12;return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2);}

// 1D Lloyd k-means: 16 centroids for one group; fills code[] (index) and returns centroids in cb[16]
static void kmeans16(const float* w,int n,float* cb,int* code){
  vector<float> s(w,w+n); std::sort(s.begin(),s.end());
  for(int e=0;e<16;++e) cb[e]=s[(size_t)(e+0.5)*n/16];                  // init = quantiles
  for(int it=0;it<15;++it){
    double sum[16]={0}; int cnt[16]={0};
    for(int i=0;i<n;++i){ int be=0;float bd=1e30f;for(int e=0;e<16;++e){float d=std::fabs(cb[e]-w[i]);if(d<bd){bd=d;be=e;}} code[i]=be; sum[be]+=w[i]; cnt[be]++; }
    for(int e=0;e<16;++e) if(cnt[e]) cb[e]=(float)(sum[e]/cnt[e]);
  }
  for(int i=0;i<n;++i){ int be=0;float bd=1e30f;for(int e=0;e<16;++e){float d=std::fabs(cb[e]-w[i]);if(d<bd){bd=d;be=e;}} code[i]=be; }
}

static void measure(const char* label,const vector<float>& W,const vector<float>& A,int K,int N,int M,int G){
  vector<float> Wkm((size_t)K*N), Wun((size_t)K*N);
  double sqW=0,seKM=0,seUN=0;
  vector<float> g(G); vector<float> cb(16); vector<int> code(G);
  for(int n=0;n<N;++n) for(int k0=0;k0<K;k0+=G){
    for(int i=0;i<G;++i) g[i]=W[(size_t)(k0+i)*N+n];
    kmeans16(g.data(),G,cb.data(),code.data());
    for(int i=0;i<G;++i) Wkm[(size_t)(k0+i)*N+n]=cb[code[i]];
    float lo=g[0],hi=g[0]; for(float v:g){lo=std::min(lo,v);hi=std::max(hi,v);} float st=(hi-lo)/15.f+1e-12f;
    for(int i=0;i<G;++i){ int q=(int)std::lround((g[i]-lo)/st); q=std::max(0,std::min(15,q)); Wun[(size_t)(k0+i)*N+n]=lo+q*st; }
    for(int i=0;i<G;++i){ double w=g[i]; sqW+=w*w; double ek=w-Wkm[(size_t)(k0+i)*N+n],eu=w-Wun[(size_t)(k0+i)*N+n]; seKM+=ek*ek; seUN+=eu*eu; }
  }
  auto gemmerr=[&](const vector<float>& Wd){ double num=0,den=0;
    for(int m=0;m<M;++m) for(int n=0;n<N;++n){ double t=0,q=0; for(int k=0;k<K;++k){ t+=(double)A[(size_t)m*K+k]*W[(size_t)k*N+n]; q+=(double)A[(size_t)m*K+k]*Wd[(size_t)k*N+n]; } num+=(t-q)*(t-q); den+=t*t; } return std::sqrt(num/den); };
  std::printf("[%s] 4-bit per-%d-weight groups:\n",label,G);
  std::printf("  k-means      SQNR %.2f dB  wt-err %.4f  GEMM-err %.4f\n", 10*std::log10(sqW/seKM), std::sqrt(seKM/sqW), gemmerr(Wkm));
  std::printf("  uniform int4 SQNR %.2f dB  wt-err %.4f  GEMM-err %.4f\n", 10*std::log10(sqW/seUN), std::sqrt(seUN/sqW), gemmerr(Wun));
  std::printf("  non-uniform advantage: +%.2f dB\n\n", 10*std::log10(sqW/seKM)-10*std::log10(sqW/seUN));
}

int main(){
  const int K=2048,N=512,M=16,G=64;                 // G=64-weight groups (NF4-style)
  Rng r{0xACCULL};
  vector<float> A((size_t)M*K); for(auto&x:A) x=(float)nrm(r);
  vector<float> Wg((size_t)K*N); for(auto&x:Wg) x=(float)nrm(r);                       // clean Gaussian
  vector<float> Wt((size_t)K*N); for(auto&x:Wt){ double z=nrm(r); if(r.u()<0.03) z*=8.0; x=(float)z; } // 3% outliers x8 (LLM-like)
  measure("Gaussian", Wg, A, K,N,M,G);
  measure("heavy-tailed (3% outliers x8)", Wt, A, K,N,M,G);
  std::printf("(kernel computes A.dequant EXACTLY -> GEMM-err IS the kernel output error.)\n");
  return 0;
}

