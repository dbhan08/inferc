// GATE for the fused AMX flash-attention: M1 AMX can't do an accurate exp in-register
// (no VECINT left-shift for 2^k, no clean float<->int convert). The only float-only
// option is a crude exp -- exp(x) ~= (1 + x/2^m)^(2^m) via m VECFP squarings. This
// tests, in C++, whether that crude exp degrades ATTENTION OUTPUT vs a true-softmax
// fp64 reference, across realistic score distributions and m (= squaring count = cost).
// If output rms-rel stays small (<~1%) at a cheap m, the fused kernel is viable; if
// not, the missing exp is the wall and in-register softmax is foreclosed.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct Rng { uint64_t s; double u(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return ((s>>11)&((1ULL<<53)-1))/9007199254740992.0; } };
static double nrm(Rng& r){ double u1=r.u(),u2=r.u(); if(u1<1e-12)u1=1e-12; return std::sqrt(-2*std::log(u1))*std::cos(6.283185307179586*u2); }
// crude exp via m squarings of (1 + x/2^m), float32 throughout (what VECFP can do)
static float crude_exp(float x, int m){ int n=1<<m; float t=1.f + x/(float)n; for(int i=0;i<m;++i) t=t*t; return t; }

int main() {
  const int S = 128, d = 64;
  const int Ms[] = {6, 8, 10, 12};
  Rng r{0xABCDEF1234567ULL};
  std::vector<float> Q(S*d), K(S*d), V(S*d);
  for (auto& x : Q) x = (float)nrm(r); for (auto& x : K) x = (float)nrm(r); for (auto& x : V) x = (float)nrm(r);
  const float scale = 1.f / std::sqrt((float)d);

  // true-softmax attention reference (fp64), output[S,d]
  std::vector<double> ref(S*d, 0.0);
  for (int i = 0; i < S; ++i) {
    std::vector<double> p(S); double mx=-1e30;
    for (int j=0;j<S;++j){ double s=0; for(int t=0;t<d;++t) s+=(double)Q[i*d+t]*K[j*d+t]; p[j]=s*scale; mx=std::max(mx,p[j]); }
    double sum=0; for(int j=0;j<S;++j){ p[j]=std::exp(p[j]-mx); sum+=p[j]; }
    for(int j=0;j<S;++j){ double w=p[j]/sum; for(int t=0;t<d;++t) ref[i*d+t]+=w*V[j*d+t]; }
  }

  std::printf("score range check: typical |score| after 1/sqrt(d) ~ O(1-3)\n");
  std::printf("%-10s %-16s %s\n", "m (sqr)", "exp max-rel-err", "ATTENTION out rms-rel vs fp64");
  for (int m : Ms) {
    // exp accuracy over the score range
    double emax=0; for(double x=-6; x<=6; x+=0.25){ double e=std::exp(x), c=crude_exp((float)x,m); emax=std::max(emax,std::fabs(c-e)/e); }
    // attention with crude exp (skip max-subtraction; scores bounded)
    double sse=0, sref=0;
    for (int i=0;i<S;++i){
      std::vector<float> p(S); float sum=0;
      for(int j=0;j<S;++j){ float s=0; for(int t=0;t<d;++t) s+=Q[i*d+t]*K[j*d+t]; p[j]=crude_exp(s*scale,m); sum+=p[j]; }
      for(int t=0;t<d;++t){ double o=0; for(int j=0;j<S;++j) o+=(double)(p[j]/sum)*V[j*d+t]; double e=o-ref[i*d+t]; sse+=e*e; sref+=ref[i*d+t]*ref[i*d+t]; }
    }
    std::printf("%-10d %-16.2e %.2e\n", m, emax, std::sqrt(sse/sref));
  }
  std::printf("\nIf attention rms-rel < ~1%% at m=8-10, the crude VECFP-squaring exp is\n"
              "viable -> build the fused kernel. If it stays high, in-register softmax on\n"
              "M1 AMX is foreclosed by the missing accurate-exp primitives.\n");
  return 0;
}
