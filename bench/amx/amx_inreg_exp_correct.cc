// CORRECTNESS FIRST: does the in-register AMX squaring exp actually compute exp?
// M1 VECFP is accumulate-only (z += x*y; no skip-z, mode-10 overwrite is M2-only),
// so a correct square t->t^2 needs: t in X (EXTRH), t in Y (EXTRH dest-Y), Z zeroed
// (LDZ of zeros), then VECFP z += x*y. This verifies one score row through the full
// crude exp (init t=1+x/2^m, then m squarings) against the C++ reference, lane by
// lane. Only if this matches do we trust the fused-attention timing and build on it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"

static inline uint64_t Vfp(int alu,int z,int xo,int yo){return (uint64_t(alu)<<47)|(4ULL<<42)|(uint64_t(z)<<20)|(uint64_t(xo)<<10)|uint64_t(yo);}
static inline uint64_t ExX(int zrow,int xoff){return (uint64_t(zrow)<<20)|(1ULL<<26)|(uint64_t(xoff)&0x1FF);}            // Z->X
static inline uint64_t ExY(int zrow,int yoff){return (uint64_t(zrow)<<20)|(1ULL<<26)|(1ULL<<10)|(uint64_t(yoff)&0x1FF);} // Z->Y
static float cexp(float x,int m){int n=1<<m;float t=1.f+x/(float)n;for(int i=0;i<m;++i)t=t*t;return t;}

int main(){
  const int M=8; const float scale=0.125f, c=scale/(float)(1<<M);
  float *scores,*cvec,*ones,*zero,*outv;
  posix_memalign((void**)&scores,128,64); posix_memalign((void**)&cvec,128,64);
  posix_memalign((void**)&ones,128,64); posix_memalign((void**)&zero,128,64); posix_memalign((void**)&outv,128,64);
  for(int i=0;i<16;++i){ scores[i]=(float)(i-8)*0.7f; cvec[i]=c; ones[i]=1.f; zero[i]=0.f; }
  const uint64_t sc=(uint64_t)scores,cv=(uint64_t)cvec,on=(uint64_t)ones,ze=(uint64_t)zero,ov=(uint64_t)outv;

  AMX_SET();
  AMX_LDX(sc|(0ULL<<56));                       // X[0] = scores
  AMX_LDY(cv|(0ULL<<56));                        // Y[0] = c
  AMX_LDZ(on|(0ULL<<56));                        // Z[0] = 1.0
  AMX_VECFP(Vfp(0,0,0,0));                       // Z[0] = 1 + scores*c = t0
  for(int q=0;q<M;++q){
    AMX_EXTRX(ExX(0,0));                         // t -> X[0]
    AMX_EXTRX(ExY(0,0));                         // t -> Y[0]
    AMX_LDZ(ze|(0ULL<<56));                      // Z[0] = 0
    AMX_VECFP(Vfp(0,0,0,0));                     // Z[0] = 0 + t*t = t^2
  }
  AMX_STZ(ov|(0ULL<<56));                        // out = exp(scores)
  AMX_CLR();

  int bad=0; double maxrel=0;
  for(int i=0;i<16;++i){ float ref=cexp(scores[i]*scale,M);
    double rel=std::fabs(outv[i]-ref)/std::fabs(ref); maxrel=std::max(maxrel,rel);
    if(rel>1e-3){ ++bad; if(bad<=4) std::printf("  lane %2d: amx=%.5f ref=%.5f rel=%.2e\n",i,outv[i],ref,rel); } }
  std::printf("in-register squaring exp (m=%d): max-rel=%.2e  %s\n", M, maxrel,
              bad==0 ? "CORRECT -- matches crude_exp reference" : "MISMATCH -- encoding bug, do NOT proceed");
  free(scores);free(cvec);free(ones);free(zero);free(outv); return bad==0?0:1;
}
