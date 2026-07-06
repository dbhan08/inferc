// Accurate GENLUT-exp, correctness-first. exp(x) ~= a_k + b_k*x via a per-bucket
// linear fit: genlut-generate(x)->bucket k, genlut-lookup(k)->a_k, lookup(k)->b_k,
// then VECFP z = a_k + b_k*x. ~6 ops vs the squaring exp's 32. Verify vs libm exp
// over the attention score range BEFORE any timing (skill: correctness before speed).
// Encodings: generate (mode0 f32->u4), lookup (mode11 u4->f32) from int4_gemv_amx.cc.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "amx/aarch64.h"

static inline uint64_t glut(int mode,int tblreg,bool tbly,bool destz,bool desty,int destreg,bool srcy,int srcoff){
  return (uint64_t(mode)<<53)|(uint64_t(tblreg)<<60)|(uint64_t(tbly)<<59)|(uint64_t(destz)<<26)
       |(uint64_t(desty)<<25)|(uint64_t(destreg)<<20)|(uint64_t(srcy)<<10)|uint64_t(srcoff&0x1FF);
}
static inline uint64_t Vfp(int alu,int z,int xo,int yo){return (uint64_t(alu)<<47)|(4ULL<<42)|(uint64_t(z)<<20)|(uint64_t(xo)<<10)|uint64_t(yo);}

int main(){
  const int N=16; const float lo=-8.f, w=1.f;                 // 16 buckets, width 1 over [-8,8]
  // host: thresholds + per-bucket linear fit a_k + b_k*x (least-squares over the bucket)
  alignas(64) float thr[16], a[16], b[16], xin[16], ones[16], out[16];
  for(int k=0;k<16;++k){ thr[k]=lo+k*w; }                     // thr[k]=lo+k*w -> idx = bucket m
  for(int k=0;k<16;++k){ double x0=lo+k*w, x1=x0+w;           // bucket [x0,x1)
    // linear least-squares fit of exp on [x0,x1]: slope b, intercept a
    int S=8; double sx=0,sy=0,sxx=0,sxy=0; for(int i=0;i<S;++i){double xx=x0+(i+0.5)*w/S,yy=std::exp(xx);sx+=xx;sy+=yy;sxx+=xx*xx;sxy+=xx*yy;}
    double den=S*sxx-sx*sx; b[k]=(float)((S*sxy-sx*sy)/den); a[k]=(float)((sy-b[k]*sx)/S); }
  for(int i=0;i<16;++i){ xin[i]=-7.5f+i*0.95f; ones[i]=1.f; }  // test points spanning the range
  const uint64_t T=(uint64_t)thr,A=(uint64_t)a,B=(uint64_t)b,X=(uint64_t)xin,ON=(uint64_t)ones,O=(uint64_t)out;

  AMX_SET();
  AMX_LDX(X|(0ULL<<56));                                       // X[0] = x
  AMX_LDY(T|(1ULL<<56));                                       // Y[1] = thresholds
  AMX_LDY(A|(2ULL<<56));                                       // Y[2] = a table
  AMX_LDY(B|(3ULL<<56));                                       // Y[3] = b table
  AMX_LDY(ON|(5ULL<<56));                                      // Y[5] = ones
  alignas(64) float z0[16]={0}; const uint64_t Z0=(uint64_t)z0;
  // 1) generate x -> bucket indices in X[2]
  AMX_GENLUT(glut(0,/*tbl*/1,true,false,false,/*dest X*/2,false,/*src X[0]*/0));
  // 2) lookup a_k -> X[3] (src = indices at X[2] = byte 128)
  AMX_GENLUT(glut(11,/*tbl*/2,true,false,false,/*dest X*/3,false,128));
  // 3) lookup b_k -> Y[4] (need b in Y so VECFP can do x*b)
  AMX_GENLUT(glut(11,/*tbl*/3,true,/*destz*/false,/*desty*/true,/*destreg*/4,false,128));
  // 4) Z[0] = a_k : zero Z then += a(X[3]) * ones(Y[5])
  AMX_LDZ(Z0|(0ULL<<56));
  AMX_VECFP(Vfp(0,0,/*X[3]=a*/3*64,/*Y[5]=ones*/5*64));       // Z[0] = a_k
  // 5) Z[0] += x(X[0]) * b(Y[4])  -> a_k + b_k*x = exp(x)
  AMX_VECFP(Vfp(0,0,/*X[0]=x*/0,/*Y[4]=b*/4*64));
  AMX_STZ(O|(0ULL<<56));
  AMX_CLR();
  std::printf("genlut-exp = a_k + b_k*x (16-bucket linear), verifying:\n");
  double maxrel=0; int bad=0;
  for(int i=0;i<16;++i){ float ref=std::exp(xin[i]); double rel=std::fabs(out[i]-ref)/ref; maxrel=std::max(maxrel,rel);
    if(i<8) std::printf("  x=%+.2f  amx=%.4f  ref=%.4f  rel=%.1e\n",xin[i],out[i],ref,rel); if(rel>0.05) ++bad; }
  std::printf("genlut-exp max-rel=%.2e  (%s)\n", maxrel, bad==0?"OK <5%":"needs fix");
  return 0;
}
