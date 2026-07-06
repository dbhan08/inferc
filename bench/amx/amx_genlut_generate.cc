// CRUX test for the GENLUT-exp lever: does genlut GENERATE (mode 0, f32->u4) quantize
// a fp32 value into an index by searching a threshold table? This is the float->int
// step that blocked the accurate exp (no VECINT left-shift / no float<->int convert).
// If generate works, exp = generate(x/ln2)->k, lookup(k)->2^k, + short poly = ~6 ops
// (vs 32 for squaring) and fused attention's softmax gets ~5x cheaper. Verify indices
// + measure cost. genlut enc: (mode<<53)|(tblreg<<60)|(tbly<<59)|(destz<<26)|(desty<<25)
// |(destreg<<20)|(srcy<<10)|srcoff. Generate modes 0-6; mode 0 = f32->u4 (16 buckets).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"
using clk = std::chrono::steady_clock;
static double sec(clk::duration d){return std::chrono::duration<double>(d).count();}
static inline uint64_t glut(int mode,int tblreg,bool tbly,bool destz,bool desty,int destreg,bool srcy,int srcoff){
  return (uint64_t(mode)<<53)|(uint64_t(tblreg)<<60)|(uint64_t(tbly)<<59)|(uint64_t(destz)<<26)
       |(uint64_t(desty)<<25)|(uint64_t(destreg)<<20)|(uint64_t(srcy)<<10)|uint64_t(srcoff&0x1FF);
}

int main(){
  float *thr,*src; void* zout;
  posix_memalign((void**)&thr,128,64); posix_memalign((void**)&src,128,64); posix_memalign(&zout,128,64);
  for(int i=0;i<16;++i) thr[i]=(float)(i+1);          // thresholds 1,2,...,16
  for(int i=0;i<16;++i) src[i]=(float)i + 0.5f;        // values 0.5,1.5,...,15.5 -> expect idx 0,1,...,15
  std::memset(zout,0,64);
  const uint64_t T=(uint64_t)thr, S=(uint64_t)src, Z=(uint64_t)zout;

  AMX_SET();
  AMX_LDY(T|(0ULL<<56));                               // Y[0] = thresholds (table)
  AMX_LDX(S|(0ULL<<56));                               // X[0] = source values
  // try table in X instead: generate searches the table; corsix says table reg selectable.
  // dest -> Z[0]. Dump raw to find where/how indices land.
  AMX_GENLUT(glut(0,/*tblreg*/0,/*tbly*/true,/*destz*/false,/*desty*/false,/*destreg*/4,/*srcy*/false,/*srcoff*/0));
  AMX_STX(Z|(4ULL<<56));                               // store X[4] (generate dest) to mem
  AMX_CLR();
  std::printf("raw dest bytes 0-15: ");
  for(int i=0;i<16;++i) std::printf("%02x ", ((const uint8_t*)zout)[i]);
  std::printf("\n");

  // read packed u4 indices from the first 8 bytes of Z[0]
  std::printf("genlut GENERATE (f32 value -> u4 index, thresholds=1..16):\n");
  const uint8_t* b=(const uint8_t*)zout; bool ok=true;
  for(int i=0;i<16;++i){ int idx = (b[i/2]>>((i&1)*4))&0xF; int exp_idx=i;
    if(i<8) std::printf("  src=%.1f -> idx=%d (expect %d)%s\n", src[i], idx, exp_idx, idx==exp_idx?"":"  <-- MISMATCH");
    if(idx!=exp_idx) ok=false; }
  std::printf("generate quantization: %s\n", ok?"CORRECT (fp32->index works -> GENLUT-exp viable)":"needs encoding fix / different packing");

  // cost: generate throughput
  const int64_t IT=20'000'000; const double GHZ=3.2;
  AMX_SET(); AMX_LDY(T|(0ULL<<56)); AMX_LDX(S|(0ULL<<56));
  auto t0=clk::now();
  for(int64_t i=0;i<IT;++i){ AMX_GENLUT(glut(0,0,true,false,false,2,false,0)); AMX_GENLUT(glut(0,0,true,false,false,3,false,0));
                             AMX_GENLUT(glut(0,0,true,false,false,4,false,0)); AMX_GENLUT(glut(0,0,true,false,false,5,false,0)); }
  double t=sec(clk::now()-t0); AMX_CLR();
  std::printf("genlut generate cost: %.2f cyc/op  (1 generate quantizes 16 lanes)\n", t*GHZ*1e9/(IT*4));
  free(thr);free(src);free(zout); return ok?0:1;
}
