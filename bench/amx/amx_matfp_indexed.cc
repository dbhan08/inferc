// Indexed MATFP gather (op21) — the CORRECT op for fused gather (FMA doesn't support
// indexed; corsix fma.c has no indexed path; matfp.c does). Encoding verbatim from
// corsix matfp.c: lane bits42-45=4 (fp32), bit53=indexed, bit48=4-bit, bit47=0(index X),
// bits49-51=table reg, Xoff=bits10-18 (holds indices), Yoff=bits0-8, bits54-56 must be 0.
// Semantics: x loaded from Xoff = the indices; then x[i] = table[idx[i]]; outer product
// z[j][i] += x[i]*y[j]. With y=ones, z[0][i] = table[idx[i]]. Verify the gather + RE the
// index packing empirically (idx=[0..15], table[k]=100+k -> z[0][i] should be 100+something).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"
static inline uint64_t MatfpIdxX(int zrow,int xoff,int yoff,int src){
  return (4ULL<<42)|(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)src<<49)
       |((uint64_t)xoff<<10)|(uint64_t)yoff|((uint64_t)zrow<<20);
}

int main(){
  float* table=nullptr; uint8_t* idxbuf=nullptr; float* ones=nullptr; float* zero=nullptr; void* zout=nullptr;
  posix_memalign((void**)&table,128,64); posix_memalign((void**)&idxbuf,128,64);
  posix_memalign((void**)&ones,128,64); posix_memalign((void**)&zero,128,64); posix_memalign(&zout,128,64);
  for(int i=0;i<16;++i){ table[i]=100.f+i; ones[i]=1.f; zero[i]=0.f; }
  std::memset(idxbuf,0,64);
  // GUESS A: 16 packed 4-bit indices [0,1,..,15] in low 8 bytes (2 per byte, low nibble first)
  for(int i=0;i<16;++i) idxbuf[i/2] |= (uint8_t)(i&0xF)<<((i&1)*4);
  std::memset(zout,0,64);
  const uint64_t T=(uint64_t)table, IX=(uint64_t)idxbuf, ON=(uint64_t)ones, ZE=(uint64_t)zero, Z=(uint64_t)zout;

  AMX_SET();
  AMX_LDX(IX|(0ULL<<56));               // X[0] = indices (the operand the gather indexes)
  AMX_LDX(T|(1ULL<<56));                // X[1] = table [100..115]  (src_reg=1)
  AMX_LDY(ON|(0ULL<<56));               // Y[0] = ones
  AMX_LDZ(ZE|(0ULL<<56));               // Z[0] = 0 (matfp accumulates)
  AMX_MATFP(MatfpIdxX(/*zrow*/0,/*xoff*/0,/*yoff*/0,/*src_reg*/1));
  AMX_STZ(Z|(0ULL<<56));
  AMX_CLR();

  const float* z=(const float*)zout;
  std::printf("table[k]=100+k, idx=[0..15] packed-4bit. z[0][i] = table[idx[i]] if gather works.\n");
  for(int i=0;i<16;++i) std::printf("  z[0][%2d] = %.1f\n", i, z[i]);
  std::printf("(if z = 100,101,...,115 -> gather works + packing correct. Other pattern -> RE packing.)\n");
  return 0;
}
