// RESULT: FMA indexed gather produces GARBAGE -> FMA does NOT support indexed-load
// (confirmed by corsix fma.c). This negative is why we moved to MATFP (amx_matfp_indexed.cc),
// where the gather verifies exactly. Kept to document the correctness check that caught it.
//
// VERIFY (skill: correctness before believing speed): does indexed-FMA actually
// GATHER a codebook value per lane, or were the indexed bits ignored (running as a
// plain FMA)? Only correctness distinguishes "free fused gather anomaly" from
// "ignored-bits no-op". Setup: table X[1]=[100..115], 16 packed 4-bit indices in
// X[0], Y=ones, skip-Z. If indexed gather works, z[0][i] = table[idx[i]] = 100+idx[i].
// If z[0][i] = X[0] raw / garbage -> bits ignored, NO anomaly.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"
static inline uint64_t IdxFma32(int z,int xo,int yo,int tbl,bool skipZ){
  return (uint64_t(z)<<20)|(uint64_t(xo)<<10)|uint64_t(yo)|(skipZ?(1ULL<<27):0)
        |(1ULL<<53)|(1ULL<<48)|(0ULL<<47)|((uint64_t)tbl<<49);
}

int main(){
  float* table=nullptr; uint8_t* idx=nullptr; float* ones=nullptr; void* zout=nullptr;
  posix_memalign((void**)&table,128,64); posix_memalign((void**)&idx,128,64);
  posix_memalign((void**)&ones,128,64); posix_memalign(&zout,128,64);
  for(int i=0;i<16;++i){ table[i]=100.f+i; ones[i]=1.f; }
  std::memset(idx,0,64);
  // pack 16 4-bit indices = [0,1,2,...,15] into low 8 bytes (2 per byte, low nibble first)
  for(int i=0;i<16;++i) idx[i/2] |= (uint8_t)(i & 0xF) << ((i&1)*4);
  std::memset(zout,0,64);
  const uint64_t T=(uint64_t)table, X=(uint64_t)idx, ON=(uint64_t)ones, Z=(uint64_t)zout;

  AMX_SET();
  AMX_LDX(X|(0ULL<<56));                 // X[0] = packed 4-bit indices
  AMX_LDX(T|(1ULL<<56));                 // X[1] = codebook table [100..115]
  AMX_LDY(ON|(0ULL<<56));                // Y[0] = ones
  // indexed-X FMA: z = gather(X[0] indices -> X[1] table) outer ones  (skip-Z = fresh)
  AMX_FMA32(IdxFma32(0,0,0,1,true));
  AMX_STZ(Z|(0ULL<<56));                 // Z row 0
  AMX_CLR();

  const float* z=(const float*)zout;
  std::printf("idx[i]=i, table[k]=100+k.  Expect z[0][i]=100+i if gather works.\n");
  int ok=1;
  for(int i=0;i<16;++i){ float exp=100.f+i; if(i<8) std::printf("  lane %2d: z=%.1f (expect %.1f)%s\n",i,z[i],exp,z[i]==exp?"":"  <-- MISMATCH"); if(z[i]!=exp) ok=0; }
  std::printf("%s\n", ok? "GATHER WORKS -> free fused-gather anomaly CONFIRMED (indexed-FMA real)"
                       : "mismatch -> indexed bits ignored OR index packing differs; inspect z values");
  return ok?0:1;
}
