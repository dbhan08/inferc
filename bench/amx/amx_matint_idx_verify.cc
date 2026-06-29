// GATE for the int8 codebook kernel: does indexed-load work in int8 MATINT? Gather an int8
// codebook (Y) via 4-bit indices, X=ones -> z[m][n] = codebook[idx[n]] for all m. Layout:
// z[n + i%4][i/4] (i=X byte). With X=ones, output-col c=i/4, row = n + i%4. Check codebook gather.
// indexed int8 MATINT: op20, bit53(indexed)+bit54(->alumode8=int8)+lane10(42-45)+bit48(4bit)+bit47(idxY)+src.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"
static inline uint64_t MatintIdxY(int z,int xo,int yo,int src){
  return (10ULL<<42)|(1ULL<<53)|(1ULL<<54)|(1ULL<<48)|(1ULL<<47)|((uint64_t)src<<49)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;
}
int main(){
  int8_t* X; uint8_t* idx; int8_t* cb; void* Z;
  posix_memalign((void**)&X,128,64); posix_memalign((void**)&idx,128,64); posix_memalign((void**)&cb,128,64); posix_memalign(&Z,128,4096);
  for(int i=0;i<64;++i) X[i]=1;
  for(int e=0;e<16;++e) cb[e]=(int8_t)(20+e);                 // codebook [20..35]
  std::memset(idx,0,64);
  // 16 N-values, 4-bit index each. Y read at stride 4? indices packed -- try 16 nibbles in low 8 bytes
  for(int n=0;n<16;++n) idx[2*n] = (uint8_t)(n&0xF);   // index for N=n at nibble 4n
  std::memset(Z,0,4096);
  AMX_SET();
  AMX_LDY((uint64_t)idx|(0ULL<<56));         // Y[0] = indices
  AMX_LDY((uint64_t)cb|(1ULL<<56));          // Y[1] = int8 codebook (src=1)
  AMX_LDX((uint64_t)X|(0ULL<<56));           // X[0] = ones (64 m)
  for(int r=0;r<64;++r) AMX_LDZ((uint64_t)Z|((uint64_t)r<<56));
  AMX_MATINT(MatintIdxY(0,0,0,1));
  int32_t buf[64*16]; for(int r=0;r<64;++r){ AMX_STZ((uint64_t)Z|((uint64_t)r<<56)); std::memcpy(buf+r*16,Z,64); }
  AMX_CLR();
  std::printf("idx[n]=n, cb[e]=20+e. With X=ones, expect z[row=n+i%%4][col=i/4]=cb[n]=20+n.\n");
  std::printf("nonzero outputs (row,col,val):\n");
  int ok=1; for(int n=0;n<16;++n){ int v=buf[(4*n)*16+0]; std::printf("  N=%2d (row %2d): %d (expect %d)%s\n",n,4*n,v,20+n,v==20+n?"":" <-- MISMATCH"); if(v!=20+n)ok=0; }
  std::printf("%s\n", ok?"INDEXED INT8 GATHER WORKS (codebook gather correct)":"mismatch -> packing still off");
  return 0;
}
