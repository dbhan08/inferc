// Determine the AMX MATINT int8->int32 tile (MACs/instruction) so we can compute GFLOPS
// and answer: does matching ggml's int8 precision keep our speed advantage, or erase it?
// X = int8 ones, Y = int8 [0,1,2,...]; z[i][j] += x[i]*y[j]. Dump Z (int32) to read the tile.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "amx/aarch64.h"
static inline uint64_t Matint8(int z,int xo,int yo){return (10ULL<<42)|(8ULL<<47)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
int main(){
  int8_t* X; int8_t* Y; void* Z; posix_memalign((void**)&X,128,64); posix_memalign((void**)&Y,128,64); posix_memalign(&Z,128,256);
  for(int i=0;i<64;++i){ X[i]=1; Y[i]=0; } Y[0]=1;
  std::memset(Z,0,256);
  AMX_SET();
  AMX_LDX((uint64_t)X|(0ULL<<56)); AMX_LDY((uint64_t)Y|(0ULL<<56));
  for(int r=0;r<64;++r) AMX_LDZ((uint64_t)Z|((uint64_t)r<<56));   // zero Z rows
  AMX_MATINT(Matint8(0,0,0));
  // dump several Z rows as int32 (16 per row)
  for(int r=0;r<64;++r){ AMX_STZ((uint64_t)Z|((uint64_t)r<<56)); int32_t* z=(int32_t*)Z;
    std::printf("Z row %2d: ",r); for(int j=0;j<16;++j) std::printf("%d ",z[j]); std::printf("\n"); }
  AMX_CLR();
  std::printf("(x=ones,y=[0..]: z[i][j]=y[j]. Which rows are nonzero + values -> tile i-extent & layout.)\n");
  free(X); free(Y); free(Z); return 0;
}
