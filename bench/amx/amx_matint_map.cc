// Decode the int8 MATINT i32 output layout: for output element z[a][b] (a=X-lane, b=Y-lane),
// find where it lands in Z (row r, col c). X[a]=1, Y[b]=1 -> exactly one nonzero -> f(a,b)=(r,c).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include "amx/aarch64.h"
static inline uint64_t Matint8(int z,int xo,int yo){return (10ULL<<42)|(8ULL<<47)|((uint64_t)z<<20)|((uint64_t)xo<<10)|(uint64_t)yo;}
int main(){
  int8_t* X; int8_t* Y; void* Z; posix_memalign((void**)&X,128,64); posix_memalign((void**)&Y,128,64); posix_memalign(&Z,128,4096);
  AMX_SET();
  auto probe=[&](int a,int b){
    std::memset(X,0,64); std::memset(Y,0,64); X[a]=1; Y[b]=1;
    AMX_LDX((uint64_t)X|(0ULL<<56)); AMX_LDY((uint64_t)Y|(0ULL<<56));
    for(int r=0;r<64;++r) AMX_LDZ((uint64_t)Z|((uint64_t)r<<56));
    AMX_MATINT(Matint8(0,0,0));
    int32_t buf[64*16];
    for(int r=0;r<64;++r){ AMX_STZ((uint64_t)Z|((uint64_t)r<<56)); std::memcpy(buf+r*16,Z,64); }
    for(int r=0;r<64;++r) for(int c=0;c<16;++c) if(buf[r*16+c]) { std::printf("z[i=%2d][j=%2d] -> row %2d col %2d (=%d)\n",a,b,r,c,buf[r*16+c]); return; }
    std::printf("z[i=%2d][j=%2d] -> (none)\n",a,b);
  };
  for(int a : {0,1,2,3,4,16,17}) probe(a,0);
  std::printf("--\n");
  for(int b : {0,1,2,3,4,16,17}) probe(0,b);
  AMX_CLR(); free(X); free(Y); free(Z); return 0;
}
