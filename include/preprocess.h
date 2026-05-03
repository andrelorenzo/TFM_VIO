#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void LaunchRgb8ToChwF32Norm(const unsigned char* d_src, size_t src_step,int H, int W,float* d_dstCHW,const float* mean,const float* stdv,cudaStream_t stream);

#ifdef __cplusplus
}
#endif
