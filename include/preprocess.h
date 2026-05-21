#pragma once
#include <cuda_runtime.h>

struct DepthDirectionStatsGpu {
    float sum_dx;
    float sum_dy;
    float valid_count;
    float min_depth;
};

#ifdef __cplusplus
extern "C" {
#endif

void LaunchRgb8ToChwF32Norm(const unsigned char* d_src, size_t src_step,int H, int W,float* d_dstCHW,const float* mean,const float* stdv,cudaStream_t stream);
void LaunchDepthDirectionStats(const float* d_depth, int H, int W, float min_depth, float max_depth, DepthDirectionStatsGpu* d_stats, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
