#include "preprocess.h"

__global__ void Rgb8ToChwF32NormKernel(
    const unsigned char* src, size_t step,
    int H, int W,
    float* dst,
    float m0, float m1, float m2,
    float s0, float s1, float s2
){
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    const unsigned char* row = src + (size_t)y * step;
    const int idx_hwc = x * 3;

    float r = row[idx_hwc + 0] * (1.0f / 255.0f);
    float g = row[idx_hwc + 1] * (1.0f / 255.0f);
    float b = row[idx_hwc + 2] * (1.0f / 255.0f);

    r = (r - m0) / s0;
    g = (g - m1) / s1;
    b = (b - m2) / s2;

    const int plane = H * W;
    const int idx = y * W + x;

    dst[0 * plane + idx] = r;
    dst[1 * plane + idx] = g;
    dst[2 * plane + idx] = b;
}

extern "C" void LaunchRgb8ToChwF32Norm(
    const unsigned char* d_src, size_t src_step,
    int H, int W,
    float* d_dstCHW,
    const float* mean,
    const float* stdv,
    cudaStream_t stream
){
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    Rgb8ToChwF32NormKernel<<<grid, block, 0, stream>>>(
        d_src, src_step, H, W, d_dstCHW,
        mean[0], mean[1], mean[2],
        stdv[0], stdv[1], stdv[2]
    );
}
