#include "preprocess.h"

#include <math.h>

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

__device__ inline float DepthSampleOrZero(const float* depth, int H, int W, int y, int x)
{
    x = max(0, min(W - 1, x));
    y = max(0, min(H - 1, y));

    const float value = depth[y * W + x];
    if (!isfinite(value)) return 0.0f;
    return value;
}

__device__ inline void AtomicMinPositiveFloat(float* address, float value)
{
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;

    while (__int_as_float(old) > value) {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

__global__ void DepthDirectionStatsKernel(
    const float* depth,
    int H,
    int W,
    float min_depth,
    float max_depth,
    DepthDirectionStatsGpu* stats
){
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    const float tl = DepthSampleOrZero(depth, H, W, y - 1, x - 1);
    const float tc = DepthSampleOrZero(depth, H, W, y - 1, x);
    const float tr = DepthSampleOrZero(depth, H, W, y - 1, x + 1);
    const float ml = DepthSampleOrZero(depth, H, W, y, x - 1);
    const float mr = DepthSampleOrZero(depth, H, W, y, x + 1);
    const float bl = DepthSampleOrZero(depth, H, W, y + 1, x - 1);
    const float bc = DepthSampleOrZero(depth, H, W, y + 1, x);
    const float br = DepthSampleOrZero(depth, H, W, y + 1, x + 1);

    const float gx = (-tl - 2.0f * ml - bl) + (tr + 2.0f * mr + br);
    const float gy = (-tl - 2.0f * tc - tr) + (bl + 2.0f * bc + br);

    atomicAdd(&stats->sum_dx, gx);
    atomicAdd(&stats->sum_dy, gy);

    const float center = depth[y * W + x];
    if (isfinite(center) && center >= min_depth && center <= max_depth) {
        atomicAdd(&stats->valid_count, 1.0f);
        AtomicMinPositiveFloat(&stats->min_depth, center);
    }
}

extern "C" void LaunchDepthDirectionStats(
    const float* d_depth,
    int H,
    int W,
    float min_depth,
    float max_depth,
    DepthDirectionStatsGpu* d_stats,
    cudaStream_t stream
){
    const DepthDirectionStatsGpu init_stats{0.0f, 0.0f, 0.0f, max_depth};
    cudaMemcpy(d_stats, &init_stats, sizeof(init_stats), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    DepthDirectionStatsKernel<<<grid, block, 0, stream>>>(
        d_depth, H, W, min_depth, max_depth, d_stats
    );
}
