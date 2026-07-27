#ifndef GEO_REDUCTION_CUDA_CUH
#define GEO_REDUCTION_CUDA_CUH

#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float geo_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float geo_warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(FULL_MASK, val, offset);
        if (other > val) {
            val = other;
        }
    }
    return val;
}

__device__ __forceinline__ float geo_block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = geo_warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = geo_warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ float geo_block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = geo_warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -1e30f;

    if (wid == 0) {
        val = geo_warp_reduce_max(val);
    }
    return val;
}

#endif
