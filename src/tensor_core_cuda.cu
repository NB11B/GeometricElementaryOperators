#include "geo/tensor_core_cuda.h"

#include <cuda_runtime.h>

namespace {
constexpr int kThreads = 256;

__global__ void add_forward_kernel(const float *a, const float *b, float *out, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}

__global__ void add_vjp_kernel(const float *grad_out, float *grad_a, float *grad_b, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) grad_a[i] = grad_b[i] = grad_out[i];
}

__global__ void mul_forward_kernel(const float *a, const float *b, float *out, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] * b[i];
}

__global__ void mul_vjp_kernel(const float *a, const float *b, const float *grad_out, float *grad_a, float *grad_b, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        grad_a[i] = grad_out[i] * b[i];
        grad_b[i] = grad_out[i] * a[i];
    }
}

__global__ void scale_kernel(const float *x, float scalar, float *out, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = x[i] * scalar;
}

__global__ void rms_forward_kernel(
    const float *x, const float *weight, float epsilon, float *out, float *inv_rms,
    size_t rows, size_t features
) {
    extern __shared__ float shared[];
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = row * features;
    for (size_t f = threadIdx.x; f < features; f += blockDim.x) {
        const float v = x[base + f];
        sum += v * v;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_rms[row] = rsqrtf(shared[0] / static_cast<float>(features) + epsilon);
    __syncthreads();
    const float inv = inv_rms[row];
    for (size_t f = threadIdx.x; f < features; f += blockDim.x) {
        out[base + f] = x[base + f] * inv * weight[f];
    }
}

__global__ void rms_vjp_kernel(
    const float *x, const float *weight, const float *grad_out, const float *inv_rms,
    float *grad_x, float *grad_weight, size_t rows, size_t features
) {
    extern __shared__ float shared[];
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = row * features;
    const float inv = inv_rms[row];
    float ux = 0.0f;
    for (size_t f = threadIdx.x; f < features; f += blockDim.x) {
        ux += grad_out[base + f] * weight[f] * x[base + f];
    }
    shared[threadIdx.x] = ux;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean_ux = shared[0] / static_cast<float>(features);
    for (size_t f = threadIdx.x; f < features; f += blockDim.x) {
        const float g = grad_out[base + f];
        const float u = g * weight[f];
        grad_x[base + f] = inv * u - x[base + f] * inv * inv * inv * mean_ux;
        atomicAdd(&grad_weight[f], g * x[base + f] * inv);
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

unsigned blocks_for(size_t count) {
    return static_cast<unsigned>((count + static_cast<size_t>(kThreads) - 1u) / static_cast<size_t>(kThreads));
}
}

extern "C" geo_tensor_status geo_tensor_add_cuda_forward(const float *a, const float *b, float *out, size_t count, void *stream) {
    if (!a || !b || !out || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    add_forward_kernel<<<blocks_for(count), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a, b, out, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_add_cuda_vjp(const float *grad_out, float *grad_a, float *grad_b, size_t count, void *stream) {
    if (!grad_out || !grad_a || !grad_b || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    add_vjp_kernel<<<blocks_for(count), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(grad_out, grad_a, grad_b, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_mul_cuda_forward(const float *a, const float *b, float *out, size_t count, void *stream) {
    if (!a || !b || !out || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    mul_forward_kernel<<<blocks_for(count), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a, b, out, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_mul_cuda_vjp(const float *a, const float *b, const float *grad_out, float *grad_a, float *grad_b, size_t count, void *stream) {
    if (!a || !b || !grad_out || !grad_a || !grad_b || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    mul_vjp_kernel<<<blocks_for(count), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a, b, grad_out, grad_a, grad_b, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_scale_cuda_forward(const float *x, float scalar, float *out, size_t count, void *stream) {
    if (!x || !out || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    scale_kernel<<<blocks_for(count), kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(x, scalar, out, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_scale_cuda_vjp(const float *grad_out, float scalar, float *grad_x, size_t count, void *stream) {
    return geo_tensor_scale_cuda_forward(grad_out, scalar, grad_x, count, stream);
}

extern "C" geo_tensor_status geo_tensor_rms_norm_cuda_forward(const float *x, const float *weight, float epsilon, float *out, float *inv_rms, geo_tensor_norm_shape shape, void *stream) {
    if (!x || !weight || !out || !inv_rms || shape.rows == 0u || shape.features == 0u || epsilon <= 0.0f) return GEO_TENSOR_INVALID_ARGUMENT;
    rms_forward_kernel<<<static_cast<unsigned>(shape.rows), kThreads, kThreads * sizeof(float), reinterpret_cast<cudaStream_t>(stream)>>>(x, weight, epsilon, out, inv_rms, shape.rows, shape.features);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_rms_norm_cuda_vjp(const float *x, const float *weight, const float *grad_out, const float *inv_rms, float *grad_x, float *grad_weight, geo_tensor_norm_shape shape, void *stream) {
    if (!x || !weight || !grad_out || !inv_rms || !grad_x || !grad_weight || shape.rows == 0u || shape.features == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    cudaError_t zero_status = cudaMemsetAsync(grad_weight, 0, shape.features * sizeof(float), reinterpret_cast<cudaStream_t>(stream));
    if (zero_status != cudaSuccess) return GEO_TENSOR_CUDA_ERROR;
    rms_vjp_kernel<<<static_cast<unsigned>(shape.rows), kThreads, kThreads * sizeof(float), reinterpret_cast<cudaStream_t>(stream)>>>(x, weight, grad_out, inv_rms, grad_x, grad_weight, shape.rows, shape.features);
    return launch_status();
}
