#include "geo/tensor_activation_cuda.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ float geo_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gelu_forward_kernel(const float *x, float *out, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.7071067811865475f));
    }
}

__global__ void gelu_vjp_kernel(const float *x, const float *grad_out, float *grad_x, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float xi = x[i];
        const float cdf = 0.5f * (1.0f + erff(xi * 0.7071067811865475f));
        const float pdf_term = xi * 0.3989422804014327f * expf(-0.5f * xi * xi);
        grad_x[i] = grad_out[i] * (cdf + pdf_term);
    }
}

__global__ void silu_mul_forward_kernel(const float *gate, const float *up, float *out, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float s = geo_sigmoidf(gate[i]);
        out[i] = gate[i] * s * up[i];
    }
}

__global__ void silu_mul_vjp_kernel(const float *gate, const float *up, const float *grad_out, float *grad_gate, float *grad_up, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float g = gate[i];
        const float s = geo_sigmoidf(g);
        const float silu = g * s;
        const float dsilu = s * (1.0f + g * (1.0f - s));
        grad_gate[i] = grad_out[i] * up[i] * dsilu;
        grad_up[i] = grad_out[i] * silu;
    }
}

geo_tensor_status launch_status(void) {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

}  // namespace

extern "C" geo_tensor_status geo_tensor_gelu_cuda_forward(const float *x, float *out, size_t count, void *stream) {
    if (x == nullptr || out == nullptr || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    const int threads = 256;
    const int blocks = (int)((count + threads - 1u) / threads);
    gelu_forward_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(x, out, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_gelu_cuda_vjp(const float *x, const float *grad_out, float *grad_x, size_t count, void *stream) {
    if (x == nullptr || grad_out == nullptr || grad_x == nullptr || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    const int threads = 256;
    const int blocks = (int)((count + threads - 1u) / threads);
    gelu_vjp_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(x, grad_out, grad_x, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_silu_mul_cuda_forward(const float *gate, const float *up, float *out, size_t count, void *stream) {
    if (gate == nullptr || up == nullptr || out == nullptr || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    const int threads = 256;
    const int blocks = (int)((count + threads - 1u) / threads);
    silu_mul_forward_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(gate, up, out, count);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_silu_mul_cuda_vjp(const float *gate, const float *up, const float *grad_out, float *grad_gate, float *grad_up, size_t count, void *stream) {
    if (gate == nullptr || up == nullptr || grad_out == nullptr || grad_gate == nullptr || grad_up == nullptr || count == 0u) return GEO_TENSOR_INVALID_ARGUMENT;
    const int threads = 256;
    const int blocks = (int)((count + threads - 1u) / threads);
    silu_mul_vjp_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(gate, up, grad_out, grad_gate, grad_up, count);
    return launch_status();
}
