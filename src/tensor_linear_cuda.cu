#include "geo/tensor_linear_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

static int geo_cuda_mul_overflows(size_t a, size_t b) {
    return a != 0u && b > SIZE_MAX / a;
}

static geo_tensor_status geo_cuda_validate(
    const void *x,
    const void *weight,
    const void *output,
    geo_tensor_linear_shape shape
) {
    if (x == NULL || weight == NULL || output == NULL ||
        shape.rows == 0u || shape.in_features == 0u || shape.out_features == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    if (geo_cuda_mul_overflows(shape.rows, shape.in_features) ||
        geo_cuda_mul_overflows(shape.rows, shape.out_features) ||
        geo_cuda_mul_overflows(shape.out_features, shape.in_features)) {
        return GEO_TENSOR_OVERFLOW;
    }
    return GEO_TENSOR_OK;
}

__global__ static void geo_tensor_linear_forward_kernel(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    size_t rows,
    size_t in_features,
    size_t out_features
) {
    const size_t linear_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * out_features;
    if (linear_index >= total) {
        return;
    }

    const size_t row = linear_index / out_features;
    const size_t out = linear_index - row * out_features;
    const geo_real_t *x_row = x + row * in_features;
    const geo_real_t *w_row = weight + out * in_features;
    geo_real_t sum = (geo_real_t)0;
    size_t in;
    for (in = 0u; in < in_features; ++in) {
        sum += x_row[in] * w_row[in];
    }
    y[linear_index] = sum;
}

__global__ static void geo_tensor_linear_grad_x_kernel(
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    size_t rows,
    size_t in_features,
    size_t out_features
) {
    const size_t linear_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * in_features;
    if (linear_index >= total) {
        return;
    }

    const size_t row = linear_index / in_features;
    const size_t in = linear_index - row * in_features;
    const geo_real_t *gy_row = grad_y + row * out_features;
    geo_real_t sum = (geo_real_t)0;
    size_t out;
    for (out = 0u; out < out_features; ++out) {
        sum += gy_row[out] * weight[out * in_features + in];
    }
    grad_x[linear_index] = sum;
}

__global__ static void geo_tensor_linear_grad_weight_kernel(
    const geo_real_t *x,
    const geo_real_t *grad_y,
    geo_real_t *grad_weight,
    size_t rows,
    size_t in_features,
    size_t out_features
) {
    const size_t linear_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = out_features * in_features;
    if (linear_index >= total) {
        return;
    }

    const size_t out = linear_index / in_features;
    const size_t in = linear_index - out * in_features;
    geo_real_t sum = (geo_real_t)0;
    size_t row;
    for (row = 0u; row < rows; ++row) {
        sum += grad_y[row * out_features + out] * x[row * in_features + in];
    }
    grad_weight[linear_index] = sum;
}

static geo_tensor_status geo_cuda_launch_status(void) {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

extern "C" geo_tensor_status geo_tensor_linear_cuda_forward(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    geo_tensor_linear_shape shape,
    void *stream_ptr
) {
    const unsigned int threads = 256u;
    const size_t total = shape.rows * shape.out_features;
    const unsigned int blocks = (unsigned int)((total + threads - 1u) / threads);
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    geo_tensor_status status = geo_cuda_validate(x, weight, y, shape);
    if (status != GEO_TENSOR_OK) {
        return status;
    }
    geo_tensor_linear_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, y, shape.rows, shape.in_features, shape.out_features
    );
    return geo_cuda_launch_status();
}

extern "C" geo_tensor_status geo_tensor_linear_cuda_vjp(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    geo_tensor_linear_shape shape,
    void *stream_ptr
) {
    const unsigned int threads = 256u;
    const size_t grad_x_total = shape.rows * shape.in_features;
    const size_t grad_weight_total = shape.out_features * shape.in_features;
    const unsigned int grad_x_blocks = (unsigned int)((grad_x_total + threads - 1u) / threads);
    const unsigned int grad_weight_blocks = (unsigned int)((grad_weight_total + threads - 1u) / threads);
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    geo_tensor_status status = geo_cuda_validate(x, weight, grad_y, shape);
    if (status != GEO_TENSOR_OK || grad_x == NULL || grad_weight == NULL) {
        return status == GEO_TENSOR_OK ? GEO_TENSOR_INVALID_ARGUMENT : status;
    }

    geo_tensor_linear_grad_x_kernel<<<grad_x_blocks, threads, 0, stream>>>(
        weight, grad_y, grad_x, shape.rows, shape.in_features, shape.out_features
    );
    status = geo_cuda_launch_status();
    if (status != GEO_TENSOR_OK) {
        return status;
    }

    geo_tensor_linear_grad_weight_kernel<<<grad_weight_blocks, threads, 0, stream>>>(
        x, grad_y, grad_weight, shape.rows, shape.in_features, shape.out_features
    );
    return geo_cuda_launch_status();
}
