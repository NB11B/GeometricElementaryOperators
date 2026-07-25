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

#define TILE_DIM 16

__global__ static void geo_tensor_linear_forward_tiled_kernel(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    size_t rows,
    size_t in_features,
    size_t out_features
) {
    const size_t row = (size_t)blockIdx.y * TILE_DIM + threadIdx.y;
    const size_t out = (size_t)blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ geo_real_t s_x[TILE_DIM][TILE_DIM];
    __shared__ geo_real_t s_w[TILE_DIM][TILE_DIM];

    geo_real_t sum = (geo_real_t)0;

    const size_t num_tiles = (in_features + TILE_DIM - 1u) / TILE_DIM;

    for (size_t t = 0u; t < num_tiles; ++t) {
        const size_t x_col = t * TILE_DIM + threadIdx.x;
        const size_t w_col = t * TILE_DIM + threadIdx.y;

        if (row < rows && x_col < in_features) {
            s_x[threadIdx.y][threadIdx.x] = x[row * in_features + x_col];
        } else {
            s_x[threadIdx.y][threadIdx.x] = (geo_real_t)0;
        }

        if (out < out_features && w_col < in_features) {
            s_w[threadIdx.x][threadIdx.y] = weight[out * in_features + w_col];
        } else {
            s_w[threadIdx.x][threadIdx.y] = (geo_real_t)0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_x[threadIdx.y][k] * s_w[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (row < rows && out < out_features) {
        y[row * out_features + out] = sum;
    }
}

__global__ static void geo_tensor_linear_grad_x_tiled_kernel(
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    size_t rows,
    size_t in_features,
    size_t out_features
) {
    const size_t row = (size_t)blockIdx.y * TILE_DIM + threadIdx.y;
    const size_t in = (size_t)blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ geo_real_t s_gy[TILE_DIM][TILE_DIM];
    __shared__ geo_real_t s_w[TILE_DIM][TILE_DIM];

    geo_real_t sum = (geo_real_t)0;
    const size_t num_tiles = (out_features + TILE_DIM - 1u) / TILE_DIM;

    for (size_t t = 0u; t < num_tiles; ++t) {
        const size_t gy_col = t * TILE_DIM + threadIdx.x;
        const size_t w_row = t * TILE_DIM + threadIdx.y;

        if (row < rows && gy_col < out_features) {
            s_gy[threadIdx.y][threadIdx.x] = grad_y[row * out_features + gy_col];
        } else {
            s_gy[threadIdx.y][threadIdx.x] = (geo_real_t)0;
        }

        if (w_row < out_features && in < in_features) {
            s_w[threadIdx.y][threadIdx.x] = weight[w_row * in_features + in];
        } else {
            s_w[threadIdx.y][threadIdx.x] = (geo_real_t)0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_gy[threadIdx.y][k] * s_w[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rows && in < in_features) {
        grad_x[row * in_features + in] = sum;
    }
}

__global__ static void geo_tensor_linear_grad_weight_tiled_kernel(
    const geo_real_t *x,
    const geo_real_t *grad_y,
    geo_real_t *grad_weight,
    size_t rows,
    size_t in_features,
    size_t out_features
) {
    const size_t out = (size_t)blockIdx.y * TILE_DIM + threadIdx.y;
    const size_t in = (size_t)blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ geo_real_t s_gy[TILE_DIM][TILE_DIM];
    __shared__ geo_real_t s_x[TILE_DIM][TILE_DIM];

    geo_real_t sum = (geo_real_t)0;
    const size_t num_tiles = (rows + TILE_DIM - 1u) / TILE_DIM;

    for (size_t t = 0u; t < num_tiles; ++t) {
        const size_t r_row = t * TILE_DIM + threadIdx.x;
        const size_t r_col = t * TILE_DIM + threadIdx.y;

        if (out < out_features && r_row < rows) {
            s_gy[threadIdx.y][threadIdx.x] = grad_y[r_row * out_features + out];
        } else {
            s_gy[threadIdx.y][threadIdx.x] = (geo_real_t)0;
        }

        if (r_col < rows && in < in_features) {
            s_x[threadIdx.x][threadIdx.y] = x[r_col * in_features + in];
        } else {
            s_x[threadIdx.x][threadIdx.y] = (geo_real_t)0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_gy[threadIdx.y][k] * s_x[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (out < out_features && in < in_features) {
        grad_weight[out * in_features + in] = sum;
    }
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
    geo_tensor_status status = geo_cuda_validate(x, weight, y, shape);
    if (status != GEO_TENSOR_OK) {
        return status;
    }
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(
        (unsigned int)((shape.out_features + TILE_DIM - 1u) / TILE_DIM),
        (unsigned int)((shape.rows + TILE_DIM - 1u) / TILE_DIM)
    );
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    geo_tensor_linear_forward_tiled_kernel<<<blocks, threads, 0, stream>>>(
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
    geo_tensor_status status = geo_cuda_validate(x, weight, grad_y, shape);
    if (status != GEO_TENSOR_OK || grad_x == NULL || grad_weight == NULL) {
        return status == GEO_TENSOR_OK ? GEO_TENSOR_INVALID_ARGUMENT : status;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 gx_blocks(
        (unsigned int)((shape.in_features + TILE_DIM - 1u) / TILE_DIM),
        (unsigned int)((shape.rows + TILE_DIM - 1u) / TILE_DIM)
    );
    geo_tensor_linear_grad_x_tiled_kernel<<<gx_blocks, threads, 0, stream>>>(
        weight, grad_y, grad_x, shape.rows, shape.in_features, shape.out_features
    );
    status = geo_cuda_launch_status();
    if (status != GEO_TENSOR_OK) {
        return status;
    }

    dim3 gw_blocks(
        (unsigned int)((shape.in_features + TILE_DIM - 1u) / TILE_DIM),
        (unsigned int)((shape.out_features + TILE_DIM - 1u) / TILE_DIM)
    );
    geo_tensor_linear_grad_weight_tiled_kernel<<<gw_blocks, threads, 0, stream>>>(
        x, grad_y, grad_weight, shape.rows, shape.in_features, shape.out_features
    );
    return geo_cuda_launch_status();
}
