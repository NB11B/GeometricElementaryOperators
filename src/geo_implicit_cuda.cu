#include "geo/geo_implicit_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void geo_implicit_forward_kernel(
    const float * __restrict__ x,
    const float * __restrict__ u,
    const float * __restrict__ v,
    const float * __restrict__ alpha,
    const int32_t * __restrict__ perm_indices,
    const float * __restrict__ sign_masks,
    float * __restrict__ y,
    int N, int D_in, int D_out, int R
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < D_out) {
        float sum = 0.0f;
        for (int r = 0; r < R; ++r) {
            float a = alpha[r];
            float u_val = u[r * D_out + col];

            // Permuted dot product: dot( x[row, perm_r] * sign_r, v_r )
            float h_r = 0.0f;
            const int32_t *perm_r = perm_indices + r * D_in;
            const float *sign_r = sign_masks + r * D_in;
            const float *v_r = v + r * D_in;
            const float *x_row = x + row * D_in;

            for (int k = 0; k < D_in; ++k) {
                int p_idx = perm_r[k];
                float s_val = sign_r[k];
                h_r += x_row[p_idx] * s_val * v_r[k];
            }

            sum += a * h_r * u_val;
        }
        y[row * D_out + col] = sum;
    }
}

__global__ void geo_implicit_backward_dx_kernel(
    const float * __restrict__ grad_y,
    const float * __restrict__ u,
    const float * __restrict__ v,
    const float * __restrict__ alpha,
    const int32_t * __restrict__ inv_perm,
    const float * __restrict__ sign_masks,
    float * __restrict__ grad_x,
    int N, int D_in, int D_out, int R
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && k < D_in) {
        float sum_dx = 0.0f;
        const float *gy_row = grad_y + row * D_out;

        for (int r = 0; r < R; ++r) {
            float a = alpha[r];
            const float *u_r = u + r * D_out;
            const float *v_r = v + r * D_in;
            const int32_t *inv_p_r = inv_perm + r * D_in;
            const float *sign_r = sign_masks + r * D_in;

            // Compute w_r = dot(grad_y[row], u_r)
            float w_r = 0.0f;
            for (int col = 0; col < D_out; ++col) {
                w_r += gy_row[col] * u_r[col];
            }

            // Routing to x[k]
            int orig_pos = inv_p_r[k];
            float s_val = sign_r[orig_pos];
            float v_val = v_r[orig_pos];

            sum_dx += a * w_r * v_val * s_val;
        }
        grad_x[row * D_in + k] = sum_dx;
    }
}

extern "C" geo_tensor_status geo_implicit_linear_cuda_forward(
    const float *x,
    const float *u,
    const float *v,
    const float *alpha,
    const int32_t *perm_indices,
    const float *sign_masks,
    float *y,
    const geo_implicit_shape *shape,
    void *stream_ptr
) {
    if (!x || !u || !v || !alpha || !perm_indices || !sign_masks || !y || !shape) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    int N = static_cast<int>(shape->batch_tokens);
    int D_in = static_cast<int>(shape->in_features);
    int D_out = static_cast<int>(shape->out_features);
    int R = static_cast<int>(shape->rank);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (D_out + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    geo_implicit_forward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        x, u, v, alpha, perm_indices, sign_masks, y, N, D_in, D_out, R
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

extern "C" geo_tensor_status geo_implicit_linear_cuda_vjp(
    const float *x,
    const float *u,
    const float *v,
    const float *alpha,
    const int32_t *perm_indices,
    const int32_t *inv_perm,
    const float *sign_masks,
    const float *grad_y,
    float *grad_x,
    float *grad_u,
    float *grad_v,
    float *grad_alpha,
    const geo_implicit_shape *shape,
    void *stream_ptr
) {
    if (!x || !u || !v || !alpha || !perm_indices || !inv_perm || !sign_masks || !grad_y || !grad_x || !shape) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    int N = static_cast<int>(shape->batch_tokens);
    int D_in = static_cast<int>(shape->in_features);
    int D_out = static_cast<int>(shape->out_features);
    int R = static_cast<int>(shape->rank);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (D_in + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    geo_implicit_backward_dx_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        grad_y, u, v, alpha, inv_perm, sign_masks, grad_x, N, D_in, D_out, R
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}
