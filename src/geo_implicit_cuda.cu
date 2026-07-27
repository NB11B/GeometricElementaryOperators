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

            float w_r = 0.0f;
            for (int col = 0; col < D_out; ++col) {
                w_r += gy_row[col] * u_r[col];
            }

            int orig_pos = inv_p_r[k];
            float s_val = sign_r[orig_pos];
            float v_val = v_r[orig_pos];

            sum_dx += a * w_r * v_val * s_val;
        }
        grad_x[row * D_in + k] = sum_dx;
    }
}

__global__ void geo_implicit_backward_du_kernel(
    const float * __restrict__ x,
    const float * __restrict__ v,
    const float * __restrict__ alpha,
    const int32_t * __restrict__ perm_indices,
    const float * __restrict__ sign_masks,
    const float * __restrict__ grad_y,
    float * __restrict__ grad_u,
    int N, int D_in, int D_out, int R
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < R && col < D_out) {
        float a = alpha[r];
        const int32_t *perm_r = perm_indices + r * D_in;
        const float *sign_r = sign_masks + r * D_in;
        const float *v_r = v + r * D_in;

        float sum_du = 0.0f;
        for (int row = 0; row < N; ++row) {
            const float *x_row = x + row * D_in;
            float h_r = 0.0f;
            for (int k = 0; k < D_in; ++k) {
                int p_idx = perm_r[k];
                float s_val = sign_r[k];
                h_r += x_row[p_idx] * s_val * v_r[k];
            }
            float gy_val = grad_y[row * D_out + col];
            sum_du += gy_val * h_r;
        }
        grad_u[r * D_out + col] = a * sum_du;
    }
}

__global__ void geo_implicit_backward_dalpha_kernel(
    const float * __restrict__ x,
    const float * __restrict__ u,
    const float * __restrict__ v,
    const int32_t * __restrict__ perm_indices,
    const float * __restrict__ sign_masks,
    const float * __restrict__ grad_y,
    float * __restrict__ grad_alpha,
    int N, int D_in, int D_out, int R
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < R) {
        const int32_t *perm_r = perm_indices + r * D_in;
        const float *sign_r = sign_masks + r * D_in;
        const float *v_r = v + r * D_in;
        const float *u_r = u + r * D_out;

        float sum_dalpha = 0.0f;
        for (int row = 0; row < N; ++row) {
            const float *x_row = x + row * D_in;
            const float *gy_row = grad_y + row * D_out;

            float h_r = 0.0f;
            for (int k = 0; k < D_in; ++k) {
                int p_idx = perm_r[k];
                float s_val = sign_r[k];
                h_r += x_row[p_idx] * s_val * v_r[k];
            }

            float proj_u_dot = 0.0f;
            for (int col = 0; col < D_out; ++col) {
                proj_u_dot += gy_row[col] * u_r[col];
            }

            sum_dalpha += proj_u_dot * h_r;
        }
        grad_alpha[r] = sum_dalpha;
    }
}

__global__ void geo_implicit_backward_dv_kernel(
    const float * __restrict__ x,
    const float * __restrict__ u,
    const float * __restrict__ alpha,
    const int32_t * __restrict__ perm_indices,
    const float * __restrict__ sign_masks,
    const float * __restrict__ grad_y,
    float * __restrict__ grad_v,
    int N, int D_in, int D_out, int R
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < R && k < D_in) {
        float a = alpha[r];
        const int32_t *perm_r = perm_indices + r * D_in;
        const float *sign_r = sign_masks + r * D_in;
        const float *u_r = u + r * D_out;

        int p_idx = perm_r[k];
        float s_val = sign_r[k];

        float sum_dv = 0.0f;
        for (int row = 0; row < N; ++row) {
            float x_val = x[row * D_in + p_idx] * s_val;
            const float *gy_row = grad_y + row * D_out;

            float g_proj_v = 0.0f;
            for (int col = 0; col < D_out; ++col) {
                g_proj_v += gy_row[col] * u_r[col];
            }

            sum_dv += x_val * g_proj_v;
        }
        grad_v[r * D_in + k] = a * sum_dv;
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
    if (!x || !u || !v || !alpha || !perm_indices || !inv_perm || !sign_masks || !grad_y || !grad_x || !grad_u || !grad_v || !grad_alpha || !shape) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    int N = static_cast<int>(shape->batch_tokens);
    int D_in = static_cast<int>(shape->in_features);
    int D_out = static_cast<int>(shape->out_features);
    int R = static_cast<int>(shape->rank);

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // 1. grad_x kernel
    dim3 threads_dx(16, 16);
    dim3 blocks_dx((N + threads_dx.x - 1) / threads_dx.x,
                   (D_in + threads_dx.y - 1) / threads_dx.y);
    geo_implicit_backward_dx_kernel<<<blocks_dx, threads_dx, 0, stream>>>(
        grad_y, u, v, alpha, inv_perm, sign_masks, grad_x, N, D_in, D_out, R
    );

    // 2. grad_u kernel
    dim3 threads_du(4, 32);
    dim3 blocks_du((R + threads_du.x - 1) / threads_du.x,
                   (D_out + threads_du.y - 1) / threads_du.y);
    geo_implicit_backward_du_kernel<<<blocks_du, threads_du, 0, stream>>>(
        x, v, alpha, perm_indices, sign_masks, grad_y, grad_u, N, D_in, D_out, R
    );

    // 3. grad_alpha kernel
    dim3 threads_dalpha(32, 1);
    dim3 blocks_dalpha((R + threads_dalpha.x - 1) / threads_dalpha.x, 1);
    geo_implicit_backward_dalpha_kernel<<<blocks_dalpha, threads_dalpha, 0, stream>>>(
        x, u, v, perm_indices, sign_masks, grad_y, grad_alpha, N, D_in, D_out, R
    );

    // 4. grad_v kernel
    dim3 threads_dv(4, 32);
    dim3 blocks_dv((R + threads_dv.x - 1) / threads_dv.x,
                   (D_in + threads_dv.y - 1) / threads_dv.y);
    geo_implicit_backward_dv_kernel<<<blocks_dv, threads_dv, 0, stream>>>(
        x, u, alpha, perm_indices, sign_masks, grad_y, grad_v, N, D_in, D_out, R
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}
