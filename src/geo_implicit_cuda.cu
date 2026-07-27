#include "geo/geo_implicit_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 1. Compute H[n, r] = dot( x[n, perm_r] * sign_r, v_r )  [N, R]
__global__ void geo_implicit_compute_h_kernel(
    const float * __restrict__ x,
    const float * __restrict__ v,
    const int32_t * __restrict__ perm_indices,
    const float * __restrict__ sign_masks,
    float * __restrict__ h,
    int N, int D_in, int R
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && r < R) {
        const int32_t *perm_r = perm_indices + r * D_in;
        const float *sign_r = sign_masks + r * D_in;
        const float *v_r = v + r * D_in;
        const float *x_row = x + row * D_in;

        float sum_h = 0.0f;
        for (int k = 0; k < D_in; ++k) {
            int p_idx = perm_r[k];
            float s_val = sign_r[k];
            sum_h += x_row[p_idx] * s_val * v_r[k];
        }
        h[row * R + r] = sum_h;
    }
}

// 2. Compute W[n, r] = dot( grad_y[n], u_r )  [N, R]
__global__ void geo_implicit_compute_w_kernel(
    const float * __restrict__ grad_y,
    const float * __restrict__ u,
    float * __restrict__ w,
    int N, int D_out, int R
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && r < R) {
        const float *gy_row = grad_y + row * D_out;
        const float *u_r = u + r * D_out;

        float sum_w = 0.0f;
        for (int col = 0; col < D_out; ++col) {
            sum_w += gy_row[col] * u_r[col];
        }
        w[row * R + r] = sum_w;
    }
}

// 3. Fused Forward Kernel using precomputed H[n, r]
__global__ void geo_implicit_forward_fused_kernel(
    const float * __restrict__ u,
    const float * __restrict__ alpha,
    const float * __restrict__ h,
    float * __restrict__ y,
    int N, int D_out, int R
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < D_out) {
        float sum_y = 0.0f;
        const float *h_row = h + row * R;
        for (int r = 0; r < R; ++r) {
            float a = alpha[r];
            float u_val = u[r * D_out + col];
            sum_y += a * h_row[r] * u_val;
        }
        y[row * D_out + col] = sum_y;
    }
}

// 4. Fused dX Kernel using precomputed W[n, r]
__global__ void geo_implicit_backward_dx_fused_kernel(
    const float * __restrict__ v,
    const float * __restrict__ alpha,
    const int32_t * __restrict__ inv_perm,
    const float * __restrict__ sign_masks,
    const float * __restrict__ w,
    float * __restrict__ grad_x,
    int N, int D_in, int R
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && k < D_in) {
        float sum_dx = 0.0f;
        const float *w_row = w + row * R;

        for (int r = 0; r < R; ++r) {
            float a = alpha[r];
            const float *v_r = v + r * D_in;
            const int32_t *inv_p_r = inv_perm + r * D_in;
            const float *sign_r = sign_masks + r * D_in;

            int orig_pos = inv_p_r[k];
            float s_val = sign_r[orig_pos];
            float v_val = v_r[orig_pos];

            sum_dx += a * w_row[r] * v_val * s_val;
        }
        grad_x[row * D_in + k] = sum_dx;
    }
}

// 5. Fused du Kernel using precomputed H[n, r]
__global__ void geo_implicit_backward_du_fused_kernel(
    const float * __restrict__ alpha,
    const float * __restrict__ h,
    const float * __restrict__ grad_y,
    float * __restrict__ grad_u,
    int N, int D_out, int R
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < R && col < D_out) {
        float a = alpha[r];
        float sum_du = 0.0f;
        for (int row = 0; row < N; ++row) {
            float h_val = h[row * R + r];
            float gy_val = grad_y[row * D_out + col];
            sum_du += gy_val * h_val;
        }
        grad_u[r * D_out + col] = a * sum_du;
    }
}

// 6. Fused dalpha Kernel using precomputed H[n, r] and W[n, r]
__global__ void geo_implicit_backward_dalpha_fused_kernel(
    const float * __restrict__ h,
    const float * __restrict__ w,
    float * __restrict__ grad_alpha,
    int N, int R
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < R) {
        float sum_dalpha = 0.0f;
        for (int row = 0; row < N; ++row) {
            sum_dalpha += h[row * R + r] * w[row * R + r];
        }
        grad_alpha[r] = sum_dalpha;
    }
}

// 7. Fused dv Kernel using precomputed W[n, r]
__global__ void geo_implicit_backward_dv_fused_kernel(
    const float * __restrict__ x,
    const float * __restrict__ alpha,
    const int32_t * __restrict__ perm_indices,
    const float * __restrict__ sign_masks,
    const float * __restrict__ w,
    float * __restrict__ grad_v,
    int N, int D_in, int R
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < R && k < D_in) {
        float a = alpha[r];
        const int32_t *perm_r = perm_indices + r * D_in;
        const float *sign_r = sign_masks + r * D_in;

        int p_idx = perm_r[k];
        float s_val = sign_r[k];

        float sum_dv = 0.0f;
        for (int row = 0; row < N; ++row) {
            float x_val = x[row * D_in + p_idx] * s_val;
            float w_val = w[row * R + r];
            sum_dv += x_val * w_val;
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

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Allocate temporary compact H buffer [N, R]
    float *h_buffer = nullptr;
    cudaMallocAsync(&h_buffer, N * R * sizeof(float), stream);

    dim3 threads_h(16, 4);
    dim3 blocks_h((N + threads_h.x - 1) / threads_h.x,
                  (R + threads_h.y - 1) / threads_h.y);
    geo_implicit_compute_h_kernel<<<blocks_h, threads_h, 0, stream>>>(
        x, v, perm_indices, sign_masks, h_buffer, N, D_in, R
    );

    dim3 threads_y(16, 16);
    dim3 blocks_y((N + threads_y.x - 1) / threads_y.x,
                  (D_out + threads_y.y - 1) / threads_y.y);
    geo_implicit_forward_fused_kernel<<<blocks_y, threads_y, 0, stream>>>(
        u, alpha, h_buffer, y, N, D_out, R
    );

    cudaFreeAsync(h_buffer, stream);
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

    // Allocate temporary compact H [N, R] and W [N, R] buffers
    float *h_buffer = nullptr;
    float *w_buffer = nullptr;
    cudaMallocAsync(&h_buffer, N * R * sizeof(float), stream);
    cudaMallocAsync(&w_buffer, N * R * sizeof(float), stream);

    // Compute H[N, R]
    dim3 threads_h(16, 4);
    dim3 blocks_h((N + threads_h.x - 1) / threads_h.x,
                  (R + threads_h.y - 1) / threads_h.y);
    geo_implicit_compute_h_kernel<<<blocks_h, threads_h, 0, stream>>>(
        x, v, perm_indices, sign_masks, h_buffer, N, D_in, R
    );

    // Compute W[N, R]
    dim3 threads_w(16, 4);
    dim3 blocks_w((N + threads_w.x - 1) / threads_w.x,
                  (R + threads_w.y - 1) / threads_w.y);
    geo_implicit_compute_w_kernel<<<blocks_w, threads_w, 0, stream>>>(
        grad_y, u, w_buffer, N, D_out, R
    );

    // 1. Fused dX kernel
    dim3 threads_dx(16, 16);
    dim3 blocks_dx((N + threads_dx.x - 1) / threads_dx.x,
                   (D_in + threads_dx.y - 1) / threads_dx.y);
    geo_implicit_backward_dx_fused_kernel<<<blocks_dx, threads_dx, 0, stream>>>(
        v, alpha, inv_perm, sign_masks, w_buffer, grad_x, N, D_in, R
    );

    // 2. Fused du kernel
    dim3 threads_du(4, 32);
    dim3 blocks_du((R + threads_du.x - 1) / threads_du.x,
                   (D_out + threads_du.y - 1) / threads_du.y);
    geo_implicit_backward_du_fused_kernel<<<blocks_du, threads_du, 0, stream>>>(
        alpha, h_buffer, grad_y, grad_u, N, D_out, R
    );

    // 3. Fused dalpha kernel
    dim3 threads_dalpha(32, 1);
    dim3 blocks_dalpha((R + threads_dalpha.x - 1) / threads_dalpha.x, 1);
    geo_implicit_backward_dalpha_fused_kernel<<<blocks_dalpha, threads_dalpha, 0, stream>>>(
        h_buffer, w_buffer, grad_alpha, N, R
    );

    // 4. Fused dv kernel
    dim3 threads_dv(4, 32);
    dim3 blocks_dv((R + threads_dv.x - 1) / threads_dv.x,
                   (D_in + threads_dv.y - 1) / threads_dv.y);
    geo_implicit_backward_dv_fused_kernel<<<blocks_dv, threads_dv, 0, stream>>>(
        x, alpha, perm_indices, sign_masks, w_buffer, grad_v, N, D_in, R
    );

    cudaFreeAsync(h_buffer, stream);
    cudaFreeAsync(w_buffer, stream);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}
