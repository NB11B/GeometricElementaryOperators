#include "geo/tensor_relational_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_CUDA_EXP(x) exp(x)
#define GEO_CUDA_LOG(x) log(x)
#define GEO_CUDA_FABS(x) fabs(x)
#define GEO_CUDA_SQRT(x) sqrt(x)
#define GEO_CUDA_MAX_REAL DBL_MAX
#else
#define GEO_CUDA_EXP(x) expf(x)
#define GEO_CUDA_LOG(x) logf(x)
#define GEO_CUDA_FABS(x) fabsf(x)
#define GEO_CUDA_SQRT(x) sqrtf(x)
#define GEO_CUDA_MAX_REAL FLT_MAX
#endif

// CUDA LogSumExp row helper kernel
__device__ static geo_real_t dev_logsumexp_row(const geo_real_t *row, size_t n) {
    geo_real_t max_val = -GEO_CUDA_MAX_REAL;
    for (size_t j = 0; j < n; ++j) {
        if (row[j] > max_val) max_val = row[j];
    }
    if (isnan(max_val) || isinf(max_val)) return max_val;
    geo_real_t sum = (geo_real_t)0.0;
    for (size_t j = 0; j < n; ++j) {
        sum += GEO_CUDA_EXP(row[j] - max_val);
    }
    return max_val + GEO_CUDA_LOG(sum);
}

// CUDA LogSumExp col helper kernel
__device__ static geo_real_t dev_logsumexp_col(const geo_real_t *mat, size_t col, size_t streams, size_t stride) {
    geo_real_t max_val = -GEO_CUDA_MAX_REAL;
    for (size_t i = 0; i < streams; ++i) {
        geo_real_t val = mat[i * stride + col];
        if (val > max_val) max_val = val;
    }
    if (isnan(max_val) || isinf(max_val)) return max_val;
    geo_real_t sum = (geo_real_t)0.0;
    for (size_t i = 0; i < streams; ++i) {
        sum += GEO_CUDA_EXP(mat[i * stride + col] - max_val);
    }
    return max_val + GEO_CUDA_LOG(sum);
}

__global__ void kernel_relational_project_forward(
    const geo_real_t *logits,
    geo_real_t *relationship,
    geo_real_t *workspace,
    size_t P,
    size_t M,
    uint32_t K
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const geo_real_t *A = logits + m * P * P;
    geo_real_t *H = relationship + m * P * P;
    geo_real_t *ws_m = workspace + m * (K + 1) * P * P;

    // Copy A to step 0 workspace
    for (size_t i = 0; i < P * P; ++i) {
        ws_m[i] = A[i];
    }

    for (uint32_t k = 0; k < K; ++k) {
        const geo_real_t *curr = ws_m + k * P * P;
        geo_real_t *next = ws_m + (k + 1) * P * P;

        geo_real_t Z_row[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];

        // 1. Row norm
        for (size_t p = 0; p < P; ++p) {
            geo_real_t lse = dev_logsumexp_row(curr + p * P, P);
            for (size_t q = 0; q < P; ++q) {
                Z_row[p * P + q] = curr[p * P + q] - lse;
            }
        }

        // 2. Col norm
        for (size_t q = 0; q < P; ++q) {
            geo_real_t lse = dev_logsumexp_col(Z_row, q, P, P);
            for (size_t p = 0; p < P; ++p) {
                next[p * P + q] = Z_row[p * P + q] - lse;
            }
        }
    }

    // Output S = exp(Z_K)
    const geo_real_t *Z_K = ws_m + K * P * P;
    for (size_t i = 0; i < P * P; ++i) {
        H[i] = GEO_CUDA_EXP(Z_K[i]);
    }
}

__global__ void kernel_relational_project_vjp(
    const geo_real_t *logits,
    const geo_real_t *relationship_cotangent,
    geo_real_t *logits_cotangent,
    geo_real_t *workspace,
    size_t P,
    size_t M,
    uint32_t K
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const geo_real_t *cot_S = relationship_cotangent + m * P * P;
    geo_real_t *cot_A = logits_cotangent + m * P * P;
    const geo_real_t *ws_m = workspace + m * (K + 1) * P * P;

    const geo_real_t *Z_K = ws_m + K * P * P;
    geo_real_t dZ[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];

    for (size_t i = 0; i < P * P; ++i) {
        dZ[i] = cot_S[i] * GEO_CUDA_EXP(Z_K[i]);
    }

    for (int k = (int)K - 1; k >= 0; --k) {
        const geo_real_t *Z_prev = ws_m + k * P * P;

        geo_real_t Z_row[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];
        for (size_t p = 0; p < P; ++p) {
            geo_real_t lse = dev_logsumexp_row(Z_prev + p * P, P);
            for (size_t q = 0; q < P; ++q) {
                Z_row[p * P + q] = Z_prev[p * P + q] - lse;
            }
        }

        geo_real_t dZ_row[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];
        for (size_t q = 0; q < P; ++q) {
            geo_real_t sum_col_dZ = (geo_real_t)0.0;
            for (size_t p = 0; p < P; ++p) {
                sum_col_dZ += dZ[p * P + q];
            }
            geo_real_t col_lse = dev_logsumexp_col(Z_row, q, P, P);
            for (size_t p = 0; p < P; ++p) {
                geo_real_t sm = GEO_CUDA_EXP(Z_row[p * P + q] - col_lse);
                dZ_row[p * P + q] = dZ[p * P + q] - sm * sum_col_dZ;
            }
        }

        for (size_t p = 0; p < P; ++p) {
            geo_real_t sum_row_dZ = (geo_real_t)0.0;
            for (size_t q = 0; q < P; ++q) {
                sum_row_dZ += dZ_row[p * P + q];
            }
            geo_real_t row_lse = dev_logsumexp_row(Z_prev + p * P, P);
            for (size_t q = 0; q < P; ++q) {
                geo_real_t sm = GEO_CUDA_EXP(Z_prev[p * P + q] - row_lse);
                dZ[p * P + q] = dZ_row[p * P + q] - sm * sum_row_dZ;
            }
        }
    }

    for (size_t i = 0; i < P * P; ++i) {
        cot_A[i] = dZ[i];
    }
}

__global__ void kernel_identity_gate_forward(
    const geo_real_t *projected,
    const geo_real_t *gate,
    geo_real_t *effective,
    size_t P,
    size_t M
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * P * P) return;

    size_t m = idx / (P * P);
    size_t rem = idx % (P * P);
    size_t p = rem / P;
    size_t q = rem % P;

    geo_real_t g_logit = gate[m];
    geo_real_t g = (geo_real_t)1.0 / ((geo_real_t)1.0 + GEO_CUDA_EXP(-g_logit));
    geo_real_t eye_val = (p == q) ? (geo_real_t)1.0 : (geo_real_t)0.0;
    effective[idx] = ((geo_real_t)1.0 - g) * eye_val + g * projected[idx];
}

__global__ void kernel_identity_gate_vjp(
    const geo_real_t *projected,
    const geo_real_t *gate,
    const geo_real_t *cot_H,
    geo_real_t *cot_S,
    geo_real_t *cot_g,
    size_t P,
    size_t M
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    geo_real_t g_logit = gate[m];
    geo_real_t g = (geo_real_t)1.0 / ((geo_real_t)1.0 + GEO_CUDA_EXP(-g_logit));
    geo_real_t d_sigmoid = g * ((geo_real_t)1.0 - g);

    const geo_real_t *S_m = projected + m * P * P;
    const geo_real_t *cot_H_m = cot_H + m * P * P;
    geo_real_t *cot_S_m = cot_S + m * P * P;

    geo_real_t sum_g = (geo_real_t)0.0;
    for (size_t p = 0; p < P; ++p) {
        for (size_t q = 0; q < P; ++q) {
            size_t idx = p * P + q;
            geo_real_t eye_val = (p == q) ? (geo_real_t)1.0 : (geo_real_t)0.0;
            geo_real_t h_c = cot_H_m[idx];
            cot_S_m[idx] = g * h_c;
            sum_g += h_c * (S_m[idx] - eye_val);
        }
    }
    cot_g[m] = sum_g * d_sigmoid;
}

__global__ void kernel_mix_forward(
    const geo_real_t *state,
    const geo_real_t *relationship,
    geo_real_t *output,
    size_t G,
    size_t P,
    size_t D,
    size_t M
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * P * D) return;

    size_t g = idx / (P * D);
    size_t rem = idx % (P * D);
    size_t p = rem / D;
    size_t d = rem % D;

    size_t m = (M == 1) ? 0 : g;
    const geo_real_t *H = relationship + m * P * P;
    const geo_real_t *X = state + g * P * D;

    geo_real_t sum = (geo_real_t)0.0;
    for (size_t q = 0; q < P; ++q) {
        sum += H[p * P + q] * X[q * D + d];
    }
    output[idx] = sum;
}

__global__ void kernel_mix_vjp_state(
    const geo_real_t *relationship,
    const geo_real_t *cot_Y,
    geo_real_t *cot_X,
    size_t G,
    size_t P,
    size_t D,
    size_t M
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * P * D) return;

    size_t g = idx / (P * D);
    size_t rem = idx % (P * D);
    size_t q = rem / D;
    size_t d = rem % D;

    size_t m = (M == 1) ? 0 : g;
    const geo_real_t *H = relationship + m * P * P;
    const geo_real_t *cot_Y_g = cot_Y + g * P * D;

    geo_real_t sum = (geo_real_t)0.0;
    for (size_t p = 0; p < P; ++p) {
        sum += H[p * P + q] * cot_Y_g[p * D + d];
    }
    cot_X[idx] = sum;
}

__global__ void kernel_mix_vjp_rel(
    const geo_real_t *state,
    const geo_real_t *cot_Y,
    geo_real_t *cot_H,
    size_t G,
    size_t P,
    size_t D,
    size_t M
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    geo_real_t *cot_H_m = cot_H + m * P * P;

    for (size_t p = 0; p < P; ++p) {
        for (size_t q = 0; q < P; ++q) {
            geo_real_t sum = (geo_real_t)0.0;
            for (size_t g = 0; g < G; ++g) {
                if (M != 1 && g != m) continue;
                const geo_real_t *cot_Y_g = cot_Y + g * P * D;
                const geo_real_t *X_g = state + g * P * D;
                for (size_t d = 0; d < D; ++d) {
                    sum += cot_Y_g[p * D + d] * X_g[q * D + d];
                }
            }
            cot_H_m[p * P + q] = sum;
        }
    }
}

__global__ void kernel_read_forward(
    const geo_real_t *state,
    const geo_real_t *r_weights,
    geo_real_t *read_state,
    size_t G,
    size_t P,
    size_t D,
    size_t M
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * D) return;

    size_t g = idx / D;
    size_t d = idx % D;

    size_t m = (M == 1) ? 0 : g;
    const geo_real_t *r = r_weights + m * P;
    const geo_real_t *X = state + g * P * D;

    geo_real_t sum = (geo_real_t)0.0;
    for (size_t p = 0; p < P; ++p) {
        sum += r[p] * X[p * D + d];
    }
    read_state[idx] = sum;
}

__global__ void kernel_write_add_forward(
    const geo_real_t *transported,
    const geo_real_t *source,
    const geo_real_t *w_weights,
    const geo_real_t *source_scale,
    geo_real_t *output,
    size_t G,
    size_t P,
    size_t D,
    size_t M_w,
    size_t M_s
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * P * D) return;

    size_t g = idx / (P * D);
    size_t rem = idx % (P * D);
    size_t p = rem / D;
    size_t d = rem % D;

    size_t mw = (M_w == 1) ? 0 : g;
    const geo_real_t *w = w_weights + mw * P;
    const geo_real_t *u = source + g * D;

    geo_real_t beta = (geo_real_t)1.0;
    if (source_scale && M_s > 0) {
        size_t ms = (M_s == 1) ? 0 : g;
        beta = source_scale[ms];
    }

    output[idx] = transported[idx] + beta * w[p] * u[d];
}

extern "C" {

geo_relational_status geo_relational_project_forward_cuda(
    const geo_real_t *d_logits,
    geo_real_t *d_relationship,
    geo_real_t *d_workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options,
    geo_relational_certificate *certificates,
    void *stream
) {
    if (!d_logits || !d_relationship || !d_workspace || !shape || !options) return GEO_RELATIONAL_INVALID_ARGUMENT;
    size_t M = shape->matrix_count;
    size_t P = shape->streams;
    uint32_t K = options->iterations;

    size_t req_ws = geo_relational_projection_workspace_elements(M, P, K, 1);
    if (workspace_elements < req_ws) return GEO_RELATIONAL_INSUFFICIENT_WORKSPACE;

    cudaStream_t st = (cudaStream_t)stream;
    size_t threads = 64;
    size_t blocks = (M + threads - 1) / threads;

    kernel_relational_project_forward<<<blocks, threads, 0, st>>>(
        d_logits, d_relationship, d_workspace, P, M, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return GEO_RELATIONAL_CUDA_ERROR;

    if (certificates) {
        size_t bytes = M * P * P * sizeof(geo_real_t);
        geo_real_t *h_rel = (geo_real_t *)malloc(bytes);
        if (!h_rel) return GEO_RELATIONAL_INVALID_ARGUMENT;

        if (st) cudaStreamSynchronize(st); else cudaDeviceSynchronize();
        cudaMemcpy(h_rel, d_relationship, bytes, cudaMemcpyDeviceToHost);

        geo_relational_status cst = geo_relational_certify(h_rel, options->epsilon, shape, certificates);
        free(h_rel);
        if (cst != GEO_RELATIONAL_OK) return cst;

        if (options->require_certificate) {
            for (size_t m = 0; m < M; ++m) {
                if (!certificates[m].accepted) return GEO_RELATIONAL_CONSTRAINT_FAILURE;
            }
        }
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_project_vjp_cuda(
    const geo_real_t *d_logits,
    const geo_real_t *d_relationship_cotangent,
    geo_real_t *d_logits_cotangent,
    geo_real_t *d_workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options,
    void *stream
) {
    if (!d_logits || !d_relationship_cotangent || !d_logits_cotangent || !d_workspace || !shape || !options)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t M = shape->matrix_count;
    size_t P = shape->streams;
    uint32_t K = options->iterations;

    size_t req_ws = geo_relational_projection_workspace_elements(M, P, K, 1);
    if (workspace_elements < req_ws) return GEO_RELATIONAL_INSUFFICIENT_WORKSPACE;

    cudaStream_t st = (cudaStream_t)stream;

    geo_relational_status fst = geo_relational_project_forward_cuda(
        d_logits, d_logits_cotangent, d_workspace, workspace_elements, shape, options, NULL, stream
    );
    if (fst != GEO_RELATIONAL_OK) return fst;

    size_t threads = 64;
    size_t blocks = (M + threads - 1) / threads;

    kernel_relational_project_vjp<<<blocks, threads, 0, st>>>(
        d_logits, d_relationship_cotangent, d_logits_cotangent, d_workspace, P, M, K
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_identity_gate_forward_cuda(
    const geo_real_t *d_projected_relationship,
    const geo_real_t *d_gate,
    geo_real_t *d_effective_relationship,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_projected_relationship || !d_gate || !d_effective_relationship || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t M = shape->matrix_count;
    size_t P = shape->streams;
    size_t total = M * P * P;

    cudaStream_t st = (cudaStream_t)stream;
    size_t threads = 256;
    size_t blocks = (total + threads - 1) / threads;

    kernel_identity_gate_forward<<<blocks, threads, 0, st>>>(
        d_projected_relationship, d_gate, d_effective_relationship, P, M
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_identity_gate_vjp_cuda(
    const geo_real_t *d_projected_relationship,
    const geo_real_t *d_gate,
    const geo_real_t *d_effective_relationship_cotangent,
    geo_real_t *d_projected_relationship_cotangent,
    geo_real_t *d_gate_cotangent,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_projected_relationship || !d_gate || !d_effective_relationship_cotangent ||
        !d_projected_relationship_cotangent || !d_gate_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t M = shape->matrix_count;
    size_t P = shape->streams;

    cudaStream_t st = (cudaStream_t)stream;
    size_t threads = 64;
    size_t blocks = (M + threads - 1) / threads;

    kernel_identity_gate_vjp<<<blocks, threads, 0, st>>>(
        d_projected_relationship, d_gate, d_effective_relationship_cotangent,
        d_projected_relationship_cotangent, d_gate_cotangent, P, M
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_mix_forward_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_relationship,
    geo_real_t *d_output,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_state || !d_relationship || !d_output || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    size_t M = shape->matrix_count;
    size_t total = G * P * D;

    cudaStream_t st = (cudaStream_t)stream;
    size_t threads = 256;
    size_t blocks = (total + threads - 1) / threads;

    kernel_mix_forward<<<blocks, threads, 0, st>>>(
        d_state, d_relationship, d_output, G, P, D, M
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_mix_vjp_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_relationship,
    const geo_real_t *d_output_cotangent,
    geo_real_t *d_state_cotangent,
    geo_real_t *d_relationship_cotangent,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_state || !d_relationship || !d_output_cotangent || !d_state_cotangent || !d_relationship_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    size_t M = shape->matrix_count;

    cudaStream_t st = (cudaStream_t)stream;

    size_t total_x = G * P * D;
    size_t threads_x = 256;
    size_t blocks_x = (total_x + threads_x - 1) / threads_x;

    kernel_mix_vjp_state<<<blocks_x, threads_x, 0, st>>>(
        d_relationship, d_output_cotangent, d_state_cotangent, G, P, D, M
    );

    size_t threads_r = 64;
    size_t blocks_r = (M + threads_r - 1) / threads_r;

    kernel_mix_vjp_rel<<<blocks_r, threads_r, 0, st>>>(
        d_state, d_output_cotangent, d_relationship_cotangent, G, P, D, M
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_read_forward_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_read_weights,
    size_t weight_count,
    geo_real_t *d_read_state,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_state || !d_read_weights || !d_read_state || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    size_t total = G * D;

    cudaStream_t st = (cudaStream_t)stream;
    size_t threads = 256;
    size_t blocks = (total + threads - 1) / threads;

    kernel_read_forward<<<blocks, threads, 0, st>>>(
        d_state, d_read_weights, d_read_state, G, P, D, weight_count
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_read_vjp_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_read_weights,
    size_t weight_count,
    const geo_real_t *d_read_state_cotangent,
    geo_real_t *d_state_cotangent,
    geo_real_t *d_read_weights_cotangent,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_state || !d_read_weights || !d_read_state_cotangent || !d_state_cotangent || !d_read_weights_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;

    size_t state_bytes = G * P * D * sizeof(geo_real_t);
    size_t weight_bytes = weight_count * P * sizeof(geo_real_t);
    size_t read_bytes = G * D * sizeof(geo_real_t);

    geo_real_t *h_state = (geo_real_t*)malloc(state_bytes);
    geo_real_t *h_weights = (geo_real_t*)malloc(weight_bytes);
    geo_real_t *h_cot_z = (geo_real_t*)malloc(read_bytes);
    geo_real_t *h_cot_state = (geo_real_t*)malloc(state_bytes);
    geo_real_t *h_cot_weights = (geo_real_t*)malloc(weight_bytes);

    cudaStream_t st = (cudaStream_t)stream;
    if (st) cudaStreamSynchronize(st); else cudaDeviceSynchronize();

    cudaMemcpy(h_state, d_state, state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weights, d_read_weights, weight_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cot_z, d_read_state_cotangent, read_bytes, cudaMemcpyDeviceToHost);

    geo_relational_status status = geo_relational_read_vjp(
        h_state, h_weights, weight_count, h_cot_z, h_cot_state, h_cot_weights, shape
    );

    cudaMemcpy(d_state_cotangent, h_cot_state, state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_read_weights_cotangent, h_cot_weights, weight_bytes, cudaMemcpyHostToDevice);

    free(h_state); free(h_weights); free(h_cot_z); free(h_cot_state); free(h_cot_weights);
    return status;
}

geo_relational_status geo_relational_write_add_forward_cuda(
    const geo_real_t *d_transported_state,
    const geo_real_t *d_source,
    const geo_real_t *d_write_weights,
    size_t weight_count,
    const geo_real_t *d_source_scale,
    size_t scale_count,
    geo_real_t *d_output,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_transported_state || !d_source || !d_write_weights || !d_output || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    size_t total = G * P * D;

    cudaStream_t st = (cudaStream_t)stream;
    size_t threads = 256;
    size_t blocks = (total + threads - 1) / threads;

    kernel_write_add_forward<<<blocks, threads, 0, st>>>(
        d_transported_state, d_source, d_write_weights, d_source_scale, d_output, G, P, D, weight_count, scale_count
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? GEO_RELATIONAL_OK : GEO_RELATIONAL_CUDA_ERROR;
}

geo_relational_status geo_relational_write_add_vjp_cuda(
    const geo_real_t *d_source,
    const geo_real_t *d_write_weights,
    size_t weight_count,
    const geo_real_t *d_source_scale,
    size_t scale_count,
    const geo_real_t *d_output_cotangent,
    geo_real_t *d_transported_state_cotangent,
    geo_real_t *d_source_cotangent,
    geo_real_t *d_write_weights_cotangent,
    geo_real_t *d_source_scale_cotangent,
    const geo_relational_shape *shape,
    void *stream
) {
    if (!d_source || !d_write_weights || !d_output_cotangent || !d_transported_state_cotangent ||
        !d_source_cotangent || !d_write_weights_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;

    size_t state_bytes = G * P * D * sizeof(geo_real_t);
    size_t source_bytes = G * D * sizeof(geo_real_t);
    size_t weight_bytes = weight_count * P * sizeof(geo_real_t);
    size_t scale_bytes = (scale_count > 0) ? (scale_count * sizeof(geo_real_t)) : 0;

    geo_real_t *h_src = (geo_real_t*)malloc(source_bytes);
    geo_real_t *h_w = (geo_real_t*)malloc(weight_bytes);
    geo_real_t *h_scale = scale_bytes ? (geo_real_t*)malloc(scale_bytes) : NULL;
    geo_real_t *h_cot_out = (geo_real_t*)malloc(state_bytes);

    geo_real_t *h_cot_trans = (geo_real_t*)malloc(state_bytes);
    geo_real_t *h_cot_src = (geo_real_t*)malloc(source_bytes);
    geo_real_t *h_cot_w = (geo_real_t*)malloc(weight_bytes);
    geo_real_t *h_cot_scale = scale_bytes ? (geo_real_t*)malloc(scale_bytes) : NULL;

    cudaStream_t st = (cudaStream_t)stream;
    if (st) cudaStreamSynchronize(st); else cudaDeviceSynchronize();

    cudaMemcpy(h_src, d_source, source_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, d_write_weights, weight_bytes, cudaMemcpyDeviceToHost);
    if (h_scale && d_source_scale) cudaMemcpy(h_scale, d_source_scale, scale_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cot_out, d_output_cotangent, state_bytes, cudaMemcpyDeviceToHost);

    geo_relational_status status = geo_relational_write_add_vjp(
        h_src, h_w, weight_count, h_scale, scale_count, h_cot_out,
        h_cot_trans, h_cot_src, h_cot_w, h_cot_scale, shape
    );

    cudaMemcpy(d_transported_state_cotangent, h_cot_trans, state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_cotangent, h_cot_src, source_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_write_weights_cotangent, h_cot_w, weight_bytes, cudaMemcpyHostToDevice);
    if (h_cot_scale && d_source_scale_cotangent) {
        cudaMemcpy(d_source_scale_cotangent, h_cot_scale, scale_bytes, cudaMemcpyHostToDevice);
    }

    free(h_src); free(h_w); if (h_scale) free(h_scale); free(h_cot_out);
    free(h_cot_trans); free(h_cot_src); free(h_cot_w); if (h_cot_scale) free(h_cot_scale);
    return status;
}

} // extern "C"
