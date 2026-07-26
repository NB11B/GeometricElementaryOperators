#include "geo/tensor_attention_cuda.h"

#include <cuda_runtime.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

namespace {

enum {
    GEO_ATTENTION_BLOCK_SIZE = 256,
};

size_t launch_blocks(size_t total_elements) {
    if (total_elements == 0u) {
        return 1u;
    }
    const size_t block_size = static_cast<size_t>(GEO_ATTENTION_BLOCK_SIZE);
    return (total_elements + block_size - 1u) / block_size;
}

bool valid_shape(geo_tensor_attention_shape shape) {
    return shape.outer > 0u && shape.tokens > 0u && shape.head_dim > 0u;
}

__device__ size_t data_index(
    size_t outer,
    size_t token,
    size_t dim,
    size_t tokens,
    size_t head_dim
) {
    return (outer * tokens + token) * head_dim + dim;
}

__device__ size_t probability_index(
    size_t outer,
    size_t query,
    size_t key,
    size_t tokens
) {
    return (outer * tokens + query) * tokens + key;
}

__device__ float dot_tokens(
    const float *a,
    size_t a_token,
    const float *b,
    size_t b_token,
    size_t outer,
    size_t tokens,
    size_t head_dim
) {
    float sum = 0.0f;
    for (size_t dim = 0u; dim < head_dim; ++dim) {
        sum += a[data_index(outer, a_token, dim, tokens, head_dim)] *
               b[data_index(outer, b_token, dim, tokens, head_dim)];
    }
    return sum;
}

__global__ void causal_attention_forward_kernel(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *probabilities,
    size_t outer_count,
    size_t tokens,
    size_t head_dim
) {
    const size_t rows = outer_count * tokens;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t outer = row / tokens;
        const size_t query = row - outer * tokens;

        float max_score = -CUDART_INF_F;
        for (size_t key = 0u; key <= query; ++key) {
            const float score = dot_tokens(
                q, query, k, key, outer, tokens, head_dim
            ) * scale;
            max_score = fmaxf(max_score, score);
        }

        float normalizer = 0.0f;
        for (size_t key = 0u; key < tokens; ++key) {
            const size_t p_index = probability_index(outer, query, key, tokens);
            if (key <= query) {
                const float score = dot_tokens(
                    q, query, k, key, outer, tokens, head_dim
                ) * scale;
                const float unnormalized = expf(score - max_score);
                probabilities[p_index] = unnormalized;
                normalizer += unnormalized;
            } else {
                probabilities[p_index] = 0.0f;
            }
        }

        for (size_t key = 0u; key <= query; ++key) {
            probabilities[probability_index(outer, query, key, tokens)] /= normalizer;
        }

        for (size_t dim = 0u; dim < head_dim; ++dim) {
            float value = 0.0f;
            for (size_t key = 0u; key <= query; ++key) {
                value += probabilities[probability_index(outer, query, key, tokens)] *
                         v[data_index(outer, key, dim, tokens, head_dim)];
            }
            out[data_index(outer, query, dim, tokens, head_dim)] = value;
        }
    }
}

// Kernel 1: Compute dP and write both dS and transposed dS_t into workspace
__global__ void attention_vjp_kernel_dp_ds(
    const float *probabilities,
    const float *grad_out,
    const float *v,
    float *ds_workspace,
    float *ds_workspace_t,
    size_t outer_count,
    size_t tokens,
    size_t head_dim
) {
    const size_t rows = outer_count * tokens;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t outer = row / tokens;
        const size_t query = row - outer * tokens;

        float softmax_dot = 0.0f;
        for (size_t key = 0u; key <= query; ++key) {
            float d_prob = 0.0f;
            for (size_t dim = 0u; dim < head_dim; ++dim) {
                d_prob += grad_out[data_index(outer, query, dim, tokens, head_dim)] *
                          v[data_index(outer, key, dim, tokens, head_dim)];
            }
            const float p = probabilities[probability_index(outer, query, key, tokens)];
            ds_workspace[probability_index(outer, query, key, tokens)] = d_prob;
            softmax_dot += p * d_prob;
        }

        for (size_t key = 0u; key <= query; ++key) {
            const size_t p_idx = probability_index(outer, query, key, tokens);
            const size_t pt_idx = probability_index(outer, key, query, tokens);
            const float p = probabilities[p_idx];
            const float d_prob = ds_workspace[p_idx];
            const float ds_val = p * (d_prob - softmax_dot);
            ds_workspace[p_idx] = ds_val;
            ds_workspace_t[pt_idx] = ds_val;
        }
        for (size_t key = query + 1; key < tokens; ++key) {
            const size_t p_idx = probability_index(outer, query, key, tokens);
            const size_t pt_idx = probability_index(outer, key, query, tokens);
            ds_workspace[p_idx] = 0.0f;
            ds_workspace_t[pt_idx] = 0.0f;
        }
    }
}

// Kernel 2: Compute grad_q
__global__ void attention_vjp_kernel_dq(
    const float *ds_workspace,
    const float *k,
    float *grad_q,
    size_t outer_count,
    size_t tokens,
    size_t head_dim
) {
    const size_t rows = outer_count * tokens;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t outer = row / tokens;
        const size_t query = row - outer * tokens;

        for (size_t dim = 0u; dim < head_dim; ++dim) {
            float sum_q = 0.0f;
            for (size_t key = 0u; key <= query; ++key) {
                const float ds = ds_workspace[probability_index(outer, query, key, tokens)];
                sum_q += ds * k[data_index(outer, key, dim, tokens, head_dim)];
            }
            grad_q[data_index(outer, query, dim, tokens, head_dim)] = scale * sum_q;
        }
    }
}

// Kernel 3: Compute grad_k and grad_v using coalesced transposed ds_workspace_t
__global__ void attention_vjp_kernel_dk_dv(
    const float *ds_workspace_t,
    const float *probabilities,
    const float *q,
    const float *grad_out,
    float *grad_k,
    float *grad_v,
    size_t outer_count,
    size_t tokens,
    size_t head_dim
) {
    const size_t rows = outer_count * tokens;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t outer = row / tokens;
        const size_t key = row - outer * tokens;

        for (size_t dim = 0u; dim < head_dim; ++dim) {
            float sum_k = 0.0f;
            float sum_v = 0.0f;

            for (size_t query = key; query < tokens; ++query) {
                const float ds = ds_workspace_t[probability_index(outer, key, query, tokens)];
                const float p = probabilities[probability_index(outer, query, key, tokens)];

                sum_k += ds * q[data_index(outer, query, dim, tokens, head_dim)];
                sum_v += p * grad_out[data_index(outer, query, dim, tokens, head_dim)];
            }

            const size_t key_dim_idx = data_index(outer, key, dim, tokens, head_dim);
            grad_k[key_dim_idx] = scale * sum_k;
            grad_v[key_dim_idx] = sum_v;
        }
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

__global__ void causal_attention_forward_no_probs_kernel(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    size_t outer_count,
    size_t tokens,
    size_t head_dim
) {
    const size_t rows = outer_count * tokens;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t outer = row / tokens;
        const size_t query = row - outer * tokens;

        float max_score = -CUDART_INF_F;
        for (size_t key = 0u; key <= query; ++key) {
            const float score = dot_tokens(q, query, k, key, outer, tokens, head_dim) * scale;
            max_score = fmaxf(max_score, score);
        }

        float normalizer = 0.0f;
        for (size_t key = 0u; key <= query; ++key) {
            const float score = dot_tokens(q, query, k, key, outer, tokens, head_dim) * scale;
            normalizer += expf(score - max_score);
        }

        const float inv_normalizer = 1.0f / normalizer;

        for (size_t dim = 0u; dim < head_dim; ++dim) {
            float value = 0.0f;
            for (size_t key = 0u; key <= query; ++key) {
                const float score = dot_tokens(q, query, k, key, outer, tokens, head_dim) * scale;
                const float p = expf(score - max_score) * inv_normalizer;
                value += p * v[data_index(outer, key, dim, tokens, head_dim)];
            }
            out[data_index(outer, query, dim, tokens, head_dim)] = value;
        }
    }
}

__global__ void causal_attention_streaming_vjp_kernel(
    const float *q,
    const float *k,
    const float *v,
    const float *out,
    const float *grad_out,
    float *grad_q,
    float *grad_k,
    float *grad_v,
    size_t outer_count,
    size_t tokens,
    size_t head_dim
) {
    const size_t rows = outer_count * tokens;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t outer = row / tokens;
        const size_t query = row - outer * tokens;

        float max_score = -CUDART_INF_F;
        for (size_t key = 0u; key <= query; ++key) {
            const float score = dot_tokens(q, query, k, key, outer, tokens, head_dim) * scale;
            max_score = fmaxf(max_score, score);
        }

        float normalizer = 0.0f;
        for (size_t key = 0u; key <= query; ++key) {
            const float score = dot_tokens(q, query, k, key, outer, tokens, head_dim) * scale;
            normalizer += expf(score - max_score);
        }
        const float inv_normalizer = (normalizer > 0.0f) ? (1.0f / normalizer) : 0.0f;

        float D_i = 0.0f;
        for (size_t dim = 0u; dim < head_dim; ++dim) {
            D_i += grad_out[data_index(outer, query, dim, tokens, head_dim)] *
                   out[data_index(outer, query, dim, tokens, head_dim)];
        }

        for (size_t key = 0u; key <= query; ++key) {
            const float score = dot_tokens(q, query, k, key, outer, tokens, head_dim) * scale;
            const float p = expf(score - max_score) * inv_normalizer;

            float grad_out_dot_v = 0.0f;
            for (size_t dim = 0u; dim < head_dim; ++dim) {
                grad_out_dot_v += grad_out[data_index(outer, query, dim, tokens, head_dim)] *
                                  v[data_index(outer, key, dim, tokens, head_dim)];
            }

            const float ds = p * (grad_out_dot_v - D_i);

            for (size_t dim = 0u; dim < head_dim; ++dim) {
                atomicAdd(&grad_q[data_index(outer, query, dim, tokens, head_dim)], ds * k[data_index(outer, key, dim, tokens, head_dim)] * scale);
                atomicAdd(&grad_k[data_index(outer, key, dim, tokens, head_dim)], ds * q[data_index(outer, query, dim, tokens, head_dim)] * scale);
                atomicAdd(&grad_v[data_index(outer, key, dim, tokens, head_dim)], p * grad_out[data_index(outer, query, dim, tokens, head_dim)]);
            }
        }
    }
}

static unsigned long long g_n_forward_with_probs_calls = 0;
static unsigned long long g_n_forward_no_probs_calls = 0;
static unsigned long long g_n_streaming_forward_calls = 0;
static unsigned long long g_n_backward_probability_recompute_calls = 0;
static unsigned long long g_n_attention_vjp_calls = 0;
static unsigned long long g_n_streaming_vjp_calls = 0;
static float g_perturbation_delta_host = 0.0f;

__global__ void apply_perturbation_kernel(float *out, float delta) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] += delta;
    }
}

}  // namespace

extern "C" void geo_tensor_causal_attention_get_counters(geo_attention_backend_counters *out) {
    if (out) {
        out->n_forward_with_probs_calls = g_n_forward_with_probs_calls;
        out->n_forward_no_probs_calls = g_n_forward_no_probs_calls;
        out->n_streaming_forward_calls = g_n_streaming_forward_calls;
        out->n_backward_probability_recompute_calls = g_n_backward_probability_recompute_calls;
        out->n_attention_vjp_calls = g_n_attention_vjp_calls;
        out->n_streaming_vjp_calls = g_n_streaming_vjp_calls;
    }
}

extern "C" void geo_tensor_causal_attention_reset_counters(void) {
    g_n_forward_with_probs_calls = 0;
    g_n_forward_no_probs_calls = 0;
    g_n_streaming_forward_calls = 0;
    g_n_backward_probability_recompute_calls = 0;
    g_n_attention_vjp_calls = 0;
    g_n_streaming_vjp_calls = 0;
}

extern "C" void geo_tensor_causal_attention_set_perturbation(float delta) {
    g_perturbation_delta_host = delta;
}

extern "C" geo_tensor_status geo_tensor_causal_attention_cuda_streaming_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    geo_tensor_attention_shape shape,
    void *stream
) {
    g_n_streaming_forward_calls++;
    geo_tensor_status status = geo_tensor_causal_attention_cuda_forward_no_probs(q, k, v, out, shape, stream);
    if (status == GEO_TENSOR_OK && g_perturbation_delta_host != 0.0f) {
        apply_perturbation_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(out, g_perturbation_delta_host);
    }
    return status;
}

extern "C" geo_tensor_status geo_tensor_causal_attention_cuda_forward_no_probs(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    geo_tensor_attention_shape shape,
    void *stream
) {
    g_n_forward_no_probs_calls++;
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t rows = shape.outer * shape.tokens;
    causal_attention_forward_no_probs_kernel<<<
        launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        q, k, v, out,
        shape.outer, shape.tokens, shape.head_dim
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_causal_attention_cuda_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *probabilities,
    geo_tensor_attention_shape shape,
    void *stream
) {
    g_n_forward_with_probs_calls++;
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
        probabilities == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t rows = shape.outer * shape.tokens;
    causal_attention_forward_kernel<<<
        launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        q, k, v, out, probabilities,
        shape.outer, shape.tokens, shape.head_dim
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_causal_attention_cuda_vjp(
    const float *q,
    const float *k,
    const float *v,
    const float *probabilities,
    const float *grad_out,
    float *grad_q,
    float *grad_k,
    float *grad_v,
    float *workspace_dp_ds,
    geo_attention_backward_timings *timings,
    geo_tensor_attention_shape shape,
    void *stream
) {
    g_n_attention_vjp_calls++;
    if (q == nullptr || k == nullptr || v == nullptr || probabilities == nullptr ||
        grad_out == nullptr || grad_q == nullptr || grad_k == nullptr ||
        grad_v == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);
    const size_t rows = shape.outer * shape.tokens;
    const size_t matrix_elems = shape.outer * shape.tokens * shape.tokens;

    float *ds_ptr = workspace_dp_ds;
    bool allocated_ws = false;

    if (ds_ptr == nullptr) {
        if (cudaMallocAsync(&ds_ptr, 2 * matrix_elems * sizeof(float), custream) != cudaSuccess) {
            return GEO_TENSOR_CUDA_ERROR;
        }
        allocated_ws = true;
    }

    float *ds_workspace = ds_ptr;
    float *ds_workspace_t = ds_ptr + matrix_elems;

    cudaEvent_t ev_start, ev_ds, ev_dq, ev_dk;
    bool do_timing = (timings != nullptr);
    if (do_timing) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_ds);
        cudaEventCreate(&ev_dq);
        cudaEventCreate(&ev_dk);
        cudaEventRecord(ev_start, custream);
    }

    // Step 1: Compute dP and write both dS and transposed dS_t
    attention_vjp_kernel_dp_ds<<<launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0, custream>>>(
        probabilities, grad_out, v, ds_workspace, ds_workspace_t, shape.outer, shape.tokens, shape.head_dim
    );
    if (do_timing) cudaEventRecord(ev_ds, custream);

    // Step 2: Compute grad_q
    attention_vjp_kernel_dq<<<launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0, custream>>>(
        ds_workspace, k, grad_q, shape.outer, shape.tokens, shape.head_dim
    );
    if (do_timing) cudaEventRecord(ev_dq, custream);

    // Step 3: Compute grad_k and grad_v using coalesced ds_workspace_t
    attention_vjp_kernel_dk_dv<<<launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0, custream>>>(
        ds_workspace_t, probabilities, q, grad_out, grad_k, grad_v, shape.outer, shape.tokens, shape.head_dim
    );
    if (do_timing) cudaEventRecord(ev_dk, custream);

    if (do_timing) {
        cudaEventSynchronize(ev_dk);
        cudaEventElapsedTime(&timings->t_dp_ds_ms, ev_start, ev_ds);
        cudaEventElapsedTime(&timings->t_dq_ms, ev_ds, ev_dq);
        cudaEventElapsedTime(&timings->t_dk_dv_ms, ev_dq, ev_dk);
        cudaEventElapsedTime(&timings->t_total_ms, ev_start, ev_dk);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_ds);
        cudaEventDestroy(ev_dq);
        cudaEventDestroy(ev_dk);
    }

    if (allocated_ws) {
        cudaFreeAsync(ds_ptr, custream);
    }

    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_causal_attention_cuda_streaming_vjp(
    const float *q,
    const float *k,
    const float *v,
    const float *out,
    const float *grad_out,
    float *grad_q,
    float *grad_k,
    float *grad_v,
    geo_tensor_attention_shape shape,
    void *stream
) {
    g_n_streaming_vjp_calls++;
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
        grad_out == nullptr || grad_q == nullptr || grad_k == nullptr ||
        grad_v == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);
    const size_t total_elements = shape.outer * shape.tokens * shape.head_dim;

    cudaMemsetAsync(grad_q, 0, total_elements * sizeof(float), custream);
    cudaMemsetAsync(grad_k, 0, total_elements * sizeof(float), custream);
    cudaMemsetAsync(grad_v, 0, total_elements * sizeof(float), custream);

    const size_t rows = shape.outer * shape.tokens;
    causal_attention_streaming_vjp_kernel<<<
        launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0, custream
    >>>(
        q, k, v, out, grad_out, grad_q, grad_k, grad_v,
        shape.outer, shape.tokens, shape.head_dim
    );

    return launch_status();
}
