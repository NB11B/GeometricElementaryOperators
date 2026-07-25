#include "geo/tensor_attention_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

constexpr unsigned int GEO_ATTENTION_BLOCK_SIZE = 128u;
constexpr unsigned int GEO_ATTENTION_MAX_BLOCKS = 65535u;

bool valid_shape(geo_tensor_attention_shape shape) {
    if (shape.outer == 0u || shape.tokens == 0u || shape.head_dim == 0u) {
        return false;
    }
    if (shape.outer > SIZE_MAX / shape.tokens) {
        return false;
    }
    const size_t rows = shape.outer * shape.tokens;
    if (rows > SIZE_MAX / shape.head_dim || rows > SIZE_MAX / shape.tokens) {
        return false;
    }
    return true;
}

unsigned int launch_blocks(size_t rows) {
    size_t blocks = (rows + GEO_ATTENTION_BLOCK_SIZE - 1u) / GEO_ATTENTION_BLOCK_SIZE;
    if (blocks > GEO_ATTENTION_MAX_BLOCKS) {
        blocks = GEO_ATTENTION_MAX_BLOCKS;
    }
    return static_cast<unsigned int>(blocks);
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

__global__ void causal_attention_vjp_kernel(
    const float *q,
    const float *k,
    const float *v,
    const float *probabilities,
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

        for (size_t dim = 0u; dim < head_dim; ++dim) {
            grad_q[data_index(outer, query, dim, tokens, head_dim)] = 0.0f;
        }

        float softmax_dot = 0.0f;
        for (size_t key = 0u; key <= query; ++key) {
            const float d_probability = dot_tokens(
                grad_out, query, v, key, outer, tokens, head_dim
            );
            softmax_dot += probabilities[
                probability_index(outer, query, key, tokens)
            ] * d_probability;
        }

        for (size_t key = 0u; key <= query; ++key) {
            const float probability = probabilities[
                probability_index(outer, query, key, tokens)
            ];
            const float d_probability = dot_tokens(
                grad_out, query, v, key, outer, tokens, head_dim
            );
            const float d_score = probability * (d_probability - softmax_dot);

            for (size_t dim = 0u; dim < head_dim; ++dim) {
                const size_t query_index =
                    data_index(outer, query, dim, tokens, head_dim);
                const size_t key_index =
                    data_index(outer, key, dim, tokens, head_dim);
                grad_q[query_index] += scale * d_score * k[key_index];
                atomicAdd(&grad_k[key_index], scale * d_score * q[query_index]);
                atomicAdd(&grad_v[key_index], probability * grad_out[query_index]);
            }
        }
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

}  // namespace

extern "C" geo_tensor_status geo_tensor_causal_attention_cuda_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *probabilities,
    geo_tensor_attention_shape shape,
    void *stream
) {
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
    geo_tensor_attention_shape shape,
    void *stream
) {
    if (q == nullptr || k == nullptr || v == nullptr || probabilities == nullptr ||
        grad_out == nullptr || grad_q == nullptr || grad_k == nullptr || grad_v == nullptr ||
        !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t data_count = shape.outer * shape.tokens * shape.head_dim;
    const size_t bytes = data_count * sizeof(float);
    if (cudaMemsetAsync(grad_k, 0, bytes, cuda_stream) != cudaSuccess ||
        cudaMemsetAsync(grad_v, 0, bytes, cuda_stream) != cudaSuccess) {
        return GEO_TENSOR_CUDA_ERROR;
    }

    const size_t rows = shape.outer * shape.tokens;
    causal_attention_vjp_kernel<<<
        launch_blocks(rows), GEO_ATTENTION_BLOCK_SIZE, 0, cuda_stream
    >>>(
        q, k, v, probabilities, grad_out,
        grad_q, grad_k, grad_v,
        shape.outer, shape.tokens, shape.head_dim
    );
    return launch_status();
}
