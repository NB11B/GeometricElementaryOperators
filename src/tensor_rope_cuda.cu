#include "geo/tensor_rope_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr unsigned int GEO_ROPE_BLOCK_SIZE = 256u;
constexpr unsigned int GEO_ROPE_MAX_BLOCKS = 65535u;

bool valid_table_shape(geo_tensor_rope_table_shape shape) {
    if (shape.seq_len == 0u || shape.head_dim == 0u || (shape.head_dim & 1u) != 0u) {
        return false;
    }
    const size_t half = shape.head_dim / 2u;
    return half > 0u && shape.seq_len <= SIZE_MAX / half;
}

bool valid_apply_shape(geo_tensor_rope_apply_shape shape) {
    if (shape.outer == 0u || shape.tokens == 0u || shape.head_dim == 0u ||
        shape.table_seq_len < shape.tokens || (shape.head_dim & 1u) != 0u) {
        return false;
    }
    const size_t half = shape.head_dim / 2u;
    if (half == 0u || shape.table_seq_len > SIZE_MAX / half) {
        return false;
    }
    if (shape.outer > SIZE_MAX / shape.tokens) {
        return false;
    }
    const size_t rows = shape.outer * shape.tokens;
    return rows <= SIZE_MAX / shape.head_dim;
}

unsigned int launch_blocks(size_t work_items) {
    size_t blocks = (work_items + GEO_ROPE_BLOCK_SIZE - 1u) / GEO_ROPE_BLOCK_SIZE;
    if (blocks > GEO_ROPE_MAX_BLOCKS) {
        blocks = GEO_ROPE_MAX_BLOCKS;
    }
    return static_cast<unsigned int>(blocks);
}

__global__ void rope_build_kernel(
    float theta,
    float *cos_out,
    float *sin_out,
    size_t seq_len,
    size_t head_dim
) {
    const size_t half = head_dim / 2u;
    const size_t total = seq_len * half;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += stride) {
        const size_t position = index / half;
        const size_t pair = index - position * half;
        const float exponent = -static_cast<float>(2u * pair) / static_cast<float>(head_dim);
        const float angle = static_cast<float>(position) * powf(theta, exponent);
        cos_out[index] = cosf(angle);
        sin_out[index] = sinf(angle);
    }
}

__global__ void rope_apply_kernel(
    const float *x,
    const float *cos_table,
    const float *sin_table,
    float *out,
    size_t outer,
    size_t tokens,
    size_t head_dim
) {
    const size_t half = head_dim / 2u;
    const size_t total_pairs = outer * tokens * half;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_pairs;
         index += stride) {
        const size_t pair = index % half;
        const size_t row = index / half;
        const size_t token = row % tokens;
        const size_t data_base = row * head_dim;
        const size_t table_index = token * half + pair;
        const float x1 = x[data_base + pair];
        const float x2 = x[data_base + half + pair];
        const float c = cos_table[table_index];
        const float s = sin_table[table_index];
        out[data_base + pair] = x1 * c - x2 * s;
        out[data_base + half + pair] = x2 * c + x1 * s;
    }
}

__global__ void rope_vjp_kernel(
    const float *grad_out,
    const float *cos_table,
    const float *sin_table,
    float *grad_x,
    size_t outer,
    size_t tokens,
    size_t head_dim
) {
    const size_t half = head_dim / 2u;
    const size_t total_pairs = outer * tokens * half;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total_pairs;
         index += stride) {
        const size_t pair = index % half;
        const size_t row = index / half;
        const size_t token = row % tokens;
        const size_t data_base = row * head_dim;
        const size_t table_index = token * half + pair;
        const float grad_y1 = grad_out[data_base + pair];
        const float grad_y2 = grad_out[data_base + half + pair];
        const float c = cos_table[table_index];
        const float s = sin_table[table_index];
        grad_x[data_base + pair] = grad_y1 * c + grad_y2 * s;
        grad_x[data_base + half + pair] = -grad_y1 * s + grad_y2 * c;
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

}  // namespace

extern "C" geo_tensor_status geo_tensor_rope_cuda_build(
    float theta,
    float *cos_out,
    float *sin_out,
    geo_tensor_rope_table_shape shape,
    void *stream
) {
    if (cos_out == nullptr || sin_out == nullptr || !(theta > 0.0f) ||
        !valid_table_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t total = shape.seq_len * (shape.head_dim / 2u);
    rope_build_kernel<<<launch_blocks(total), GEO_ROPE_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        theta, cos_out, sin_out, shape.seq_len, shape.head_dim
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_rope_cuda_apply_forward(
    const float *x,
    const float *cos_table,
    const float *sin_table,
    float *out,
    geo_tensor_rope_apply_shape shape,
    void *stream
) {
    if (x == nullptr || cos_table == nullptr || sin_table == nullptr || out == nullptr ||
        !valid_apply_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t total_pairs = shape.outer * shape.tokens * (shape.head_dim / 2u);
    rope_apply_kernel<<<launch_blocks(total_pairs), GEO_ROPE_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        x, cos_table, sin_table, out, shape.outer, shape.tokens, shape.head_dim
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_rope_cuda_apply_vjp(
    const float *grad_out,
    const float *cos_table,
    const float *sin_table,
    float *grad_x,
    geo_tensor_rope_apply_shape shape,
    void *stream
) {
    if (grad_out == nullptr || cos_table == nullptr || sin_table == nullptr || grad_x == nullptr ||
        !valid_apply_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t total_pairs = shape.outer * shape.tokens * (shape.head_dim / 2u);
    rope_vjp_kernel<<<launch_blocks(total_pairs), GEO_ROPE_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        grad_out, cos_table, sin_table, grad_x, shape.outer, shape.tokens, shape.head_dim
    );
    return launch_status();
}
