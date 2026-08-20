#include "geo/tensor_embedding_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr unsigned int GEO_EMBEDDING_BLOCK_SIZE = 256u;
constexpr unsigned int GEO_EMBEDDING_MAX_BLOCKS = 65535u;

bool valid_shape(geo_tensor_embedding_shape shape) {
    return shape.indices > 0u && shape.vocabulary > 0u && shape.dimension > 0u &&
           shape.vocabulary <= SIZE_MAX / shape.dimension &&
           shape.indices <= SIZE_MAX / shape.dimension;
}

unsigned int launch_blocks(size_t work_items) {
    size_t blocks = (work_items + GEO_EMBEDDING_BLOCK_SIZE - 1u) /
                    GEO_EMBEDDING_BLOCK_SIZE;
    if (blocks > GEO_EMBEDDING_MAX_BLOCKS) {
        blocks = GEO_EMBEDDING_MAX_BLOCKS;
    }
    return static_cast<unsigned int>(blocks);
}

__global__ void embedding_forward_kernel(
    const int64_t *token_indices,
    const float *weight,
    float *out,
    size_t indices,
    size_t vocabulary,
    size_t dimension
) {
    const size_t total = indices * dimension;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < total;
         element += stride) {
        const size_t index = element / dimension;
        const size_t dim = element - index * dimension;
        const int64_t token = token_indices[index];
        if (token < 0 || static_cast<uint64_t>(token) >= static_cast<uint64_t>(vocabulary)) {
            out[element] = nanf("");
        } else {
            out[element] = weight[static_cast<size_t>(token) * dimension + dim];
        }
    }
}

__global__ void embedding_vjp_kernel(
    const int64_t *token_indices,
    const float *grad_out,
    float *grad_weight,
    size_t indices,
    size_t vocabulary,
    size_t dimension
) {
    const size_t total = indices * dimension;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < total;
         element += stride) {
        const size_t index = element / dimension;
        const size_t dim = element - index * dimension;
        const int64_t token = token_indices[index];
        if (token >= 0 && static_cast<uint64_t>(token) < static_cast<uint64_t>(vocabulary)) {
            atomicAdd(
                &grad_weight[static_cast<size_t>(token) * dimension + dim],
                grad_out[element]
            );
        }
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

}  // namespace

extern "C" geo_tensor_status geo_tensor_embedding_cuda_forward(
    const int64_t *token_indices,
    const float *weight,
    float *out,
    geo_tensor_embedding_shape shape,
    void *stream
) {
    if (token_indices == nullptr || weight == nullptr || out == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t total = shape.indices * shape.dimension;
    embedding_forward_kernel<<<
        launch_blocks(total), GEO_EMBEDDING_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        token_indices, weight, out,
        shape.indices, shape.vocabulary, shape.dimension
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_embedding_cuda_vjp(
    const int64_t *token_indices,
    const float *grad_out,
    float *grad_weight,
    geo_tensor_embedding_shape shape,
    void *stream
) {
    if (token_indices == nullptr || grad_out == nullptr || grad_weight == nullptr ||
        !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t weight_count = shape.vocabulary * shape.dimension;
    if (cudaMemsetAsync(
            grad_weight, 0, weight_count * sizeof(float), cuda_stream
        ) != cudaSuccess) {
        return GEO_TENSOR_CUDA_ERROR;
    }
    const size_t total = shape.indices * shape.dimension;
    embedding_vjp_kernel<<<
        launch_blocks(total), GEO_EMBEDDING_BLOCK_SIZE, 0, cuda_stream
    >>>(
        token_indices, grad_out, grad_weight,
        shape.indices, shape.vocabulary, shape.dimension
    );
    return launch_status();
}
