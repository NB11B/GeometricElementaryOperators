#include "geo/tensor_embedding.h"

#include <stdint.h>

static int geo_tensor_embedding_shape_valid(geo_tensor_embedding_shape shape) {
    if (shape.indices == 0u || shape.vocabulary == 0u || shape.dimension == 0u) {
        return 0;
    }
    return shape.vocabulary <= SIZE_MAX / shape.dimension &&
           shape.indices <= SIZE_MAX / shape.dimension;
}

geo_tensor_status geo_tensor_embedding_forward(
    const int64_t *token_indices,
    const geo_real_t *weight,
    geo_real_t *out,
    geo_tensor_embedding_shape shape
) {
    if (token_indices == NULL || weight == NULL || out == NULL ||
        !geo_tensor_embedding_shape_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    for (size_t index = 0u; index < shape.indices; ++index) {
        const int64_t token = token_indices[index];
        if (token < 0 || (uint64_t)token >= (uint64_t)shape.vocabulary) {
            return GEO_TENSOR_INVALID_ARGUMENT;
        }
    }
    for (size_t index = 0u; index < shape.indices; ++index) {
        const size_t token = (size_t)token_indices[index];
        const size_t source = token * shape.dimension;
        const size_t destination = index * shape.dimension;
        for (size_t dim = 0u; dim < shape.dimension; ++dim) {
            out[destination + dim] = weight[source + dim];
        }
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_embedding_vjp(
    const int64_t *token_indices,
    const geo_real_t *grad_out,
    geo_real_t *grad_weight,
    geo_tensor_embedding_shape shape
) {
    if (token_indices == NULL || grad_out == NULL || grad_weight == NULL ||
        !geo_tensor_embedding_shape_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    for (size_t index = 0u; index < shape.indices; ++index) {
        const int64_t token = token_indices[index];
        if (token < 0 || (uint64_t)token >= (uint64_t)shape.vocabulary) {
            return GEO_TENSOR_INVALID_ARGUMENT;
        }
    }
    const size_t weight_count = shape.vocabulary * shape.dimension;
    for (size_t element = 0u; element < weight_count; ++element) {
        grad_weight[element] = (geo_real_t)0;
    }
    for (size_t index = 0u; index < shape.indices; ++index) {
        const size_t token = (size_t)token_indices[index];
        const size_t source = index * shape.dimension;
        const size_t destination = token * shape.dimension;
        for (size_t dim = 0u; dim < shape.dimension; ++dim) {
            grad_weight[destination + dim] += grad_out[source + dim];
        }
    }
    return GEO_TENSOR_OK;
}
