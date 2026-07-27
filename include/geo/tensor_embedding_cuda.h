#ifndef GEO_TENSOR_EMBEDDING_CUDA_H
#define GEO_TENSOR_EMBEDDING_CUDA_H

#include "geo/tensor_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_embedding_cuda_forward(
    const int64_t *token_indices,
    const float *weight,
    float *out,
    geo_tensor_embedding_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_embedding_cuda_vjp(
    const int64_t *token_indices,
    const float *grad_out,
    float *grad_weight,
    geo_tensor_embedding_shape shape,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
