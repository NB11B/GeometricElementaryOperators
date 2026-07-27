#ifndef GEO_TENSOR_EMBEDDING_H
#define GEO_TENSOR_EMBEDDING_H

#include <stddef.h>
#include <stdint.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct geo_tensor_embedding_shape {
    size_t indices;
    size_t vocabulary;
    size_t dimension;
} geo_tensor_embedding_shape;

/* Row lookup: out[index, dim] = weight[token[index], dim]. */
geo_tensor_status geo_tensor_embedding_forward(
    const int64_t *token_indices,
    const geo_real_t *weight,
    geo_real_t *out,
    geo_tensor_embedding_shape shape
);

/* Scatter-add VJP into grad_weight. The output buffer is overwritten. */
geo_tensor_status geo_tensor_embedding_vjp(
    const int64_t *token_indices,
    const geo_real_t *grad_out,
    geo_real_t *grad_weight,
    geo_tensor_embedding_shape shape
);

#ifdef __cplusplus
}
#endif

#endif
