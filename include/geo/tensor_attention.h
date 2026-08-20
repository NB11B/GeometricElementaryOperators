#ifndef GEO_TENSOR_ATTENTION_H
#define GEO_TENSOR_ATTENTION_H

#include <stddef.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct geo_tensor_attention_shape {
    size_t outer;
    size_t tokens;
    size_t head_dim;
} geo_tensor_attention_shape;

/*
 * Causal scaled dot-product attention over row-major tensors shaped
 * [outer, tokens, head_dim]. probabilities is [outer, tokens, tokens].
 * Masked entries are written as zero.
 */
geo_tensor_status geo_tensor_causal_attention_forward(
    const geo_real_t *q,
    const geo_real_t *k,
    const geo_real_t *v,
    geo_real_t *out,
    geo_real_t *probabilities,
    geo_tensor_attention_shape shape
);

/*
 * Analytic VJP for q, k, and v using probabilities saved by forward.
 * All gradient buffers are overwritten, not accumulated.
 */
geo_tensor_status geo_tensor_causal_attention_vjp(
    const geo_real_t *q,
    const geo_real_t *k,
    const geo_real_t *v,
    const geo_real_t *probabilities,
    const geo_real_t *grad_out,
    geo_real_t *grad_q,
    geo_real_t *grad_k,
    geo_real_t *grad_v,
    geo_tensor_attention_shape shape
);

#ifdef __cplusplus
}
#endif

#endif
