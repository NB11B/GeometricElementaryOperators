#ifndef GEO_TENSOR_ROPE_H
#define GEO_TENSOR_ROPE_H

#include <stddef.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct geo_tensor_rope_table_shape {
    size_t seq_len;
    size_t head_dim;
} geo_tensor_rope_table_shape;

typedef struct geo_tensor_rope_apply_shape {
    size_t outer;
    size_t tokens;
    size_t head_dim;
    size_t table_seq_len;
} geo_tensor_rope_apply_shape;

/*
 * Build split-half rotary tables with shape [seq_len, head_dim / 2]:
 *   inv_freq[j] = theta ^ (-(2*j) / head_dim)
 *   angle[p,j] = p * inv_freq[j]
 */
geo_tensor_status geo_tensor_rope_build(
    geo_real_t theta,
    geo_real_t *cos_out,
    geo_real_t *sin_out,
    geo_tensor_rope_table_shape shape
);

/*
 * Apply split-half rotation to row-major [outer, tokens, head_dim]:
 *   y1 = x1 * cos - x2 * sin
 *   y2 = x2 * cos + x1 * sin
 */
geo_tensor_status geo_tensor_rope_apply_forward(
    const geo_real_t *x,
    const geo_real_t *cos_table,
    const geo_real_t *sin_table,
    geo_real_t *out,
    geo_tensor_rope_apply_shape shape
);

/*
 * VJP with respect to x. RoPE tables are treated as constant buffers:
 *   grad_x1 = grad_y1 * cos + grad_y2 * sin
 *   grad_x2 = -grad_y1 * sin + grad_y2 * cos
 */
geo_tensor_status geo_tensor_rope_apply_vjp(
    const geo_real_t *grad_out,
    const geo_real_t *cos_table,
    const geo_real_t *sin_table,
    geo_real_t *grad_x,
    geo_tensor_rope_apply_shape shape
);

#ifdef __cplusplus
}
#endif

#endif
