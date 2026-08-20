#ifndef GEO_TENSOR_CORE_H
#define GEO_TENSOR_CORE_H

#include <stddef.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocation-free elementwise tensor primitives. */
geo_tensor_status geo_tensor_add_forward(
    const geo_real_t *a,
    const geo_real_t *b,
    geo_real_t *out,
    size_t count
);

geo_tensor_status geo_tensor_add_vjp(
    const geo_real_t *grad_out,
    geo_real_t *grad_a,
    geo_real_t *grad_b,
    size_t count
);

geo_tensor_status geo_tensor_mul_forward(
    const geo_real_t *a,
    const geo_real_t *b,
    geo_real_t *out,
    size_t count
);

geo_tensor_status geo_tensor_mul_vjp(
    const geo_real_t *a,
    const geo_real_t *b,
    const geo_real_t *grad_out,
    geo_real_t *grad_a,
    geo_real_t *grad_b,
    size_t count
);

geo_tensor_status geo_tensor_scale_forward(
    const geo_real_t *x,
    geo_real_t scalar,
    geo_real_t *out,
    size_t count
);

geo_tensor_status geo_tensor_scale_vjp(
    const geo_real_t *grad_out,
    geo_real_t scalar,
    geo_real_t *grad_x,
    size_t count
);

/*
 * RMSNorm over the last dimension of a row-major [rows, features] tensor.
 * inv_rms is optional in forward, but supplying it avoids recomputation in VJP.
 */
typedef struct geo_tensor_norm_shape {
    size_t rows;
    size_t features;
} geo_tensor_norm_shape;

geo_tensor_status geo_tensor_rms_norm_forward(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t epsilon,
    geo_real_t *out,
    geo_real_t *inv_rms,
    geo_tensor_norm_shape shape
);

geo_tensor_status geo_tensor_rms_norm_vjp(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_out,
    const geo_real_t *inv_rms,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    geo_tensor_norm_shape shape
);

#ifdef __cplusplus
}
#endif

#endif
