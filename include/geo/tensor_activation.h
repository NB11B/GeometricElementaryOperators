#ifndef GEO_TENSOR_ACTIVATION_H
#define GEO_TENSOR_ACTIVATION_H

#include <stddef.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_gelu_forward(
    const geo_real_t *x,
    geo_real_t *out,
    size_t count
);

geo_tensor_status geo_tensor_gelu_vjp(
    const geo_real_t *x,
    const geo_real_t *grad_out,
    geo_real_t *grad_x,
    size_t count
);

geo_tensor_status geo_tensor_silu_mul_forward(
    const geo_real_t *gate,
    const geo_real_t *up,
    geo_real_t *out,
    size_t count
);

geo_tensor_status geo_tensor_silu_mul_vjp(
    const geo_real_t *gate,
    const geo_real_t *up,
    const geo_real_t *grad_out,
    geo_real_t *grad_gate,
    geo_real_t *grad_up,
    size_t count
);

#ifdef __cplusplus
}
#endif

#endif
