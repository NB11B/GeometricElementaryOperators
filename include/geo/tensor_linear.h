#ifndef GEO_TENSOR_LINEAR_H
#define GEO_TENSOR_LINEAR_H

#include <stddef.h>

#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum geo_tensor_status {
    GEO_TENSOR_OK = 0,
    GEO_TENSOR_INVALID_ARGUMENT = 1,
    GEO_TENSOR_OVERFLOW = 2,
    GEO_TENSOR_CUDA_ERROR = 3,
    GEO_TENSOR_UNSUPPORTED = 4
} geo_tensor_status;

typedef struct geo_tensor_linear_shape {
    size_t rows;
    size_t in_features;
    size_t out_features;
} geo_tensor_linear_shape;

/*
 * Row-major dense linear operator:
 *   y[row, out] = sum_in x[row, in] * weight[out, in]
 *
 * This API is intentionally allocation-free. The caller owns every buffer.
 */
geo_tensor_status geo_tensor_linear_forward(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    const geo_tensor_linear_shape *shape
);

/*
 * Vector-Jacobian product for the same operator:
 *   grad_x[row, in] = sum_out grad_y[row, out] * weight[out, in]
 *   grad_weight[out, in] = sum_row grad_y[row, out] * x[row, in]
 *
 * grad_x and grad_weight are overwritten, not accumulated.
 */
geo_tensor_status geo_tensor_linear_vjp(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    const geo_tensor_linear_shape *shape
);

const char *geo_tensor_status_string(geo_tensor_status status);

#ifdef __cplusplus
}
#endif

#endif
