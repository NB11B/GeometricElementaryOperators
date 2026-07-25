#include "geo/tensor_linear.h"

#include <stdint.h>
#include <string.h>

static int geo_tensor_mul_overflows(size_t a, size_t b) {
    return a != 0u && b > SIZE_MAX / a;
}

static geo_tensor_status geo_tensor_validate_shape(geo_tensor_linear_shape shape) {
    if (shape.rows == 0u || shape.in_features == 0u || shape.out_features == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    if (geo_tensor_mul_overflows(shape.rows, shape.in_features) ||
        geo_tensor_mul_overflows(shape.rows, shape.out_features) ||
        geo_tensor_mul_overflows(shape.out_features, shape.in_features)) {
        return GEO_TENSOR_OVERFLOW;
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_linear_forward(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    const geo_tensor_linear_shape *shape
) {
    geo_tensor_status status;
    size_t row;
    size_t out;
    size_t in;

    if (x == NULL || weight == NULL || y == NULL || shape == NULL) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    status = geo_tensor_validate_shape(*shape);
    if (status != GEO_TENSOR_OK) {
        return status;
    }

    for (row = 0u; row < shape->rows; ++row) {
        const geo_real_t *x_row = x + row * shape->in_features;
        geo_real_t *y_row = y + row * shape->out_features;
        for (out = 0u; out < shape->out_features; ++out) {
            const geo_real_t *w_row = weight + out * shape->in_features;
            geo_real_t sum = (geo_real_t)0;
            for (in = 0u; in < shape->in_features; ++in) {
                sum += x_row[in] * w_row[in];
            }
            y_row[out] = sum;
        }
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_linear_vjp(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    const geo_tensor_linear_shape *shape
) {
    geo_tensor_status status;
    size_t row;
    size_t out;
    size_t in;
    size_t grad_x_count;
    size_t grad_weight_count;

    if (x == NULL || weight == NULL || grad_y == NULL || grad_x == NULL || grad_weight == NULL || shape == NULL) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    status = geo_tensor_validate_shape(*shape);
    if (status != GEO_TENSOR_OK) {
        return status;
    }

    grad_x_count = shape->rows * shape->in_features;
    grad_weight_count = shape->out_features * shape->in_features;
    memset(grad_x, 0, grad_x_count * sizeof(*grad_x));
    memset(grad_weight, 0, grad_weight_count * sizeof(*grad_weight));

    for (row = 0u; row < shape->rows; ++row) {
        const geo_real_t *x_row = x + row * shape->in_features;
        const geo_real_t *gy_row = grad_y + row * shape->out_features;
        geo_real_t *gx_row = grad_x + row * shape->in_features;
        for (out = 0u; out < shape->out_features; ++out) {
            const geo_real_t gy = gy_row[out];
            const geo_real_t *w_row = weight + out * shape->in_features;
            geo_real_t *gw_row = grad_weight + out * shape->in_features;
            for (in = 0u; in < shape->in_features; ++in) {
                gx_row[in] += gy * w_row[in];
                gw_row[in] += gy * x_row[in];
            }
        }
    }
    return GEO_TENSOR_OK;
}

const char *geo_tensor_status_string(geo_tensor_status status) {
    switch (status) {
        case GEO_TENSOR_OK:
            return "ok";
        case GEO_TENSOR_INVALID_ARGUMENT:
            return "invalid argument";
        case GEO_TENSOR_OVERFLOW:
            return "size overflow";
        case GEO_TENSOR_CUDA_ERROR:
            return "cuda error";
        case GEO_TENSOR_UNSUPPORTED:
            return "unsupported";
        default:
            return "unknown tensor status";
    }
}
