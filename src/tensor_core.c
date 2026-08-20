#include "geo/tensor_core.h"

#include <math.h>
#include <string.h>

static int geo_tensor_core_valid(const void *a, const void *out, size_t count) {
    return a != NULL && out != NULL && count > 0u;
}

geo_tensor_status geo_tensor_add_forward(
    const geo_real_t *a, const geo_real_t *b, geo_real_t *out, size_t count
) {
    size_t i;
    if (!geo_tensor_core_valid(a, out, count) || b == NULL) return GEO_TENSOR_INVALID_ARGUMENT;
    for (i = 0u; i < count; ++i) out[i] = a[i] + b[i];
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_add_vjp(
    const geo_real_t *grad_out, geo_real_t *grad_a, geo_real_t *grad_b, size_t count
) {
    if (!geo_tensor_core_valid(grad_out, grad_a, count) || grad_b == NULL) return GEO_TENSOR_INVALID_ARGUMENT;
    memcpy(grad_a, grad_out, count * sizeof(*grad_a));
    memcpy(grad_b, grad_out, count * sizeof(*grad_b));
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_mul_forward(
    const geo_real_t *a, const geo_real_t *b, geo_real_t *out, size_t count
) {
    size_t i;
    if (!geo_tensor_core_valid(a, out, count) || b == NULL) return GEO_TENSOR_INVALID_ARGUMENT;
    for (i = 0u; i < count; ++i) out[i] = a[i] * b[i];
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_mul_vjp(
    const geo_real_t *a, const geo_real_t *b, const geo_real_t *grad_out,
    geo_real_t *grad_a, geo_real_t *grad_b, size_t count
) {
    size_t i;
    if (a == NULL || b == NULL || grad_out == NULL || grad_a == NULL || grad_b == NULL || count == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    for (i = 0u; i < count; ++i) {
        grad_a[i] = grad_out[i] * b[i];
        grad_b[i] = grad_out[i] * a[i];
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_scale_forward(
    const geo_real_t *x, geo_real_t scalar, geo_real_t *out, size_t count
) {
    size_t i;
    if (!geo_tensor_core_valid(x, out, count)) return GEO_TENSOR_INVALID_ARGUMENT;
    for (i = 0u; i < count; ++i) out[i] = x[i] * scalar;
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_scale_vjp(
    const geo_real_t *grad_out, geo_real_t scalar, geo_real_t *grad_x, size_t count
) {
    return geo_tensor_scale_forward(grad_out, scalar, grad_x, count);
}

geo_tensor_status geo_tensor_rms_norm_forward(
    const geo_real_t *x, const geo_real_t *weight, geo_real_t epsilon,
    geo_real_t *out, geo_real_t *inv_rms, geo_tensor_norm_shape shape
) {
    size_t row, feature;
    if (x == NULL || weight == NULL || out == NULL || shape.rows == 0u || shape.features == 0u || epsilon <= (geo_real_t)0) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    for (row = 0u; row < shape.rows; ++row) {
        geo_real_t sum_sq = (geo_real_t)0;
        geo_real_t inverse;
        const size_t base = row * shape.features;
        for (feature = 0u; feature < shape.features; ++feature) {
            const geo_real_t value = x[base + feature];
            sum_sq += value * value;
        }
        inverse = (geo_real_t)1 / (geo_real_t)sqrt((double)(sum_sq / (geo_real_t)shape.features + epsilon));
        if (inv_rms != NULL) inv_rms[row] = inverse;
        for (feature = 0u; feature < shape.features; ++feature) {
            out[base + feature] = x[base + feature] * inverse * weight[feature];
        }
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_rms_norm_vjp(
    const geo_real_t *x, const geo_real_t *weight, const geo_real_t *grad_out,
    const geo_real_t *inv_rms, geo_real_t *grad_x, geo_real_t *grad_weight,
    geo_tensor_norm_shape shape
) {
    size_t row, feature;
    if (x == NULL || weight == NULL || grad_out == NULL || inv_rms == NULL || grad_x == NULL || grad_weight == NULL || shape.rows == 0u || shape.features == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    memset(grad_weight, 0, shape.features * sizeof(*grad_weight));
    for (row = 0u; row < shape.rows; ++row) {
        const size_t base = row * shape.features;
        const geo_real_t inverse = inv_rms[row];
        geo_real_t mean_ux = (geo_real_t)0;
        for (feature = 0u; feature < shape.features; ++feature) {
            const geo_real_t u = grad_out[base + feature] * weight[feature];
            mean_ux += u * x[base + feature];
            grad_weight[feature] += grad_out[base + feature] * x[base + feature] * inverse;
        }
        mean_ux /= (geo_real_t)shape.features;
        for (feature = 0u; feature < shape.features; ++feature) {
            const geo_real_t u = grad_out[base + feature] * weight[feature];
            grad_x[base + feature] = inverse * u - x[base + feature] * inverse * inverse * inverse * mean_ux;
        }
    }
    return GEO_TENSOR_OK;
}
