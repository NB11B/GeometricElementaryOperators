#include "geo/tensor_activation.h"

#include <math.h>

static int geo_tensor_activation_valid(const void *a, const void *b, size_t count) {
    return a != NULL && b != NULL && count > 0u;
}

static geo_real_t geo_sigmoid(geo_real_t x) {
    return (geo_real_t)1 / ((geo_real_t)1 + (geo_real_t)exp((double)-x));
}

geo_tensor_status geo_tensor_gelu_forward(
    const geo_real_t *x,
    geo_real_t *out,
    size_t count
) {
    if (!geo_tensor_activation_valid(x, out, count)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const geo_real_t inv_sqrt2 = (geo_real_t)0.70710678118654752440;
    for (size_t i = 0; i < count; ++i) {
        out[i] = (geo_real_t)0.5 * x[i] * ((geo_real_t)1 + (geo_real_t)erf((double)(x[i] * inv_sqrt2)));
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_gelu_vjp(
    const geo_real_t *x,
    const geo_real_t *grad_out,
    geo_real_t *grad_x,
    size_t count
) {
    if (x == NULL || grad_out == NULL || grad_x == NULL || count == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const geo_real_t inv_sqrt2 = (geo_real_t)0.70710678118654752440;
    const geo_real_t inv_sqrt_2pi = (geo_real_t)0.39894228040143267794;
    for (size_t i = 0; i < count; ++i) {
        const geo_real_t xi = x[i];
        const geo_real_t cdf = (geo_real_t)0.5 * ((geo_real_t)1 + (geo_real_t)erf((double)(xi * inv_sqrt2)));
        const geo_real_t pdf_term = xi * inv_sqrt_2pi * (geo_real_t)exp((double)((geo_real_t)-0.5 * xi * xi));
        grad_x[i] = grad_out[i] * (cdf + pdf_term);
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_silu_mul_forward(
    const geo_real_t *gate,
    const geo_real_t *up,
    geo_real_t *out,
    size_t count
) {
    if (gate == NULL || up == NULL || out == NULL || count == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; ++i) {
        const geo_real_t sigmoid = geo_sigmoid(gate[i]);
        out[i] = gate[i] * sigmoid * up[i];
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_silu_mul_vjp(
    const geo_real_t *gate,
    const geo_real_t *up,
    const geo_real_t *grad_out,
    geo_real_t *grad_gate,
    geo_real_t *grad_up,
    size_t count
) {
    if (gate == NULL || up == NULL || grad_out == NULL || grad_gate == NULL || grad_up == NULL || count == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; ++i) {
        const geo_real_t g = gate[i];
        const geo_real_t sigmoid = geo_sigmoid(g);
        const geo_real_t silu = g * sigmoid;
        const geo_real_t dsilu = sigmoid * ((geo_real_t)1 + g * ((geo_real_t)1 - sigmoid));
        grad_gate[i] = grad_out[i] * up[i] * dsilu;
        grad_up[i] = grad_out[i] * silu;
    }
    return GEO_TENSOR_OK;
}
