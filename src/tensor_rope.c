#include "geo/tensor_rope.h"

#include <math.h>
#include <stdint.h>

static int geo_tensor_rope_table_valid(geo_tensor_rope_table_shape shape) {
    if (shape.seq_len == 0u || shape.head_dim == 0u || (shape.head_dim & 1u) != 0u) {
        return 0;
    }
    const size_t half = shape.head_dim / 2u;
    return half > 0u && shape.seq_len <= SIZE_MAX / half;
}

static int geo_tensor_rope_apply_valid(geo_tensor_rope_apply_shape shape) {
    if (shape.outer == 0u || shape.tokens == 0u || shape.head_dim == 0u ||
        shape.table_seq_len < shape.tokens || (shape.head_dim & 1u) != 0u) {
        return 0;
    }
    const size_t half = shape.head_dim / 2u;
    if (half == 0u || shape.table_seq_len > SIZE_MAX / half) {
        return 0;
    }
    if (shape.outer > SIZE_MAX / shape.tokens) {
        return 0;
    }
    const size_t rows = shape.outer * shape.tokens;
    return rows <= SIZE_MAX / shape.head_dim;
}

geo_tensor_status geo_tensor_rope_build(
    geo_real_t theta,
    geo_real_t *cos_out,
    geo_real_t *sin_out,
    geo_tensor_rope_table_shape shape
) {
    if (cos_out == NULL || sin_out == NULL || !(theta > (geo_real_t)0) ||
        !geo_tensor_rope_table_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    const size_t half = shape.head_dim / 2u;
    for (size_t position = 0u; position < shape.seq_len; ++position) {
        for (size_t pair = 0u; pair < half; ++pair) {
            const geo_real_t exponent =
                -((geo_real_t)(2u * pair) / (geo_real_t)shape.head_dim);
            const geo_real_t inv_frequency =
                (geo_real_t)pow((double)theta, (double)exponent);
            const geo_real_t angle = (geo_real_t)position * inv_frequency;
            const size_t index = position * half + pair;
            cos_out[index] = (geo_real_t)cos((double)angle);
            sin_out[index] = (geo_real_t)sin((double)angle);
        }
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_rope_apply_forward(
    const geo_real_t *x,
    const geo_real_t *cos_table,
    const geo_real_t *sin_table,
    geo_real_t *out,
    geo_tensor_rope_apply_shape shape
) {
    if (x == NULL || cos_table == NULL || sin_table == NULL || out == NULL ||
        !geo_tensor_rope_apply_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    const size_t half = shape.head_dim / 2u;
    for (size_t outer = 0u; outer < shape.outer; ++outer) {
        for (size_t token = 0u; token < shape.tokens; ++token) {
            const size_t row = outer * shape.tokens + token;
            const size_t data_base = row * shape.head_dim;
            const size_t table_base = token * half;
            for (size_t pair = 0u; pair < half; ++pair) {
                const geo_real_t x1 = x[data_base + pair];
                const geo_real_t x2 = x[data_base + half + pair];
                const geo_real_t c = cos_table[table_base + pair];
                const geo_real_t s = sin_table[table_base + pair];
                out[data_base + pair] = x1 * c - x2 * s;
                out[data_base + half + pair] = x2 * c + x1 * s;
            }
        }
    }
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_rope_apply_vjp(
    const geo_real_t *grad_out,
    const geo_real_t *cos_table,
    const geo_real_t *sin_table,
    geo_real_t *grad_x,
    geo_tensor_rope_apply_shape shape
) {
    if (grad_out == NULL || cos_table == NULL || sin_table == NULL || grad_x == NULL ||
        !geo_tensor_rope_apply_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    const size_t half = shape.head_dim / 2u;
    for (size_t outer = 0u; outer < shape.outer; ++outer) {
        for (size_t token = 0u; token < shape.tokens; ++token) {
            const size_t row = outer * shape.tokens + token;
            const size_t data_base = row * shape.head_dim;
            const size_t table_base = token * half;
            for (size_t pair = 0u; pair < half; ++pair) {
                const geo_real_t grad_y1 = grad_out[data_base + pair];
                const geo_real_t grad_y2 = grad_out[data_base + half + pair];
                const geo_real_t c = cos_table[table_base + pair];
                const geo_real_t s = sin_table[table_base + pair];
                grad_x[data_base + pair] = grad_y1 * c + grad_y2 * s;
                grad_x[data_base + half + pair] = -grad_y1 * s + grad_y2 * c;
            }
        }
    }
    return GEO_TENSOR_OK;
}
