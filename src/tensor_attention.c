#include "geo/tensor_attention.h"

#include <math.h>
#include <stdint.h>

static int geo_tensor_attention_shape_valid(geo_tensor_attention_shape shape) {
    if (shape.outer == 0u || shape.tokens == 0u || shape.head_dim == 0u) {
        return 0;
    }
    if (shape.outer > SIZE_MAX / shape.tokens) {
        return 0;
    }
    const size_t rows = shape.outer * shape.tokens;
    if (rows > SIZE_MAX / shape.head_dim) {
        return 0;
    }
    if (rows > SIZE_MAX / shape.tokens) {
        return 0;
    }
    return 1;
}

static size_t geo_attention_data_index(
    size_t outer,
    size_t token,
    size_t dim,
    geo_tensor_attention_shape shape
) {
    return (outer * shape.tokens + token) * shape.head_dim + dim;
}

static size_t geo_attention_probability_index(
    size_t outer,
    size_t query,
    size_t key,
    geo_tensor_attention_shape shape
) {
    return (outer * shape.tokens + query) * shape.tokens + key;
}

static geo_real_t geo_attention_dot(
    const geo_real_t *a,
    size_t a_token,
    const geo_real_t *b,
    size_t b_token,
    size_t outer,
    geo_tensor_attention_shape shape
) {
    geo_real_t sum = (geo_real_t)0;
    for (size_t dim = 0u; dim < shape.head_dim; ++dim) {
        sum += a[geo_attention_data_index(outer, a_token, dim, shape)] *
               b[geo_attention_data_index(outer, b_token, dim, shape)];
    }
    return sum;
}

geo_tensor_status geo_tensor_causal_attention_forward(
    const geo_real_t *q,
    const geo_real_t *k,
    const geo_real_t *v,
    geo_real_t *out,
    geo_real_t *probabilities,
    geo_tensor_attention_shape shape
) {
    if (q == NULL || k == NULL || v == NULL || out == NULL || probabilities == NULL ||
        !geo_tensor_attention_shape_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    const geo_real_t scale = (geo_real_t)1 /
        (geo_real_t)sqrt((double)shape.head_dim);

    for (size_t outer = 0u; outer < shape.outer; ++outer) {
        for (size_t query = 0u; query < shape.tokens; ++query) {
            geo_real_t max_score = (geo_real_t)-INFINITY;
            for (size_t key = 0u; key <= query; ++key) {
                const geo_real_t score =
                    geo_attention_dot(q, query, k, key, outer, shape) * scale;
                if (score > max_score) {
                    max_score = score;
                }
            }

            geo_real_t normalizer = (geo_real_t)0;
            for (size_t key = 0u; key < shape.tokens; ++key) {
                const size_t probability_index =
                    geo_attention_probability_index(outer, query, key, shape);
                if (key <= query) {
                    const geo_real_t score =
                        geo_attention_dot(q, query, k, key, outer, shape) * scale;
                    const geo_real_t unnormalized =
                        (geo_real_t)exp((double)(score - max_score));
                    probabilities[probability_index] = unnormalized;
                    normalizer += unnormalized;
                } else {
                    probabilities[probability_index] = (geo_real_t)0;
                }
            }

            for (size_t key = 0u; key <= query; ++key) {
                const size_t probability_index =
                    geo_attention_probability_index(outer, query, key, shape);
                probabilities[probability_index] /= normalizer;
            }

            for (size_t dim = 0u; dim < shape.head_dim; ++dim) {
                geo_real_t value = (geo_real_t)0;
                for (size_t key = 0u; key <= query; ++key) {
                    value += probabilities[
                        geo_attention_probability_index(outer, query, key, shape)
                    ] * v[geo_attention_data_index(outer, key, dim, shape)];
                }
                out[geo_attention_data_index(outer, query, dim, shape)] = value;
            }
        }
    }
    return GEO_TENSOR_OK;
}

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
) {
    if (q == NULL || k == NULL || v == NULL || probabilities == NULL ||
        grad_out == NULL || grad_q == NULL || grad_k == NULL || grad_v == NULL ||
        !geo_tensor_attention_shape_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    const size_t data_count = shape.outer * shape.tokens * shape.head_dim;
    for (size_t index = 0u; index < data_count; ++index) {
        grad_q[index] = (geo_real_t)0;
        grad_k[index] = (geo_real_t)0;
        grad_v[index] = (geo_real_t)0;
    }

    const geo_real_t scale = (geo_real_t)1 /
        (geo_real_t)sqrt((double)shape.head_dim);

    for (size_t outer = 0u; outer < shape.outer; ++outer) {
        for (size_t query = 0u; query < shape.tokens; ++query) {
            geo_real_t softmax_dot = (geo_real_t)0;
            for (size_t key = 0u; key <= query; ++key) {
                const geo_real_t d_probability =
                    geo_attention_dot(grad_out, query, v, key, outer, shape);
                softmax_dot += probabilities[
                    geo_attention_probability_index(outer, query, key, shape)
                ] * d_probability;
            }

            for (size_t key = 0u; key <= query; ++key) {
                const geo_real_t probability = probabilities[
                    geo_attention_probability_index(outer, query, key, shape)
                ];
                const geo_real_t d_probability =
                    geo_attention_dot(grad_out, query, v, key, outer, shape);
                const geo_real_t d_score =
                    probability * (d_probability - softmax_dot);

                for (size_t dim = 0u; dim < shape.head_dim; ++dim) {
                    const size_t query_index =
                        geo_attention_data_index(outer, query, dim, shape);
                    const size_t key_index =
                        geo_attention_data_index(outer, key, dim, shape);
                    grad_q[query_index] += scale * d_score * k[key_index];
                    grad_k[key_index] += scale * d_score * q[query_index];
                    grad_v[key_index] += probability * grad_out[query_index];
                }
            }
        }
    }
    return GEO_TENSOR_OK;
}
