#include "geo/tensor_loss.h"

#include <math.h>
#include <stdint.h>

static int geo_tensor_cross_entropy_shape_valid(geo_tensor_cross_entropy_shape shape) {
    return shape.rows > 0u && shape.classes > 0u &&
           shape.rows <= SIZE_MAX / shape.classes;
}

geo_tensor_status geo_tensor_cross_entropy_forward(
    const geo_real_t *logits,
    const int64_t *targets,
    int64_t ignore_index,
    geo_real_t *loss,
    geo_real_t *probabilities,
    geo_real_t *normalizer,
    geo_tensor_cross_entropy_shape shape
) {
    if (logits == NULL || targets == NULL || loss == NULL || probabilities == NULL ||
        normalizer == NULL || !geo_tensor_cross_entropy_shape_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    for (size_t row = 0u; row < shape.rows; ++row) {
        const int64_t target = targets[row];
        if (target != ignore_index &&
            (target < 0 || (uint64_t)target >= (uint64_t)shape.classes)) {
            return GEO_TENSOR_INVALID_ARGUMENT;
        }
    }

    geo_real_t loss_sum = (geo_real_t)0;
    size_t valid_rows = 0u;
    for (size_t row = 0u; row < shape.rows; ++row) {
        const size_t base = row * shape.classes;
        const int64_t target = targets[row];
        if (target == ignore_index) {
            for (size_t class_index = 0u; class_index < shape.classes; ++class_index) {
                probabilities[base + class_index] = (geo_real_t)0;
            }
            continue;
        }

        geo_real_t max_logit = logits[base];
        for (size_t class_index = 1u; class_index < shape.classes; ++class_index) {
            if (logits[base + class_index] > max_logit) {
                max_logit = logits[base + class_index];
            }
        }

        geo_real_t exponential_sum = (geo_real_t)0;
        for (size_t class_index = 0u; class_index < shape.classes; ++class_index) {
            const geo_real_t exponential = (geo_real_t)exp(
                (double)(logits[base + class_index] - max_logit)
            );
            probabilities[base + class_index] = exponential;
            exponential_sum += exponential;
        }

        for (size_t class_index = 0u; class_index < shape.classes; ++class_index) {
            probabilities[base + class_index] /= exponential_sum;
        }

        loss_sum += max_logit + (geo_real_t)log((double)exponential_sum) -
                    logits[base + (size_t)target];
        ++valid_rows;
    }

    if (valid_rows == 0u) {
        *normalizer = (geo_real_t)0;
        *loss = (geo_real_t)NAN;
        return GEO_TENSOR_OK;
    }

    *normalizer = (geo_real_t)1 / (geo_real_t)valid_rows;
    *loss = loss_sum * *normalizer;
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_cross_entropy_vjp(
    const geo_real_t *probabilities,
    const int64_t *targets,
    int64_t ignore_index,
    geo_real_t normalizer,
    geo_real_t grad_loss,
    geo_real_t *grad_logits,
    geo_tensor_cross_entropy_shape shape
) {
    if (probabilities == NULL || targets == NULL || grad_logits == NULL ||
        normalizer < (geo_real_t)0 || !geo_tensor_cross_entropy_shape_valid(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    for (size_t row = 0u; row < shape.rows; ++row) {
        const int64_t target = targets[row];
        if (target != ignore_index &&
            (target < 0 || (uint64_t)target >= (uint64_t)shape.classes)) {
            return GEO_TENSOR_INVALID_ARGUMENT;
        }
    }

    const geo_real_t scale = grad_loss * normalizer;
    for (size_t row = 0u; row < shape.rows; ++row) {
        const size_t base = row * shape.classes;
        const int64_t target = targets[row];
        if (target == ignore_index || normalizer == (geo_real_t)0) {
            for (size_t class_index = 0u; class_index < shape.classes; ++class_index) {
                grad_logits[base + class_index] = (geo_real_t)0;
            }
            continue;
        }

        for (size_t class_index = 0u; class_index < shape.classes; ++class_index) {
            const geo_real_t indicator =
                class_index == (size_t)target ? (geo_real_t)1 : (geo_real_t)0;
            grad_logits[base + class_index] =
                scale * (probabilities[base + class_index] - indicator);
        }
    }
    return GEO_TENSOR_OK;
}
