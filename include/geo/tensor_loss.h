#ifndef GEO_TENSOR_LOSS_H
#define GEO_TENSOR_LOSS_H

#include <stddef.h>
#include <stdint.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct geo_tensor_cross_entropy_shape {
    size_t rows;
    size_t classes;
} geo_tensor_cross_entropy_shape;

/*
 * Stable mean cross-entropy for row-major logits [rows, classes].
 * probabilities receives the softmax values used by the VJP.
 * normalizer receives 1 / valid_rows and is saved for backward.
 */
geo_tensor_status geo_tensor_cross_entropy_forward(
    const geo_real_t *logits,
    const int64_t *targets,
    int64_t ignore_index,
    geo_real_t *loss,
    geo_real_t *probabilities,
    geo_real_t *normalizer,
    geo_tensor_cross_entropy_shape shape
);

/*
 * VJP with respect to logits for a scalar upstream gradient.
 * Targets equal to ignore_index produce zero gradients.
 */
geo_tensor_status geo_tensor_cross_entropy_vjp(
    const geo_real_t *probabilities,
    const int64_t *targets,
    int64_t ignore_index,
    geo_real_t normalizer,
    geo_real_t grad_loss,
    geo_real_t *grad_logits,
    geo_tensor_cross_entropy_shape shape
);

#ifdef __cplusplus
}
#endif

#endif
