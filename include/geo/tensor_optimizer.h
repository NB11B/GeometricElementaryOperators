#ifndef GEO_TENSOR_OPTIMIZER_H
#define GEO_TENSOR_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>

#include "geo/cl20.h"
#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct geo_tensor_adamw_config {
    geo_real_t learning_rate;
    geo_real_t beta1;
    geo_real_t beta2;
    geo_real_t epsilon;
    geo_real_t weight_decay;
    geo_real_t max_grad_norm;
    uint64_t step;
} geo_tensor_adamw_config;

/* Adds sum(gradient^2) into the caller-owned accumulator. */
geo_tensor_status geo_tensor_grad_square_accumulate(
    const geo_real_t *gradient,
    size_t count,
    geo_real_t *sum_square
);

/* Computes min(1, max_grad_norm / (sqrt(sum_square) + 1e-6)). */
geo_tensor_status geo_tensor_grad_clip_scale(
    geo_real_t sum_square,
    geo_real_t max_grad_norm,
    geo_real_t *clip_scale
);

/* In-place AdamW parameter and moment update with precomputed clip scale. */
geo_tensor_status geo_tensor_adamw_step(
    geo_real_t *parameter,
    const geo_real_t *gradient,
    geo_real_t *first_moment,
    geo_real_t *second_moment,
    size_t count,
    geo_real_t clip_scale,
    geo_tensor_adamw_config config
);

#ifdef __cplusplus
}
#endif

#endif
