#ifndef GEO_TENSOR_OPTIMIZER_CUDA_H
#define GEO_TENSOR_OPTIMIZER_CUDA_H

#include "geo/tensor_optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_grad_square_cuda_accumulate(
    const float *gradient,
    size_t count,
    float *sum_square,
    void *stream
);

geo_tensor_status geo_tensor_grad_clip_cuda_finalize(
    const float *sum_square,
    float max_grad_norm,
    float *clip_scale,
    void *stream
);

geo_tensor_status geo_tensor_adamw_cuda_step(
    float *parameter,
    const float *gradient,
    float *first_moment,
    float *second_moment,
    size_t count,
    const float *clip_scale,
    geo_tensor_adamw_config config,
    void *stream
);

geo_tensor_status geo_tensor_adamw_cuda_step_fused(
    float **parameters,
    const float **gradients,
    float **first_moments,
    float **second_moments,
    const size_t *counts,
    size_t num_tensors,
    const float *clip_scale,
    geo_tensor_adamw_config config,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
