#ifndef GEO_TENSOR_LOSS_CUDA_H
#define GEO_TENSOR_LOSS_CUDA_H

#include "geo/tensor_loss.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_cross_entropy_cuda_forward(
    const float *logits,
    const int64_t *targets,
    int64_t ignore_index,
    float *loss,
    float *probabilities,
    float *normalizer,
    geo_tensor_cross_entropy_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_cross_entropy_cuda_vjp(
    const float *probabilities,
    const int64_t *targets,
    int64_t ignore_index,
    const float *normalizer,
    const float *grad_loss,
    float *grad_logits,
    geo_tensor_cross_entropy_shape shape,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
