#ifndef GEO_TENSOR_ACTIVATION_CUDA_H
#define GEO_TENSOR_ACTIVATION_CUDA_H

#include <stddef.h>

#include "geo/tensor_activation.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_gelu_cuda_forward(
    const float *x,
    float *out,
    size_t count,
    void *stream
);

geo_tensor_status geo_tensor_gelu_cuda_vjp(
    const float *x,
    const float *grad_out,
    float *grad_x,
    size_t count,
    void *stream
);

geo_tensor_status geo_tensor_silu_mul_cuda_forward(
    const float *gate,
    const float *up,
    float *out,
    size_t count,
    void *stream
);

geo_tensor_status geo_tensor_silu_mul_cuda_vjp(
    const float *gate,
    const float *up,
    const float *grad_out,
    float *grad_gate,
    float *grad_up,
    size_t count,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
