#ifndef GEO_TENSOR_LINEAR_CUDA_H
#define GEO_TENSOR_LINEAR_CUDA_H

#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA stream is passed as an opaque cudaStream_t-compatible pointer. */
geo_tensor_status geo_tensor_linear_cuda_forward(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    const geo_tensor_linear_shape *shape,
    void *stream
);

geo_tensor_status geo_tensor_linear_cuda_vjp(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    const geo_tensor_linear_shape *shape,
    void *stream_ptr
);

geo_tensor_status geo_tensor_linear_cuda_vjp_decomposed_profile(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    const geo_tensor_linear_shape *shape,
    float *dx_ms,
    float *dw_ms,
    void *stream_ptr
);

#ifdef __cplusplus
}
#endif

#endif
