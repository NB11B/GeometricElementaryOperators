#ifndef GEO_TENSOR_ROPE_CUDA_H
#define GEO_TENSOR_ROPE_CUDA_H

#include "geo/tensor_rope.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_rope_cuda_build(
    float theta,
    float *cos_out,
    float *sin_out,
    geo_tensor_rope_table_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_rope_cuda_apply_forward(
    const float *x,
    const float *cos_table,
    const float *sin_table,
    float *out,
    geo_tensor_rope_apply_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_rope_cuda_apply_vjp(
    const float *grad_out,
    const float *cos_table,
    const float *sin_table,
    float *grad_x,
    geo_tensor_rope_apply_shape shape,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
