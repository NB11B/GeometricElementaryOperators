#ifndef GEO_TENSOR_ATTENTION_CUDA_H
#define GEO_TENSOR_ATTENTION_CUDA_H

#include "geo/tensor_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_causal_attention_cuda_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *probabilities,
    geo_tensor_attention_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_causal_attention_cuda_vjp(
    const float *q,
    const float *k,
    const float *v,
    const float *probabilities,
    const float *grad_out,
    float *grad_q,
    float *grad_k,
    float *grad_v,
    geo_tensor_attention_shape shape,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
