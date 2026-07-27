#ifndef GEO_IMPLICIT_CUDA_H
#define GEO_IMPLICIT_CUDA_H

#include "geo/tensor_core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t batch_tokens;   // B * T
    size_t in_features;    // in_features
    size_t out_features;   // out_features
    size_t rank;           // r (e.g. 4)
} geo_implicit_shape;

geo_tensor_status geo_implicit_linear_cuda_forward(
    const float *x,              // [B*T, in_features]
    const float *u,              // [rank, out_features]
    const float *v,              // [rank, in_features]
    const float *alpha,          // [rank]
    const int32_t *perm_indices, // [rank, in_features]
    const float *sign_masks,     // [rank, in_features]
    float *y,                    // [B*T, out_features]
    const geo_implicit_shape *shape,
    void *stream_ptr
);

geo_tensor_status geo_implicit_linear_cuda_vjp(
    const float *x,              // [B*T, in_features]
    const float *u,              // [rank, out_features]
    const float *v,              // [rank, in_features]
    const float *alpha,          // [rank]
    const int32_t *perm_indices, // [rank, in_features]
    const int32_t *inv_perm,     // [rank, in_features]
    const float *sign_masks,     // [rank, in_features]
    const float *grad_y,         // [B*T, out_features]
    float *grad_x,               // [B*T, in_features]
    float *grad_u,               // [rank, out_features]
    float *grad_v,               // [rank, in_features]
    float *grad_alpha,           // [rank]
    const geo_implicit_shape *shape,
    void *stream_ptr
);

#ifdef __cplusplus
}
#endif

#endif // GEO_IMPLICIT_CUDA_H
