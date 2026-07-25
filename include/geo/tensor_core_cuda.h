#ifndef GEO_TENSOR_CORE_CUDA_H
#define GEO_TENSOR_CORE_CUDA_H

#include <stddef.h>

#include "geo/tensor_core.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_tensor_status geo_tensor_add_cuda_forward(const float *a, const float *b, float *out, size_t count, void *stream);
geo_tensor_status geo_tensor_add_cuda_vjp(const float *grad_out, float *grad_a, float *grad_b, size_t count, void *stream);
geo_tensor_status geo_tensor_mul_cuda_forward(const float *a, const float *b, float *out, size_t count, void *stream);
geo_tensor_status geo_tensor_mul_cuda_vjp(const float *a, const float *b, const float *grad_out, float *grad_a, float *grad_b, size_t count, void *stream);
geo_tensor_status geo_tensor_scale_cuda_forward(const float *x, float scalar, float *out, size_t count, void *stream);
geo_tensor_status geo_tensor_scale_cuda_vjp(const float *grad_out, float scalar, float *grad_x, size_t count, void *stream);
geo_tensor_status geo_tensor_rms_norm_cuda_forward(const float *x, const float *weight, float epsilon, float *out, float *inv_rms, geo_tensor_norm_shape shape, void *stream);
geo_tensor_status geo_tensor_rms_norm_cuda_vjp(const float *x, const float *weight, const float *grad_out, const float *inv_rms, float *grad_x, float *grad_weight, geo_tensor_norm_shape shape, void *stream);

#ifdef __cplusplus
}
#endif

#endif
