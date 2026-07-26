#ifndef GEO_TENSOR_ATTENTION_CUDA_H
#define GEO_TENSOR_ATTENTION_CUDA_H

#include "geo/tensor_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct geo_attention_backward_timings {
    float t_dp_ds_ms;
    float t_dq_ms;
    float t_dk_dv_ms;
    float t_total_ms;
} geo_attention_backward_timings;

typedef struct geo_attention_backend_counters {
    unsigned long long n_forward_with_probs_calls;
    unsigned long long n_forward_no_probs_calls;
    unsigned long long n_streaming_forward_calls;
    unsigned long long n_backward_probability_recompute_calls;
    unsigned long long n_attention_vjp_calls;
    unsigned long long n_streaming_vjp_calls;
} geo_attention_backend_counters;

void geo_tensor_causal_attention_get_counters(geo_attention_backend_counters *out);
void geo_tensor_causal_attention_reset_counters(void);
void geo_tensor_causal_attention_set_perturbation(float delta);

geo_tensor_status geo_tensor_causal_attention_cuda_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    float *probabilities,
    geo_tensor_attention_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_causal_attention_cuda_forward_no_probs(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    geo_tensor_attention_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_causal_attention_cuda_streaming_forward(
    const float *q,
    const float *k,
    const float *v,
    float *out,
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
    float *workspace_dp_ds,  /* Optional workspace buffer of size [outer * tokens * tokens] */
    geo_attention_backward_timings *timings, /* Optional internal timing output */
    geo_tensor_attention_shape shape,
    void *stream
);

geo_tensor_status geo_tensor_causal_attention_cuda_streaming_vjp(
    const float *q,
    const float *k,
    const float *v,
    const float *out,
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
