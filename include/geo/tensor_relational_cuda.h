#ifndef GEO_TENSOR_RELATIONAL_CUDA_H
#define GEO_TENSOR_RELATIONAL_CUDA_H

#include "geo/tensor_relational.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_relational_status geo_relational_project_forward_cuda(
    const geo_real_t *d_logits,
    geo_real_t *d_relationship,
    geo_real_t *d_workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options,
    geo_relational_certificate *certificates,
    void *stream
);

geo_relational_status geo_relational_project_vjp_cuda(
    const geo_real_t *d_logits,
    const geo_real_t *d_relationship_cotangent,
    geo_real_t *d_logits_cotangent,
    geo_real_t *d_workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options,
    void *stream
);

geo_relational_status geo_relational_identity_gate_forward_cuda(
    const geo_real_t *d_projected_relationship,
    const geo_real_t *d_gate,
    geo_real_t *d_effective_relationship,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_identity_gate_vjp_cuda(
    const geo_real_t *d_projected_relationship,
    const geo_real_t *d_gate,
    const geo_real_t *d_effective_relationship_cotangent,
    geo_real_t *d_projected_relationship_cotangent,
    geo_real_t *d_gate_cotangent,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_mix_forward_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_relationship,
    geo_real_t *d_output,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_mix_vjp_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_relationship,
    const geo_real_t *d_output_cotangent,
    geo_real_t *d_state_cotangent,
    geo_real_t *d_relationship_cotangent,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_read_forward_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_read_weights,
    size_t weight_count,
    geo_real_t *d_read_state,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_read_vjp_cuda(
    const geo_real_t *d_state,
    const geo_real_t *d_read_weights,
    size_t weight_count,
    const geo_real_t *d_read_state_cotangent,
    geo_real_t *d_state_cotangent,
    geo_real_t *d_read_weights_cotangent,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_write_add_forward_cuda(
    const geo_real_t *d_transported_state,
    const geo_real_t *d_source,
    const geo_real_t *d_write_weights,
    size_t weight_count,
    const geo_real_t *d_source_scale,
    size_t scale_count,
    geo_real_t *d_output,
    const geo_relational_shape *shape,
    void *stream
);

geo_relational_status geo_relational_write_add_vjp_cuda(
    const geo_real_t *d_source,
    const geo_real_t *d_write_weights,
    size_t weight_count,
    const geo_real_t *d_source_scale,
    size_t scale_count,
    const geo_real_t *d_output_cotangent,
    geo_real_t *d_transported_state_cotangent,
    geo_real_t *d_source_cotangent,
    geo_real_t *d_write_weights_cotangent,
    geo_real_t *d_source_scale_cotangent,
    const geo_relational_shape *shape,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
