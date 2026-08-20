#ifndef GEO_TENSOR_RELATIONAL_H
#define GEO_TENSOR_RELATIONAL_H

#include <stddef.h>
#include <stdint.h>

#include "geo/tensor_linear.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_RELATIONAL_ABI_VERSION 0x00010000u
#define GEO_RELATIONAL_MAX_STREAMS 8u
#define GEO_RELATIONAL_MAX_SINKHORN_ITERATIONS 64u

typedef enum geo_relational_projection_mode {
    GEO_RELATIONAL_PROJECTION_NONE = 0,
    GEO_RELATIONAL_PROJECTION_ROW_STOCHASTIC = 1,
    GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN = 2
} geo_relational_projection_mode;

typedef enum geo_relational_status {
    GEO_RELATIONAL_OK = 0,
    GEO_RELATIONAL_INVALID_ARGUMENT = 1,
    GEO_RELATIONAL_OVERFLOW = 2,
    GEO_RELATIONAL_INSUFFICIENT_WORKSPACE = 3,
    GEO_RELATIONAL_NUMERIC_FAILURE = 4,
    GEO_RELATIONAL_CONSTRAINT_FAILURE = 5,
    GEO_RELATIONAL_CUDA_ERROR = 6,
    GEO_RELATIONAL_BACKEND_UNAVAILABLE = 7,
    GEO_RELATIONAL_UNSUPPORTED = 8
} geo_relational_status;

typedef struct geo_relational_shape {
    size_t groups;
    size_t streams;
    size_t features;
    size_t matrix_count;
} geo_relational_shape;

typedef struct geo_relational_projection_options {
    uint32_t abi_version;
    uint32_t iterations;
    geo_real_t epsilon;
    uint8_t mode;
    uint8_t fail_on_nonfinite;
    uint8_t require_certificate;
    uint8_t reserved;
} geo_relational_projection_options;

typedef struct geo_relational_certificate {
    uint32_t abi_version;
    size_t streams;
    geo_real_t minimum_entry;
    geo_real_t maximum_entry;
    geo_real_t maximum_row_sum_error;
    geo_real_t maximum_column_sum_error;
    geo_real_t forward_amax_gain;
    geo_real_t backward_amax_gain;
    geo_real_t identity_distance_frobenius;
    geo_real_t consensus_distance_frobenius;
    geo_real_t certificate_tolerance;
    uint8_t finite;
    uint8_t nonnegative;
    uint8_t row_balanced;
    uint8_t column_balanced;
    uint8_t accepted;
    uint8_t reserved[3];
} geo_relational_certificate;

uint32_t geo_relational_abi_version(void);

size_t geo_relational_projection_workspace_elements(
    size_t matrix_count,
    size_t streams,
    uint32_t iterations,
    int backward
);

geo_relational_status geo_relational_project_forward(
    const geo_real_t *logits,
    geo_real_t *relationship,
    geo_real_t *workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options,
    geo_relational_certificate *certificates
);

geo_relational_status geo_relational_project_vjp(
    const geo_real_t *logits,
    const geo_real_t *relationship_cotangent,
    geo_real_t *logits_cotangent,
    geo_real_t *workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options
);

geo_relational_status geo_relational_identity_gate_forward(
    const geo_real_t *projected_relationship,
    const geo_real_t *gate,
    geo_real_t *effective_relationship,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_identity_gate_vjp(
    const geo_real_t *projected_relationship,
    const geo_real_t *gate,
    const geo_real_t *effective_relationship_cotangent,
    geo_real_t *projected_relationship_cotangent,
    geo_real_t *gate_cotangent,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_mix_forward(
    const geo_real_t *state,
    const geo_real_t *relationship,
    geo_real_t *output,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_mix_vjp(
    const geo_real_t *state,
    const geo_real_t *relationship,
    const geo_real_t *output_cotangent,
    geo_real_t *state_cotangent,
    geo_real_t *relationship_cotangent,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_read_forward(
    const geo_real_t *state,
    const geo_real_t *read_weights,
    size_t weight_count,
    geo_real_t *read_state,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_read_vjp(
    const geo_real_t *state,
    const geo_real_t *read_weights,
    size_t weight_count,
    const geo_real_t *read_state_cotangent,
    geo_real_t *state_cotangent,
    geo_real_t *read_weights_cotangent,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_write_add_forward(
    const geo_real_t *transported_state,
    const geo_real_t *source,
    const geo_real_t *write_weights,
    size_t weight_count,
    const geo_real_t *source_scale,
    size_t scale_count,
    geo_real_t *output,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_write_add_vjp(
    const geo_real_t *source,
    const geo_real_t *write_weights,
    size_t weight_count,
    const geo_real_t *source_scale,
    size_t scale_count,
    const geo_real_t *output_cotangent,
    geo_real_t *transported_state_cotangent,
    geo_real_t *source_cotangent,
    geo_real_t *write_weights_cotangent,
    geo_real_t *source_scale_cotangent,
    const geo_relational_shape *shape
);

geo_relational_status geo_relational_certify(
    const geo_real_t *relationship,
    geo_real_t tolerance,
    const geo_relational_shape *shape,
    geo_relational_certificate *certificates
);

geo_relational_status geo_relational_compose(
    const geo_real_t *left,
    const geo_real_t *right,
    geo_real_t *product,
    size_t matrix_count,
    size_t streams
);

const char *geo_relational_status_string(geo_relational_status status);

#ifdef __cplusplus
}
#endif

#endif
