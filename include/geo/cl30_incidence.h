#ifndef GEO_CL30_INCIDENCE_H
#define GEO_CL30_INCIDENCE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum geo_cl30_incidence_status {
    GEO_CL30_INCIDENCE_OK = 0,
    GEO_CL30_INCIDENCE_INVALID_ARGUMENT = 1,
    GEO_CL30_INCIDENCE_NONFINITE = 2,
    GEO_CL30_INCIDENCE_CERTIFICATE_FAILURE = 3
} geo_cl30_incidence_status;

typedef struct geo_cl30_incidence_shape {
    int32_t batch_count;
    int32_t node_count;
    int32_t edge_count;
    int32_t relation_count;
    int32_t channel_count;
} geo_cl30_incidence_shape;

size_t geo_cl30_incidence_workspace_bytes(
    const geo_cl30_incidence_shape* shape
);

geo_cl30_incidence_status geo_cl30_incidence_forward_f32(
    const geo_cl30_incidence_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    const float* inverse_degrees,
    float* node_aggregates,
    void* workspace,
    size_t workspace_bytes,
    float* max_rotor_residual
);

geo_cl30_incidence_status geo_cl30_incidence_vjp_f32(
    const geo_cl30_incidence_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    const float* inverse_degrees,
    const float* grad_node_aggregates,
    float* grad_relation_bivectors,
    float* grad_node_states,
    void* workspace,
    size_t workspace_bytes
);

#ifdef __cplusplus
}
#endif

#endif // GEO_CL30_INCIDENCE_H
