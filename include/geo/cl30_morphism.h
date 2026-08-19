#ifndef GEO_CL30_MORPHISM_H
#define GEO_CL30_MORPHISM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum geo_cl30_morphism_status {
    GEO_CL30_MORPHISM_OK = 0,
    GEO_CL30_MORPHISM_INVALID_ARGUMENT = 1,
    GEO_CL30_MORPHISM_NONFINITE = 2,
    GEO_CL30_MORPHISM_CERTIFICATE_FAILURE = 3
} geo_cl30_morphism_status;

typedef struct geo_cl30_morphism_shape {
    int32_t batch_count;
    int32_t node_count;
    int32_t edge_count;
    int32_t relation_count;
    int32_t channel_count;
} geo_cl30_morphism_shape;

typedef struct geo_cl30_morphism_telemetry {
    float max_rotor_residual;
    float alpha_mean;
    float alpha_std;
    float alpha_min;
    float alpha_max;
    float beta_mean;
    float beta_std;
    float beta_min;
    float beta_max;
    float gate_product_mean;
    float gate_product_std;
    float gate_denominator_min;
    float gate_denominator_max;
    float alpha_sat_low_rate;
    float alpha_sat_high_rate;
    float beta_sat_low_rate;
    float beta_sat_high_rate;
} geo_cl30_morphism_telemetry;

size_t geo_cl30_morphism_workspace_bytes(
    const geo_cl30_morphism_shape* shape
);

geo_cl30_morphism_status geo_cl30_morphism_forward_f32(
    const geo_cl30_morphism_shape* shape,
    const float* relation_bivectors,    /* [R, C, 3] */
    const float* node_states,           /* [B, N, C, 3] */
    const float* goal_queries,          /* [B, C, 3] */
    const float* applicability_params,  /* [3]: w_a, w_ar, b_a */
    const float* compatibility_params,  /* [5]: w_s, theta_b, w_q, w_tau, b_beta */
    const int32_t* edge_batch,          /* [E] */
    const int32_t* edge_source,         /* [E] */
    const int32_t* edge_relation,       /* [E] */
    const int32_t* edge_destination,    /* [E] */
    int32_t lesion_mode,                /* 0=none, 1=app, 2=comp, 3=comm, 4=ident, 5=zero */
    float* node_aggregates,             /* [B, N, C, 3] */
    float* edge_alphas,                 /* [E, C] (optional, may be null) */
    float* edge_betas,                  /* [E, C] (optional, may be null) */
    float* gate_denominators,           /* [B, N, C] (optional, may be null) */
    void* workspace,
    size_t workspace_bytes,
    geo_cl30_morphism_telemetry* telemetry
);

geo_cl30_morphism_status geo_cl30_morphism_vjp_f32(
    const geo_cl30_morphism_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const float* goal_queries,
    const float* applicability_params,
    const float* compatibility_params,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    const float* grad_node_aggregates,
    float* grad_relation_bivectors,
    float* grad_node_states,
    float* grad_goal_queries,
    float* grad_applicability_params,
    float* grad_compatibility_params,
    void* workspace,
    size_t workspace_bytes
);

#ifdef __cplusplus
}
#endif

#endif /* GEO_CL30_MORPHISM_H */
