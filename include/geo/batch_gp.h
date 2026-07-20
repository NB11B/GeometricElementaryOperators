#ifndef GEO_BATCH_GP_H
#define GEO_BATCH_GP_H

#include "geo/operator_kernel.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_BATCH_GP_ABI_VERSION 0x00010000u
#define GEO_BATCH_GP_MAX_TERMS (GEO_OPERATOR_MAX_BLADES * GEO_OPERATOR_MAX_BLADES)

typedef enum {
    GEO_BATCH_GP_OK = 0,
    GEO_BATCH_GP_INVALID_ARGUMENT = 1,
    GEO_BATCH_GP_UNSUPPORTED_DIMENSION = 2,
    GEO_BATCH_GP_NUMERIC_FAILURE = 3
} geo_batch_gp_status_t;

typedef struct {
    uint32_t abi_version;
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    uint16_t blade_count;
    uint16_t term_count;
    uint8_t left_blade[GEO_BATCH_GP_MAX_TERMS];
    uint8_t right_blade[GEO_BATCH_GP_MAX_TERMS];
    uint8_t output_blade[GEO_BATCH_GP_MAX_TERMS];
    int8_t sign[GEO_BATCH_GP_MAX_TERMS];
} geo_batch_gp_plan_t;

uint32_t geo_batch_gp_abi_version(void);

geo_batch_gp_status_t geo_batch_gp_plan_init(
    geo_batch_gp_plan_t *plan,
    uint8_t dimension,
    const int8_t *signature
);

geo_batch_gp_status_t geo_batch_gp_right_forward_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    size_t batch_size,
    const double *right_parameter,
    double *outputs
);

geo_batch_gp_status_t geo_batch_gp_left_forward_f64(
    const geo_batch_gp_plan_t *plan,
    const double *left_parameter,
    const double *inputs,
    size_t batch_size,
    double *outputs
);

geo_batch_gp_status_t geo_batch_gp_right_vjp_parameter_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *output_cotangents,
    size_t batch_size,
    double *parameter_cotangent
);

geo_batch_gp_status_t geo_batch_gp_left_vjp_parameter_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *output_cotangents,
    size_t batch_size,
    double *parameter_cotangent
);

geo_batch_gp_status_t geo_batch_gp_right_mse_sgd_step_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *targets,
    size_t batch_size,
    double learning_rate,
    double *right_parameter,
    double *mean_loss
);

geo_batch_gp_status_t geo_batch_gp_left_mse_sgd_step_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *targets,
    size_t batch_size,
    double learning_rate,
    double *left_parameter,
    double *mean_loss
);

#ifdef __cplusplus
}
#endif

#endif
