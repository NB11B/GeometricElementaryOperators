#ifndef GEO_BATCH_GP_CUDA_H
#define GEO_BATCH_GP_CUDA_H

#include "geo/batch_gp.h"
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_BATCH_GP_CUDA_ABI_VERSION 0x00010001u

typedef enum {
    GEO_BATCH_GP_CUDA_OK = 0,
    GEO_BATCH_GP_CUDA_INVALID_ARGUMENT = 1,
    GEO_BATCH_GP_CUDA_ALLOCATION_FAILURE = 2,
    GEO_BATCH_GP_CUDA_RUNTIME_FAILURE = 3,
    GEO_BATCH_GP_CUDA_NUMERIC_FAILURE = 4
} geo_batch_gp_cuda_status_t;

typedef struct {
    uint32_t abi_version;
    uint16_t blade_count;
    uint16_t term_count;
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    int8_t *device_signature;
    uint8_t *device_left_blade;
    uint8_t *device_right_blade;
    uint8_t *device_output_blade;
    int8_t *device_sign;
} geo_batch_gp_cuda_plan_t;

uint32_t geo_batch_gp_cuda_abi_version(void);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_plan_upload(geo_batch_gp_cuda_plan_t *device_plan, const geo_batch_gp_plan_t *host_plan);
void geo_batch_gp_cuda_plan_destroy(geo_batch_gp_cuda_plan_t *plan);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_reference_forward_f64(const geo_batch_gp_cuda_plan_t *plan, const double *device_inputs, size_t batch_size, const double *device_parameter, int parameter_on_left, double *device_outputs, cudaStream_t stream);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_planned_forward_f64(const geo_batch_gp_cuda_plan_t *plan, const double *device_inputs, size_t batch_size, const double *device_parameter, int parameter_on_left, double *device_outputs, cudaStream_t stream);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_reference_parameter_vjp_f64(const geo_batch_gp_cuda_plan_t *plan, const double *device_inputs, const double *device_output_cotangents, size_t batch_size, int parameter_on_left, double *device_parameter_cotangent, cudaStream_t stream);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_parameter_vjp_f64(const geo_batch_gp_cuda_plan_t *plan, const double *device_inputs, const double *device_output_cotangents, size_t batch_size, int parameter_on_left, double *device_parameter_cotangent, cudaStream_t stream);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_reference_mse_sgd_step_f64(const geo_batch_gp_cuda_plan_t *plan, const double *device_inputs, const double *device_targets, size_t batch_size, double learning_rate, int parameter_on_left, double *device_parameter, double *device_residuals, double *device_gradient, double *device_loss, cudaStream_t stream);
geo_batch_gp_cuda_status_t geo_batch_gp_cuda_mse_sgd_step_f64(const geo_batch_gp_cuda_plan_t *plan, const double *device_inputs, const double *device_targets, size_t batch_size, double learning_rate, int parameter_on_left, double *device_parameter, double *device_residuals, double *device_gradient, double *device_loss, cudaStream_t stream);
const char *geo_batch_gp_cuda_status_string(geo_batch_gp_cuda_status_t status);

#ifdef __cplusplus
}
#endif

#endif
