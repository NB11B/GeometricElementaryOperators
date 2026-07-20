#ifndef GEO_HAND_CUDA_COMPARATOR_H
#define GEO_HAND_CUDA_COMPARATOR_H

#include "geo/batch_gp_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_batch_gp_cuda_status_t geo_hand_cuda_forward_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    size_t batch_size,
    const double *device_parameter,
    int parameter_on_left,
    double *device_outputs,
    cudaStream_t stream
);

geo_batch_gp_cuda_status_t geo_hand_cuda_parameter_vjp_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    const double *device_output_cotangents,
    size_t batch_size,
    int parameter_on_left,
    double *device_parameter_cotangent,
    cudaStream_t stream
);

geo_batch_gp_cuda_status_t geo_hand_cuda_mse_sgd_step_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    const double *device_targets,
    size_t batch_size,
    double learning_rate,
    int parameter_on_left,
    double *device_parameter,
    double *device_residuals,
    double *device_gradient,
    double *device_loss,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif
