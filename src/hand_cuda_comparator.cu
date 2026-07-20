#include "geo/hand_cuda_comparator.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int kThreads = 256;

bool valid(const geo_batch_gp_cuda_plan_t *plan) {
    return plan != nullptr &&
        plan->abi_version == GEO_BATCH_GP_CUDA_ABI_VERSION &&
        plan->blade_count > 0 &&
        plan->term_count == plan->blade_count * plan->blade_count &&
        plan->device_left_blade != nullptr &&
        plan->device_right_blade != nullptr &&
        plan->device_output_blade != nullptr &&
        plan->device_sign != nullptr;
}

geo_batch_gp_cuda_status_t last_error() {
    return cudaGetLastError() == cudaSuccess ?
        GEO_BATCH_GP_CUDA_OK : GEO_BATCH_GP_CUDA_RUNTIME_FAILURE;
}

__global__ void hand_forward_kernel(
    const uint8_t *left_blade,
    const uint8_t *right_blade,
    const uint8_t *output_blade,
    const int8_t *sign,
    uint16_t term_count,
    uint16_t blades,
    const double *inputs,
    size_t batch_size,
    const double *parameter,
    int parameter_on_left,
    double *outputs
) {
    const size_t sample = (size_t)blockIdx.x;
    const uint16_t out = (uint16_t)threadIdx.x;
    if (sample >= batch_size || out >= blades) return;

    double sum = 0.0;
    for (uint16_t term = 0; term < term_count; ++term) {
        if (output_blade[term] != out) continue;
        const uint16_t input_index = parameter_on_left ? right_blade[term] : left_blade[term];
        const uint16_t parameter_index = parameter_on_left ? left_blade[term] : right_blade[term];
        sum += inputs[sample * blades + input_index] *
            parameter[parameter_index] * (double)sign[term];
    }
    outputs[sample * blades + out] = sum;
}

__global__ void hand_vjp_kernel(
    const uint8_t *left_blade,
    const uint8_t *right_blade,
    const uint8_t *output_blade,
    const int8_t *sign,
    uint16_t term_count,
    uint16_t blades,
    const double *inputs,
    const double *cotangents,
    size_t batch_size,
    int parameter_on_left,
    double *gradient
) {
    const uint16_t parameter_index = (uint16_t)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (parameter_index >= blades) return;

    double sum = 0.0;
    for (uint16_t term = 0; term < term_count; ++term) {
        const uint16_t term_parameter = parameter_on_left ? left_blade[term] : right_blade[term];
        if (term_parameter != parameter_index) continue;
        const uint16_t input_index = parameter_on_left ? right_blade[term] : left_blade[term];
        const uint16_t out = output_blade[term];
        for (size_t sample = 0; sample < batch_size; ++sample) {
            sum += inputs[sample * blades + input_index] *
                cotangents[sample * blades + out] * (double)sign[term];
        }
    }
    gradient[parameter_index] = sum;
}

__global__ void hand_residual_loss_kernel(
    double *outputs,
    const double *targets,
    size_t count,
    double *loss
) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const double residual = outputs[index] - targets[index];
    outputs[index] = residual;
    atomicAdd(loss, 0.5 * residual * residual);
}

__global__ void hand_sgd_kernel(
    double *parameter,
    const double *gradient,
    uint16_t blades,
    double scale
) {
    const uint16_t index = (uint16_t)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (index < blades) parameter[index] -= scale * gradient[index];
}
}

extern "C" geo_batch_gp_cuda_status_t geo_hand_cuda_forward_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    size_t batch_size,
    const double *device_parameter,
    int parameter_on_left,
    double *device_outputs,
    cudaStream_t stream
) {
    if (!valid(plan) || device_inputs == nullptr || device_parameter == nullptr ||
        device_outputs == nullptr || batch_size == 0) {
        return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    }
    hand_forward_kernel<<<(unsigned)batch_size, plan->blade_count, 0, stream>>>(
        plan->device_left_blade,
        plan->device_right_blade,
        plan->device_output_blade,
        plan->device_sign,
        plan->term_count,
        plan->blade_count,
        device_inputs,
        batch_size,
        device_parameter,
        parameter_on_left != 0,
        device_outputs
    );
    return last_error();
}

extern "C" geo_batch_gp_cuda_status_t geo_hand_cuda_parameter_vjp_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    const double *device_output_cotangents,
    size_t batch_size,
    int parameter_on_left,
    double *device_parameter_cotangent,
    cudaStream_t stream
) {
    if (!valid(plan) || device_inputs == nullptr || device_output_cotangents == nullptr ||
        device_parameter_cotangent == nullptr || batch_size == 0) {
        return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    }
    hand_vjp_kernel<<<(plan->blade_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        plan->device_left_blade,
        plan->device_right_blade,
        plan->device_output_blade,
        plan->device_sign,
        plan->term_count,
        plan->blade_count,
        device_inputs,
        device_output_cotangents,
        batch_size,
        parameter_on_left != 0,
        device_parameter_cotangent
    );
    return last_error();
}

extern "C" geo_batch_gp_cuda_status_t geo_hand_cuda_mse_sgd_step_f64(
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
) {
    if (!valid(plan) || device_inputs == nullptr || device_targets == nullptr ||
        device_parameter == nullptr || device_residuals == nullptr || device_gradient == nullptr ||
        device_loss == nullptr || batch_size == 0 || !isfinite(learning_rate) || learning_rate <= 0.0) {
        return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    }
    if (cudaMemsetAsync(device_loss, 0, sizeof(double), stream) != cudaSuccess) {
        return GEO_BATCH_GP_CUDA_RUNTIME_FAILURE;
    }
    geo_batch_gp_cuda_status_t status = geo_hand_cuda_forward_f64(
        plan, device_inputs, batch_size, device_parameter, parameter_on_left,
        device_residuals, stream
    );
    if (status != GEO_BATCH_GP_CUDA_OK) return status;

    const size_t values = batch_size * plan->blade_count;
    hand_residual_loss_kernel<<<(unsigned)((values + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        device_residuals, device_targets, values, device_loss
    );
    if ((status = last_error()) != GEO_BATCH_GP_CUDA_OK) return status;

    status = geo_hand_cuda_parameter_vjp_f64(
        plan, device_inputs, device_residuals, batch_size, parameter_on_left,
        device_gradient, stream
    );
    if (status != GEO_BATCH_GP_CUDA_OK) return status;

    hand_sgd_kernel<<<(plan->blade_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        device_parameter,
        device_gradient,
        plan->blade_count,
        learning_rate / (double)batch_size
    );
    return last_error();
}
