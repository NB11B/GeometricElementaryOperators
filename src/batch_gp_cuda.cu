#include "geo/batch_gp_cuda.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

namespace {

constexpr int kThreads = 256;

__device__ unsigned popcount_u8(uint8_t value) {
    return (unsigned)__popc((unsigned)value);
}

__device__ int gp_sign_device(uint8_t left, uint8_t right, const int8_t *signature, uint8_t dimension) {
    int sign = 1;
    for (uint8_t axis = 0; axis < dimension; ++axis) {
        if (((left >> axis) & 1u) != 0u) {
            const uint8_t lower = (uint8_t)(right & (uint8_t)((1u << axis) - 1u));
            if ((popcount_u8(lower) & 1u) != 0u) sign = -sign;
            if (((right >> axis) & 1u) != 0u) sign *= signature[axis];
        }
    }
    return sign;
}

__global__ void reference_forward_kernel(
    const double *inputs,
    const double *parameter,
    double *outputs,
    size_t batch_size,
    uint16_t blades,
    uint8_t dimension,
    const int8_t *signature,
    int parameter_on_left
) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t values = batch_size * (size_t)blades;
    if (index >= values) return;
    const size_t sample = index / blades;
    const uint8_t output_blade = (uint8_t)(index % blades);
    double sum = 0.0;
    for (uint16_t input_blade = 0; input_blade < blades; ++input_blade) {
        const uint8_t parameter_blade = (uint8_t)(input_blade ^ output_blade);
        const uint8_t left = parameter_on_left ? parameter_blade : (uint8_t)input_blade;
        const uint8_t right = parameter_on_left ? (uint8_t)input_blade : parameter_blade;
        sum += inputs[sample * blades + input_blade] * parameter[parameter_blade] *
            (double)gp_sign_device(left, right, signature, dimension);
    }
    outputs[index] = sum;
}

__global__ void planned_forward_kernel(
    const double *inputs,
    const double *parameter,
    double *outputs,
    size_t batch_size,
    uint16_t blades,
    const int8_t *signs,
    int parameter_on_left
) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t values = batch_size * (size_t)blades;
    if (index >= values) return;
    const size_t sample = index / blades;
    const uint8_t output_blade = (uint8_t)(index % blades);
    double sum = 0.0;
    for (uint16_t input_blade = 0; input_blade < blades; ++input_blade) {
        const uint8_t parameter_blade = (uint8_t)(input_blade ^ output_blade);
        const uint16_t left = parameter_on_left ? parameter_blade : input_blade;
        const uint16_t right = parameter_on_left ? input_blade : parameter_blade;
        const size_t term = (size_t)left * blades + right;
        sum += inputs[sample * blades + input_blade] * parameter[parameter_blade] * (double)signs[term];
    }
    outputs[index] = sum;
}

__global__ void reference_parameter_vjp_kernel(
    const double *inputs,
    const double *cotangents,
    double *gradient,
    size_t batch_size,
    uint16_t blades,
    uint8_t dimension,
    const int8_t *signature,
    int parameter_on_left
) {
    const uint16_t parameter_blade = (uint16_t)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (parameter_blade >= blades) return;
    double sum = 0.0;
    for (size_t sample = 0; sample < batch_size; ++sample) {
        for (uint16_t input_blade = 0; input_blade < blades; ++input_blade) {
            const uint16_t output_blade = (uint16_t)(input_blade ^ parameter_blade);
            const uint8_t left = parameter_on_left ? (uint8_t)parameter_blade : (uint8_t)input_blade;
            const uint8_t right = parameter_on_left ? (uint8_t)input_blade : (uint8_t)parameter_blade;
            sum += inputs[sample * blades + input_blade] *
                cotangents[sample * blades + output_blade] *
                (double)gp_sign_device(left, right, signature, dimension);
        }
    }
    gradient[parameter_blade] = sum;
}

__global__ void planned_parameter_vjp_kernel(
    const double *inputs,
    const double *cotangents,
    double *gradient,
    size_t batch_size,
    uint16_t blades,
    const int8_t *signs,
    int parameter_on_left
) {
    const uint16_t parameter_blade = (uint16_t)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (parameter_blade >= blades) return;
    double sum = 0.0;
    for (size_t sample = 0; sample < batch_size; ++sample) {
        for (uint16_t input_blade = 0; input_blade < blades; ++input_blade) {
            const uint16_t output_blade = (uint16_t)(input_blade ^ parameter_blade);
            const uint16_t left = parameter_on_left ? parameter_blade : input_blade;
            const uint16_t right = parameter_on_left ? input_blade : parameter_blade;
            const size_t term = (size_t)left * blades + right;
            sum += inputs[sample * blades + input_blade] *
                cotangents[sample * blades + output_blade] * (double)signs[term];
        }
    }
    gradient[parameter_blade] = sum;
}

__global__ void residual_loss_kernel(double *outputs, const double *targets, size_t count, double *loss) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const double residual = outputs[index] - targets[index];
    outputs[index] = residual;
    atomicAdd(loss, 0.5 * residual * residual);
}

__global__ void sgd_update_kernel(double *parameter, const double *gradient, uint16_t blades, double scale) {
    const uint16_t index = (uint16_t)((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (index < blades) parameter[index] -= scale * gradient[index];
}

bool valid_plan(const geo_batch_gp_cuda_plan_t *plan) {
    return plan != nullptr && plan->abi_version == GEO_BATCH_GP_CUDA_ABI_VERSION &&
        plan->blade_count > 0 && plan->term_count == plan->blade_count * plan->blade_count &&
        plan->device_signature != nullptr && plan->device_sign != nullptr;
}

geo_batch_gp_cuda_status_t launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_BATCH_GP_CUDA_OK : GEO_BATCH_GP_CUDA_RUNTIME_FAILURE;
}

geo_batch_gp_cuda_status_t mse_sgd_step(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    const double *device_targets,
    size_t batch_size,
    double learning_rate,
    int parameter_on_left,
    int use_reference,
    double *device_parameter,
    double *device_residuals,
    double *device_gradient,
    double *device_loss,
    cudaStream_t stream
) {
    if (!valid_plan(plan) || device_inputs == nullptr || device_targets == nullptr ||
        device_parameter == nullptr || device_residuals == nullptr || device_gradient == nullptr ||
        device_loss == nullptr || batch_size == 0 || !isfinite(learning_rate) || learning_rate <= 0.0) {
        return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    }
    if (cudaMemsetAsync(device_loss, 0, sizeof(double), stream) != cudaSuccess) {
        return GEO_BATCH_GP_CUDA_RUNTIME_FAILURE;
    }
    geo_batch_gp_cuda_status_t status = use_reference ?
        geo_batch_gp_cuda_reference_forward_f64(plan, device_inputs, batch_size, device_parameter,
            parameter_on_left, device_residuals, stream) :
        geo_batch_gp_cuda_planned_forward_f64(plan, device_inputs, batch_size, device_parameter,
            parameter_on_left, device_residuals, stream);
    if (status != GEO_BATCH_GP_CUDA_OK) return status;

    const size_t values = batch_size * plan->blade_count;
    residual_loss_kernel<<<(unsigned)((values + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        device_residuals, device_targets, values, device_loss);
    if ((status = launch_status()) != GEO_BATCH_GP_CUDA_OK) return status;

    status = use_reference ?
        geo_batch_gp_cuda_reference_parameter_vjp_f64(plan, device_inputs, device_residuals,
            batch_size, parameter_on_left, device_gradient, stream) :
        geo_batch_gp_cuda_parameter_vjp_f64(plan, device_inputs, device_residuals,
            batch_size, parameter_on_left, device_gradient, stream);
    if (status != GEO_BATCH_GP_CUDA_OK) return status;

    sgd_update_kernel<<<(plan->blade_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        device_parameter, device_gradient, plan->blade_count, learning_rate / (double)batch_size);
    return launch_status();
}

}  // namespace

extern "C" uint32_t geo_batch_gp_cuda_abi_version(void) {
    return GEO_BATCH_GP_CUDA_ABI_VERSION;
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_plan_upload(
    geo_batch_gp_cuda_plan_t *device_plan,
    const geo_batch_gp_plan_t *host_plan
) {
    if (device_plan == nullptr || host_plan == nullptr ||
        host_plan->abi_version != GEO_BATCH_GP_ABI_VERSION || host_plan->blade_count == 0 ||
        host_plan->term_count != host_plan->blade_count * host_plan->blade_count) {
        return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    }
    memset(device_plan, 0, sizeof(*device_plan));
    device_plan->abi_version = GEO_BATCH_GP_CUDA_ABI_VERSION;
    device_plan->blade_count = host_plan->blade_count;
    device_plan->term_count = host_plan->term_count;
    device_plan->dimension = host_plan->dimension;
    memcpy(device_plan->signature, host_plan->signature, sizeof(device_plan->signature));

    const size_t term_bytes = host_plan->term_count * sizeof(uint8_t);
    if (cudaMalloc((void **)&device_plan->device_signature, host_plan->dimension * sizeof(int8_t)) != cudaSuccess ||
        cudaMalloc((void **)&device_plan->device_left_blade, term_bytes) != cudaSuccess ||
        cudaMalloc((void **)&device_plan->device_right_blade, term_bytes) != cudaSuccess ||
        cudaMalloc((void **)&device_plan->device_output_blade, term_bytes) != cudaSuccess ||
        cudaMalloc((void **)&device_plan->device_sign, host_plan->term_count * sizeof(int8_t)) != cudaSuccess) {
        geo_batch_gp_cuda_plan_destroy(device_plan);
        return GEO_BATCH_GP_CUDA_ALLOCATION_FAILURE;
    }
    if (cudaMemcpy(device_plan->device_signature, host_plan->signature,
            host_plan->dimension * sizeof(int8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_plan->device_left_blade, host_plan->left_blade, term_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_plan->device_right_blade, host_plan->right_blade, term_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_plan->device_output_blade, host_plan->output_blade, term_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_plan->device_sign, host_plan->sign,
            host_plan->term_count * sizeof(int8_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        geo_batch_gp_cuda_plan_destroy(device_plan);
        return GEO_BATCH_GP_CUDA_RUNTIME_FAILURE;
    }
    return GEO_BATCH_GP_CUDA_OK;
}

extern "C" void geo_batch_gp_cuda_plan_destroy(geo_batch_gp_cuda_plan_t *plan) {
    if (plan == nullptr) return;
    cudaFree(plan->device_signature);
    cudaFree(plan->device_left_blade);
    cudaFree(plan->device_right_blade);
    cudaFree(plan->device_output_blade);
    cudaFree(plan->device_sign);
    memset(plan, 0, sizeof(*plan));
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_reference_forward_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    size_t batch_size,
    const double *device_parameter,
    int parameter_on_left,
    double *device_outputs,
    cudaStream_t stream
) {
    if (!valid_plan(plan) || device_inputs == nullptr || device_parameter == nullptr ||
        device_outputs == nullptr || batch_size == 0) return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    const size_t values = batch_size * plan->blade_count;
    reference_forward_kernel<<<(unsigned)((values + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        device_inputs, device_parameter, device_outputs, batch_size, plan->blade_count,
        plan->dimension, plan->device_signature, parameter_on_left != 0);
    return launch_status();
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_planned_forward_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    size_t batch_size,
    const double *device_parameter,
    int parameter_on_left,
    double *device_outputs,
    cudaStream_t stream
) {
    if (!valid_plan(plan) || device_inputs == nullptr || device_parameter == nullptr ||
        device_outputs == nullptr || batch_size == 0) return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    const size_t values = batch_size * plan->blade_count;
    planned_forward_kernel<<<(unsigned)((values + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        device_inputs, device_parameter, device_outputs, batch_size, plan->blade_count,
        plan->device_sign, parameter_on_left != 0);
    return launch_status();
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_reference_parameter_vjp_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    const double *device_output_cotangents,
    size_t batch_size,
    int parameter_on_left,
    double *device_parameter_cotangent,
    cudaStream_t stream
) {
    if (!valid_plan(plan) || device_inputs == nullptr || device_output_cotangents == nullptr ||
        device_parameter_cotangent == nullptr || batch_size == 0) return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    reference_parameter_vjp_kernel<<<(plan->blade_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        device_inputs, device_output_cotangents, device_parameter_cotangent, batch_size,
        plan->blade_count, plan->dimension, plan->device_signature, parameter_on_left != 0);
    return launch_status();
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_parameter_vjp_f64(
    const geo_batch_gp_cuda_plan_t *plan,
    const double *device_inputs,
    const double *device_output_cotangents,
    size_t batch_size,
    int parameter_on_left,
    double *device_parameter_cotangent,
    cudaStream_t stream
) {
    if (!valid_plan(plan) || device_inputs == nullptr || device_output_cotangents == nullptr ||
        device_parameter_cotangent == nullptr || batch_size == 0) return GEO_BATCH_GP_CUDA_INVALID_ARGUMENT;
    planned_parameter_vjp_kernel<<<(plan->blade_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        device_inputs, device_output_cotangents, device_parameter_cotangent, batch_size,
        plan->blade_count, plan->device_sign, parameter_on_left != 0);
    return launch_status();
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_reference_mse_sgd_step_f64(
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
    return mse_sgd_step(plan, device_inputs, device_targets, batch_size, learning_rate,
        parameter_on_left, 1, device_parameter, device_residuals, device_gradient, device_loss, stream);
}

extern "C" geo_batch_gp_cuda_status_t geo_batch_gp_cuda_mse_sgd_step_f64(
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
    return mse_sgd_step(plan, device_inputs, device_targets, batch_size, learning_rate,
        parameter_on_left, 0, device_parameter, device_residuals, device_gradient, device_loss, stream);
}

extern "C" const char *geo_batch_gp_cuda_status_string(geo_batch_gp_cuda_status_t status) {
    switch (status) {
        case GEO_BATCH_GP_CUDA_OK: return "ok";
        case GEO_BATCH_GP_CUDA_INVALID_ARGUMENT: return "invalid_argument";
        case GEO_BATCH_GP_CUDA_ALLOCATION_FAILURE: return "allocation_failure";
        case GEO_BATCH_GP_CUDA_RUNTIME_FAILURE: return "runtime_failure";
        case GEO_BATCH_GP_CUDA_NUMERIC_FAILURE: return "numeric_failure";
        default: return "unknown";
    }
}
