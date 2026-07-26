#include "geo/tensor_optimizer_cuda.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr unsigned int GEO_OPTIMIZER_BLOCK_SIZE = 256u;
constexpr unsigned int GEO_OPTIMIZER_MAX_BLOCKS = 65535u;

bool valid_config(geo_tensor_adamw_config config) {
    return config.learning_rate >= 0.0f &&
           config.beta1 >= 0.0f && config.beta1 < 1.0f &&
           config.beta2 >= 0.0f && config.beta2 < 1.0f &&
           config.epsilon > 0.0f &&
           config.weight_decay >= 0.0f &&
           config.max_grad_norm >= 0.0f &&
           config.step > 0u;
}

unsigned int launch_blocks(size_t work_items) {
    size_t blocks = (work_items + GEO_OPTIMIZER_BLOCK_SIZE - 1u) /
                    GEO_OPTIMIZER_BLOCK_SIZE;
    if (blocks > GEO_OPTIMIZER_MAX_BLOCKS) {
        blocks = GEO_OPTIMIZER_MAX_BLOCKS;
    }
    return static_cast<unsigned int>(blocks);
}

__global__ void grad_square_kernel(
    const float *gradient,
    size_t count,
    float *sum_square
) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    float local = 0.0f;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float value = gradient[index];
        local += value * value;
    }
    atomicAdd(sum_square, local);
}

__global__ void grad_clip_finalize_kernel(
    const float *sum_square,
    float max_grad_norm,
    float *clip_scale
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    if (max_grad_norm == 0.0f) {
        *clip_scale = 1.0f;
        return;
    }
    const float norm = sqrtf(*sum_square);
    *clip_scale = fminf(1.0f, max_grad_norm / (norm + 1e-6f));
}

__global__ void adamw_step_fast_kernel(
    float *parameter,
    const float *gradient,
    float *first_moment,
    float *second_moment,
    size_t count,
    const float *clip_scale,
    float decay,
    float step_size,
    float beta1,
    float beta2,
    float epsilon
) {
    const float clip = clip_scale ? *clip_scale : 1.0f;
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float grad = gradient[index] * clip;
        const float m1 = beta1 * first_moment[index] + one_minus_beta1 * grad;
        const float m2 = beta2 * second_moment[index] + one_minus_beta2 * grad * grad;
        first_moment[index] = m1;
        second_moment[index] = m2;

        parameter[index] = parameter[index] * decay - step_size * m1 / (sqrtf(m2) + epsilon);
    }
}

__global__ void adamw_step_fused_kernel(
    float **parameters,
    const float **gradients,
    float **first_moments,
    float **second_moments,
    const size_t *counts,
    size_t num_tensors,
    const float *clip_scale,
    float decay,
    float step_size,
    float beta1,
    float beta2,
    float epsilon
) {
    const float clip = clip_scale ? *clip_scale : 1.0f;
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;

    for (size_t t = blockIdx.y; t < num_tensors; t += gridDim.y) {
        float *param = parameters[t];
        const float *grad_arr = gradients[t];
        float *m1_arr = first_moments[t];
        float *m2_arr = second_moments[t];
        const size_t count = counts[t];

        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count;
             index += stride) {
            const float grad = grad_arr[index] * clip;
            const float m1 = beta1 * m1_arr[index] + one_minus_beta1 * grad;
            const float m2 = beta2 * m2_arr[index] + one_minus_beta2 * grad * grad;
            m1_arr[index] = m1;
            m2_arr[index] = m2;

            param[index] = param[index] * decay - step_size * m1 / (sqrtf(m2) + epsilon);
        }
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

}  // namespace

extern "C" geo_tensor_status geo_tensor_grad_square_cuda_accumulate(
    const float *gradient,
    size_t count,
    float *sum_square,
    void *stream
) {
    if (gradient == nullptr || sum_square == nullptr || count == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);
    grad_square_kernel<<<launch_blocks(count), GEO_OPTIMIZER_BLOCK_SIZE, 0, custream>>>(
        gradient, count, sum_square
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_grad_clip_cuda_finalize(
    const float *sum_square,
    float max_grad_norm,
    float *clip_scale,
    void *stream
) {
    if (sum_square == nullptr || clip_scale == nullptr || max_grad_norm < 0.0f) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);
    grad_clip_finalize_kernel<<<1, 1, 0, custream>>>(
        sum_square, max_grad_norm, clip_scale
    );
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_adamw_cuda_step(
    float *parameter,
    const float *gradient,
    float *first_moment,
    float *second_moment,
    size_t count,
    const float *clip_scale,
    geo_tensor_adamw_config config,
    void *stream
) {
    if (parameter == nullptr || gradient == nullptr || first_moment == nullptr ||
        second_moment == nullptr || count == 0u || !valid_config(config)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);

    const float step_f = static_cast<float>(config.step);
    const float bc1 = 1.0f - powf(config.beta1, step_f);
    const float bc2 = 1.0f - powf(config.beta2, step_f);
    const float decay = 1.0f - config.learning_rate * config.weight_decay;
    const float step_size = config.learning_rate * (sqrtf(bc2) / bc1);
    const float eps_corrected = config.epsilon * sqrtf(bc2);

    adamw_step_fast_kernel<<<launch_blocks(count), GEO_OPTIMIZER_BLOCK_SIZE, 0, custream>>>(
        parameter, gradient, first_moment, second_moment, count,
        clip_scale, decay, step_size, config.beta1, config.beta2, eps_corrected
    );

    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_adamw_cuda_step_fused(
    float **parameters,
    const float **gradients,
    float **first_moments,
    float **second_moments,
    const size_t *counts,
    size_t num_tensors,
    const float *clip_scale,
    geo_tensor_adamw_config config,
    void *stream
) {
    if (parameters == nullptr || gradients == nullptr || first_moments == nullptr ||
        second_moments == nullptr || counts == nullptr || num_tensors == 0u || !valid_config(config)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);

    const float step_f = static_cast<float>(config.step);
    const float bc1 = 1.0f - powf(config.beta1, step_f);
    const float bc2 = 1.0f - powf(config.beta2, step_f);
    const float decay = 1.0f - config.learning_rate * config.weight_decay;
    const float step_size = config.learning_rate * (sqrtf(bc2) / bc1);
    const float eps_corrected = config.epsilon * sqrtf(bc2);

    dim3 grid(128u, static_cast<unsigned int>(num_tensors));
    dim3 block(GEO_OPTIMIZER_BLOCK_SIZE);

    adamw_step_fused_kernel<<<grid, block, 0, custream>>>(
        parameters, gradients, first_moments, second_moments, counts, num_tensors,
        clip_scale, decay, step_size, config.beta1, config.beta2, eps_corrected
    );

    return launch_status();
}
