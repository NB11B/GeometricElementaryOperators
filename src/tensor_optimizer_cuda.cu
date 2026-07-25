#include "geo/tensor_optimizer_cuda.h"

#include <cuda_runtime.h>

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

__global__ void adamw_step_kernel(
    float *parameter,
    const float *gradient,
    float *first_moment,
    float *second_moment,
    size_t count,
    const float *clip_scale,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    uint64_t step
) {
    const float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(step));
    const float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(step));
    const float decay = 1.0f - learning_rate * weight_decay;
    const float clip = *clip_scale;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const float grad = gradient[index] * clip;
        const float moment1 = beta1 * first_moment[index] + (1.0f - beta1) * grad;
        const float moment2 = beta2 * second_moment[index] + (1.0f - beta2) * grad * grad;
        first_moment[index] = moment1;
        second_moment[index] = moment2;
        const float corrected1 = moment1 / bias_correction1;
        const float corrected2 = moment2 / bias_correction2;
        parameter[index] = parameter[index] * decay -
            learning_rate * corrected1 / (sqrtf(corrected2) + epsilon);
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
    grad_square_kernel<<<
        launch_blocks(count), GEO_OPTIMIZER_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(gradient, count, sum_square);
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
    grad_clip_finalize_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
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
        second_moment == nullptr || clip_scale == nullptr || count == 0u ||
        !valid_config(config)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    adamw_step_kernel<<<
        launch_blocks(count), GEO_OPTIMIZER_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        parameter, gradient, first_moment, second_moment, count, clip_scale,
        static_cast<float>(config.learning_rate),
        static_cast<float>(config.beta1),
        static_cast<float>(config.beta2),
        static_cast<float>(config.epsilon),
        static_cast<float>(config.weight_decay),
        config.step
    );
    return launch_status();
}
