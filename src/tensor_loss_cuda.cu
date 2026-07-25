#include "geo/tensor_loss_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr unsigned int GEO_LOSS_BLOCK_SIZE = 128u;
constexpr unsigned int GEO_LOSS_MAX_BLOCKS = 65535u;

bool valid_shape(geo_tensor_cross_entropy_shape shape) {
    return shape.rows > 0u && shape.classes > 0u &&
           shape.rows <= SIZE_MAX / shape.classes;
}

unsigned int launch_blocks(size_t work_items) {
    size_t blocks = (work_items + GEO_LOSS_BLOCK_SIZE - 1u) / GEO_LOSS_BLOCK_SIZE;
    if (blocks > GEO_LOSS_MAX_BLOCKS) {
        blocks = GEO_LOSS_MAX_BLOCKS;
    }
    return static_cast<unsigned int>(blocks);
}

__global__ void cross_entropy_forward_kernel(
    const float *logits,
    const int64_t *targets,
    int64_t ignore_index,
    float *loss_sum,
    float *probabilities,
    float *valid_count,
    size_t rows,
    size_t classes
) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += stride) {
        const size_t base = row * classes;
        const int64_t target = targets[row];
        if (target == ignore_index) {
            for (size_t class_index = 0u; class_index < classes; ++class_index) {
                probabilities[base + class_index] = 0.0f;
            }
            continue;
        }
        if (target < 0 || static_cast<uint64_t>(target) >= static_cast<uint64_t>(classes)) {
            const float nan_value = nanf("");
            atomicExch(loss_sum, nan_value);
            atomicExch(valid_count, nan_value);
            for (size_t class_index = 0u; class_index < classes; ++class_index) {
                probabilities[base + class_index] = nan_value;
            }
            continue;
        }

        float max_logit = logits[base];
        for (size_t class_index = 1u; class_index < classes; ++class_index) {
            max_logit = fmaxf(max_logit, logits[base + class_index]);
        }

        float exponential_sum = 0.0f;
        for (size_t class_index = 0u; class_index < classes; ++class_index) {
            const float exponential = expf(logits[base + class_index] - max_logit);
            probabilities[base + class_index] = exponential;
            exponential_sum += exponential;
        }
        for (size_t class_index = 0u; class_index < classes; ++class_index) {
            probabilities[base + class_index] /= exponential_sum;
        }

        const float row_loss = max_logit + logf(exponential_sum) -
                               logits[base + static_cast<size_t>(target)];
        atomicAdd(loss_sum, row_loss);
        atomicAdd(valid_count, 1.0f);
    }
}

__global__ void cross_entropy_finalize_kernel(float *loss, float *normalizer) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const float count = *normalizer;
    if (isnan(count)) {
        *loss = nanf("");
        *normalizer = nanf("");
    } else if (count == 0.0f) {
        *loss = nanf("");
        *normalizer = 0.0f;
    } else {
        *normalizer = 1.0f / count;
        *loss *= *normalizer;
    }
}

__global__ void cross_entropy_vjp_kernel(
    const float *probabilities,
    const int64_t *targets,
    int64_t ignore_index,
    const float *normalizer,
    const float *grad_loss,
    float *grad_logits,
    size_t rows,
    size_t classes
) {
    const size_t total = rows * classes;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const float scale = (*normalizer) * (*grad_loss);
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += stride) {
        const size_t row = index / classes;
        const size_t class_index = index - row * classes;
        const int64_t target = targets[row];
        if (target == ignore_index || *normalizer == 0.0f) {
            grad_logits[index] = 0.0f;
        } else if (target < 0 ||
                   static_cast<uint64_t>(target) >= static_cast<uint64_t>(classes)) {
            grad_logits[index] = nanf("");
        } else {
            const float indicator =
                class_index == static_cast<size_t>(target) ? 1.0f : 0.0f;
            grad_logits[index] = scale * (probabilities[index] - indicator);
        }
    }
}

geo_tensor_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

}  // namespace

extern "C" geo_tensor_status geo_tensor_cross_entropy_cuda_forward(
    const float *logits,
    const int64_t *targets,
    int64_t ignore_index,
    float *loss,
    float *probabilities,
    float *normalizer,
    geo_tensor_cross_entropy_shape shape,
    void *stream
) {
    if (logits == nullptr || targets == nullptr || loss == nullptr ||
        probabilities == nullptr || normalizer == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cudaMemsetAsync(loss, 0, sizeof(float), cuda_stream) != cudaSuccess ||
        cudaMemsetAsync(normalizer, 0, sizeof(float), cuda_stream) != cudaSuccess) {
        return GEO_TENSOR_CUDA_ERROR;
    }

    cross_entropy_forward_kernel<<<
        launch_blocks(shape.rows), GEO_LOSS_BLOCK_SIZE, 0, cuda_stream
    >>>(
        logits, targets, ignore_index, loss, probabilities, normalizer,
        shape.rows, shape.classes
    );
    if (cudaGetLastError() != cudaSuccess) {
        return GEO_TENSOR_CUDA_ERROR;
    }
    cross_entropy_finalize_kernel<<<1, 1, 0, cuda_stream>>>(loss, normalizer);
    return launch_status();
}

extern "C" geo_tensor_status geo_tensor_cross_entropy_cuda_vjp(
    const float *probabilities,
    const int64_t *targets,
    int64_t ignore_index,
    const float *normalizer,
    const float *grad_loss,
    float *grad_logits,
    geo_tensor_cross_entropy_shape shape,
    void *stream
) {
    if (probabilities == nullptr || targets == nullptr || normalizer == nullptr ||
        grad_loss == nullptr || grad_logits == nullptr || !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    const size_t total = shape.rows * shape.classes;
    cross_entropy_vjp_kernel<<<
        launch_blocks(total), GEO_LOSS_BLOCK_SIZE, 0,
        reinterpret_cast<cudaStream_t>(stream)
    >>>(
        probabilities, targets, ignore_index, normalizer, grad_loss,
        grad_logits, shape.rows, shape.classes
    );
    return launch_status();
}
