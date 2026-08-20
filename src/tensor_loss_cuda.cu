#include "geo/tensor_loss_cuda.h"
#include "geo/geo_reduction_cuda.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr unsigned int GEO_LOSS_BLOCK_SIZE = 256u;

bool valid_shape(geo_tensor_cross_entropy_shape shape) {
    return shape.rows > 0u && shape.classes > 0u &&
           shape.rows <= SIZE_MAX / shape.classes;
}

// Parallel 2D Grid: Each blockIdx.x handles one row, threadIdx.x handles classes in parallel
__global__ void cross_entropy_forward_parallel_kernel(
    const float *logits,
    const int64_t *targets,
    int64_t ignore_index,
    float *loss_sum,
    float *probabilities,
    float *valid_count,
    size_t rows,
    size_t classes
) {
    const size_t row = blockIdx.x;
    if (row >= rows) return;

    const size_t base = row * classes;
    const int64_t target = targets[row];

    if (target == ignore_index) {
        for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
            probabilities[base + c] = 0.0f;
        }
        return;
    }

    if (target < 0 || static_cast<uint64_t>(target) >= static_cast<uint64_t>(classes)) {
        if (threadIdx.x == 0) {
            const float nan_val = nanf("");
            atomicExch(loss_sum, nan_val);
            atomicExch(valid_count, nan_val);
        }
        for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
            probabilities[base + c] = nanf("");
        }
        return;
    }

    // Step 1: Parallel Max Reduction across block
    float thread_max = -1e30f;
    for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
        thread_max = fmaxf(thread_max, logits[base + c]);
    }
    const float block_max = geo_block_reduce_max(thread_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = block_max;
    }
    __syncthreads();

    const float max_logit = s_max;

    // Step 2: Parallel Exp & Sum Reduction
    float thread_sum = 0.0f;
    for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
        const float exp_val = expf(logits[base + c] - max_logit);
        probabilities[base + c] = exp_val;
        thread_sum += exp_val;
    }
    const float block_sum = geo_block_reduce_sum(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = block_sum;
    }
    __syncthreads();

    const float exp_sum = s_sum;
    const float inv_exp_sum = 1.0f / exp_sum;

    // Step 3: Parallel Normalization
    for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
        probabilities[base + c] *= inv_exp_sum;
    }

    // Step 4: Loss Accumulation
    if (threadIdx.x == 0) {
        const float row_loss = max_logit + logf(exp_sum) - logits[base + static_cast<size_t>(target)];
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

__global__ void cross_entropy_vjp_parallel_kernel(
    const float *probabilities,
    const int64_t *targets,
    int64_t ignore_index,
    const float *normalizer,
    const float *grad_loss,
    float *grad_logits,
    size_t rows,
    size_t classes
) {
    const size_t row = blockIdx.x;
    if (row >= rows) return;

    const float norm = *normalizer;
    const float g_loss = *grad_loss;
    const float scale = isnan(norm) ? nanf("") : (g_loss * norm);

    const size_t base = row * classes;
    const int64_t target = targets[row];

    if (target == ignore_index) {
        for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
            grad_logits[base + c] = 0.0f;
        }
        return;
    }

    for (size_t c = threadIdx.x; c < classes; c += blockDim.x) {
        const float p = probabilities[base + c];
        const float indicator = (static_cast<int64_t>(c) == target) ? 1.0f : 0.0f;
        grad_logits[base + c] = scale * (p - indicator);
    }
}

__global__ void cross_entropy_forward_serial_kernel(
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
            for (size_t c = 0u; c < classes; ++c) {
                probabilities[base + c] = 0.0f;
            }
            continue;
        }
        if (target < 0 || static_cast<uint64_t>(target) >= static_cast<uint64_t>(classes)) {
            const float nan_val = nanf("");
            atomicExch(loss_sum, nan_val);
            atomicExch(valid_count, nan_val);
            for (size_t c = 0u; c < classes; ++c) {
                probabilities[base + c] = nan_val;
            }
            continue;
        }

        float max_logit = logits[base];
        for (size_t c = 1u; c < classes; ++c) {
            max_logit = fmaxf(max_logit, logits[base + c]);
        }

        float exp_sum = 0.0f;
        for (size_t c = 0u; c < classes; ++c) {
            const float exp_val = expf(logits[base + c] - max_logit);
            probabilities[base + c] = exp_val;
            exp_sum += exp_val;
        }

        const float inv_exp_sum = 1.0f / exp_sum;
        for (size_t c = 0u; c < classes; ++c) {
            probabilities[base + c] *= inv_exp_sum;
        }

        const float row_loss = max_logit + logf(exp_sum) - logits[base + static_cast<size_t>(target)];
        atomicAdd(loss_sum, row_loss);
        atomicAdd(valid_count, 1.0f);
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
        probabilities == nullptr || normalizer == nullptr ||
        !valid_shape(shape)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);
    cudaMemsetAsync(loss, 0, sizeof(float), custream);
    cudaMemsetAsync(normalizer, 0, sizeof(float), custream);

    if (shape.classes <= 512u) {
        // Small vocabulary <= 512: 1D serial row kernel
        const size_t work_items = shape.rows;
        const unsigned int block_size = 128u;
        const unsigned int grid_size = static_cast<unsigned int>((work_items + block_size - 1u) / block_size);
        cross_entropy_forward_serial_kernel<<<grid_size, block_size, 0, custream>>>(
            logits, targets, ignore_index, loss, probabilities, normalizer,
            shape.rows, shape.classes
        );
    } else {
        // Medium/Large vocabulary > 512: 2D parallel block reduction grid
        dim3 grid(static_cast<unsigned int>(shape.rows));
        dim3 block(GEO_LOSS_BLOCK_SIZE);

        cross_entropy_forward_parallel_kernel<<<grid, block, 0, custream>>>(
            logits, targets, ignore_index, loss, probabilities, normalizer,
            shape.rows, shape.classes
        );
    }

    cross_entropy_finalize_kernel<<<1, 1, 0, custream>>>(loss, normalizer);
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

    cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned int>(shape.rows));
    dim3 block(GEO_LOSS_BLOCK_SIZE);

    cross_entropy_vjp_parallel_kernel<<<grid, block, 0, custream>>>(
        probabilities, targets, ignore_index, normalizer, grad_loss, grad_logits,
        shape.rows, shape.classes
    );
    return launch_status();
}
