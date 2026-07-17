#include "geo/cuda.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

static_assert(std::is_standard_layout<geo_cl20_t>::value,
    "geo_cl20_t must be standard layout");
static_assert(sizeof(geo_cl20_t) == sizeof(geo_real_t) * 4u,
    "geo_cl20_t must contain four packed scalars");

struct geo_cuda_context {
    int device_index;
    unsigned int max_grid_x;
    cudaStream_t stream;
};

namespace {

constexpr unsigned int kThreadsPerBlock = 256u;
constexpr int kMinimumCudaRuntime = 13000;
constexpr int kMaximumCudaRuntime = 14000;

__device__ __forceinline__ geo_cl20_t device_add(
    const geo_cl20_t a,
    const geo_cl20_t b
) {
    geo_cl20_t output;
    output.scalar = a.scalar + b.scalar;
    output.e1 = a.e1 + b.e1;
    output.e2 = a.e2 + b.e2;
    output.e12 = a.e12 + b.e12;
    return output;
}

__device__ __forceinline__ geo_cl20_t device_product(
    const geo_cl20_t a,
    const geo_cl20_t b
) {
    geo_cl20_t output;
    output.scalar =
        a.scalar * b.scalar + a.e1 * b.e1 +
        a.e2 * b.e2 - a.e12 * b.e12;
    output.e1 =
        a.scalar * b.e1 + a.e1 * b.scalar -
        a.e2 * b.e12 + a.e12 * b.e2;
    output.e2 =
        a.scalar * b.e2 + a.e2 * b.scalar +
        a.e1 * b.e12 - a.e12 * b.e1;
    output.e12 =
        a.scalar * b.e12 + a.e12 * b.scalar +
        a.e1 * b.e2 - a.e2 * b.e1;
    return output;
}

__device__ __forceinline__ geo_cl20_t device_reverse(
    const geo_cl20_t value
) {
    geo_cl20_t output = value;
    output.e12 = -output.e12;
    return output;
}

__global__ void add_kernel(
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_cl20_t *output,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_add(a[index], b[index]);
}

__global__ void product_kernel(
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_cl20_t *output,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_product(a[index], b[index]);
}

__global__ void reverse_kernel(
    const geo_cl20_t *input,
    geo_cl20_t *output,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_reverse(input[index]);
}

__global__ void dot_kernel(
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_real_t *output,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] =
            a[index].e1 * b[index].e1 + a[index].e2 * b[index].e2;
    }
}

__global__ void wedge_kernel(
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_real_t *output,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] =
            a[index].e1 * b[index].e2 - a[index].e2 * b[index].e1;
    }
}

__global__ void rotor_kernel(
    const geo_cl20_t *rotor,
    const geo_cl20_t *value,
    geo_cl20_t *output,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = device_product(
            device_product(rotor[index], value[index]),
            device_reverse(rotor[index])
        );
    }
}

geo_cuda_status_t map_cuda_error(const cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return GEO_CUDA_SUCCESS;
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
            return GEO_CUDA_NO_DEVICE;
        case cudaErrorMemoryAllocation:
            return GEO_CUDA_ALLOCATION_FAILED;
        case cudaErrorInvalidValue:
            return GEO_CUDA_INVALID_ARGUMENT;
        default:
            return GEO_CUDA_RUNTIME_FAILED;
    }
}

bool checked_byte_count(
    const size_t count,
    const size_t element_size,
    size_t *bytes
) {
    if (bytes == nullptr || element_size == 0u ||
        count > std::numeric_limits<size_t>::max() / element_size) {
        return false;
    }
    *bytes = count * element_size;
    return true;
}

geo_cuda_status_t prepare_batch(
    geo_cuda_context_t *context,
    const size_t count,
    const size_t input_element_size,
    const size_t output_element_size,
    size_t *input_bytes,
    size_t *output_bytes,
    unsigned int *blocks
) {
    if (context == nullptr || input_bytes == nullptr ||
        output_bytes == nullptr || blocks == nullptr) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }

    *input_bytes = 0u;
    *output_bytes = 0u;
    *blocks = 0u;
    if (count == 0u) return GEO_CUDA_SUCCESS;

    if (!checked_byte_count(count, input_element_size, input_bytes) ||
        !checked_byte_count(count, output_element_size, output_bytes)) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }

    const size_t threads = static_cast<size_t>(kThreadsPerBlock);
    const size_t block_count = count / threads +
        (count % threads == 0u ? 0u : 1u);
    if (block_count == 0u || block_count > context->max_grid_x ||
        block_count > static_cast<size_t>(UINT_MAX)) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }

    const cudaError_t selected = cudaSetDevice(context->device_index);
    if (selected != cudaSuccess) return map_cuda_error(selected);
    *blocks = static_cast<unsigned int>(block_count);
    return GEO_CUDA_SUCCESS;
}

void release_binary_buffers(
    void *device_output,
    geo_cl20_t *device_b,
    geo_cl20_t *device_a
) {
    if (device_output != nullptr) (void)cudaFree(device_output);
    if (device_b != nullptr) (void)cudaFree(device_b);
    if (device_a != nullptr) (void)cudaFree(device_a);
}

template <typename Output, typename Launcher>
geo_cuda_status_t run_binary(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    Output *output,
    const size_t count,
    Launcher launcher
) {
    size_t input_bytes = 0u;
    size_t output_bytes = 0u;
    unsigned int blocks = 0u;
    geo_cuda_status_t status = prepare_batch(
        context,
        count,
        sizeof(geo_cl20_t),
        sizeof(Output),
        &input_bytes,
        &output_bytes,
        &blocks
    );
    if (status != GEO_CUDA_SUCCESS || count == 0u) return status;
    if (a == nullptr || b == nullptr || output == nullptr) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }

    geo_cl20_t *device_a = nullptr;
    geo_cl20_t *device_b = nullptr;
    Output *device_output = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void **>(&device_a), input_bytes
    );
    if (error != cudaSuccess) return map_cuda_error(error);
    error = cudaMalloc(reinterpret_cast<void **>(&device_b), input_bytes);
    if (error != cudaSuccess) {
        release_binary_buffers(nullptr, nullptr, device_a);
        return map_cuda_error(error);
    }
    error = cudaMalloc(
        reinterpret_cast<void **>(&device_output), output_bytes
    );
    if (error != cudaSuccess) {
        release_binary_buffers(nullptr, device_b, device_a);
        return map_cuda_error(error);
    }

    error = cudaMemcpyAsync(
        device_a,
        a,
        input_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            device_b,
            b,
            input_bytes,
            cudaMemcpyHostToDevice,
            context->stream
        );
    }
    if (error != cudaSuccess) {
        status = GEO_CUDA_COPY_FAILED;
    } else {
        launcher(
            device_a,
            device_b,
            device_output,
            count,
            blocks,
            context->stream
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            status = GEO_CUDA_LAUNCH_FAILED;
        } else {
            error = cudaMemcpyAsync(
                output,
                device_output,
                output_bytes,
                cudaMemcpyDeviceToHost,
                context->stream
            );
            if (error != cudaSuccess) status = GEO_CUDA_COPY_FAILED;
        }
    }

    if (status == GEO_CUDA_SUCCESS) {
        error = cudaStreamSynchronize(context->stream);
        if (error != cudaSuccess) status = GEO_CUDA_RUNTIME_FAILED;
    }

    release_binary_buffers(device_output, device_b, device_a);
    return status;
}

template <typename Launcher>
geo_cuda_status_t run_unary(
    geo_cuda_context_t *context,
    const geo_cl20_t *input,
    geo_cl20_t *output,
    const size_t count,
    Launcher launcher
) {
    size_t input_bytes = 0u;
    size_t output_bytes = 0u;
    unsigned int blocks = 0u;
    geo_cuda_status_t status = prepare_batch(
        context,
        count,
        sizeof(geo_cl20_t),
        sizeof(geo_cl20_t),
        &input_bytes,
        &output_bytes,
        &blocks
    );
    if (status != GEO_CUDA_SUCCESS || count == 0u) return status;
    if (input == nullptr || output == nullptr) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }

    geo_cl20_t *device_input = nullptr;
    geo_cl20_t *device_output = nullptr;

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void **>(&device_input), input_bytes
    );
    if (error != cudaSuccess) return map_cuda_error(error);
    error = cudaMalloc(
        reinterpret_cast<void **>(&device_output), output_bytes
    );
    if (error != cudaSuccess) {
        (void)cudaFree(device_input);
        return map_cuda_error(error);
    }

    error = cudaMemcpyAsync(
        device_input,
        input,
        input_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );
    if (error != cudaSuccess) {
        status = GEO_CUDA_COPY_FAILED;
    } else {
        launcher(
            device_input,
            device_output,
            count,
            blocks,
            context->stream
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            status = GEO_CUDA_LAUNCH_FAILED;
        } else {
            error = cudaMemcpyAsync(
                output,
                device_output,
                output_bytes,
                cudaMemcpyDeviceToHost,
                context->stream
            );
            if (error != cudaSuccess) status = GEO_CUDA_COPY_FAILED;
        }
    }

    if (status == GEO_CUDA_SUCCESS) {
        error = cudaStreamSynchronize(context->stream);
        if (error != cudaSuccess) status = GEO_CUDA_RUNTIME_FAILED;
    }

    (void)cudaFree(device_output);
    (void)cudaFree(device_input);
    return status;
}

}  // namespace

extern "C" const char *geo_cuda_status_string(const geo_cuda_status_t status) {
    switch (status) {
        case GEO_CUDA_SUCCESS:
            return "success";
        case GEO_CUDA_INVALID_ARGUMENT:
            return "invalid argument";
        case GEO_CUDA_NO_DEVICE:
            return "no CUDA device or insufficient driver";
        case GEO_CUDA_ALLOCATION_FAILED:
            return "CUDA allocation failed";
        case GEO_CUDA_COPY_FAILED:
            return "CUDA copy failed";
        case GEO_CUDA_LAUNCH_FAILED:
            return "CUDA kernel launch failed";
        case GEO_CUDA_RUNTIME_FAILED:
            return "CUDA runtime failure";
        case GEO_CUDA_UNSUPPORTED_TOOLKIT:
            return "CUDA Toolkit 13.x is required";
        default:
            return "unknown CUDA status";
    }
}

extern "C" geo_cuda_status_t geo_cuda_context_create(
    const int device_index,
    geo_cuda_context_t **context
) {
    if (context == nullptr || device_index < 0) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }
    *context = nullptr;

    int runtime_version = 0;
    cudaError_t error = cudaRuntimeGetVersion(&runtime_version);
    if (error != cudaSuccess) return map_cuda_error(error);
    if (runtime_version < kMinimumCudaRuntime ||
        runtime_version >= kMaximumCudaRuntime) {
        return GEO_CUDA_UNSUPPORTED_TOOLKIT;
    }

    int device_count = 0;
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) return map_cuda_error(error);
    if (device_index >= device_count) return GEO_CUDA_NO_DEVICE;

    error = cudaSetDevice(device_index);
    if (error != cudaSuccess) return map_cuda_error(error);

    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, device_index);
    if (error != cudaSuccess || properties.maxGridSize[0] <= 0) {
        return error == cudaSuccess
            ? GEO_CUDA_RUNTIME_FAILED
            : map_cuda_error(error);
    }

    geo_cuda_context_t *created = new (std::nothrow) geo_cuda_context_t;
    if (created == nullptr) return GEO_CUDA_ALLOCATION_FAILED;
    created->device_index = device_index;
    created->max_grid_x = static_cast<unsigned int>(properties.maxGridSize[0]);
    created->stream = nullptr;

    error = cudaStreamCreateWithFlags(
        &created->stream,
        cudaStreamNonBlocking
    );
    if (error != cudaSuccess) {
        delete created;
        return map_cuda_error(error);
    }

    *context = created;
    return GEO_CUDA_SUCCESS;
}

extern "C" void geo_cuda_context_destroy(geo_cuda_context_t *context) {
    if (context == nullptr) return;
    (void)cudaSetDevice(context->device_index);
    if (context->stream != nullptr) {
        (void)cudaStreamDestroy(context->stream);
    }
    delete context;
}

extern "C" geo_cuda_status_t geo_cuda_get_device_info(
    const geo_cuda_context_t *context,
    geo_cuda_device_info_t *info
) {
    if (context == nullptr || info == nullptr) {
        return GEO_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaSetDevice(context->device_index);
    if (error != cudaSuccess) return map_cuda_error(error);

    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, context->device_index);
    if (error != cudaSuccess) return map_cuda_error(error);

    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->name, properties.name, sizeof(info->name) - 1u);
    info->device_index = context->device_index;
    info->compute_major = properties.major;
    info->compute_minor = properties.minor;
    info->multiprocessor_count = properties.multiProcessorCount;
    info->max_threads_per_block = properties.maxThreadsPerBlock;
    info->global_memory_bytes = properties.totalGlobalMem;

    error = cudaRuntimeGetVersion(&info->runtime_version);
    if (error != cudaSuccess) return map_cuda_error(error);
    error = cudaDriverGetVersion(&info->driver_version);
    if (error != cudaSuccess) return map_cuda_error(error);
    return GEO_CUDA_SUCCESS;
}

extern "C" geo_cuda_status_t geo_cuda_cl20_add_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_cl20_t *output,
    const size_t count
) {
    return run_binary(
        context,
        a,
        b,
        output,
        count,
        [](
            const geo_cl20_t *device_a,
            const geo_cl20_t *device_b,
            geo_cl20_t *device_output,
            size_t element_count,
            unsigned int blocks,
            cudaStream_t stream
        ) {
            add_kernel<<<blocks, kThreadsPerBlock, 0u, stream>>>(
                device_a,
                device_b,
                device_output,
                element_count
            );
        }
    );
}

extern "C" geo_cuda_status_t geo_cuda_cl20_product_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_cl20_t *output,
    const size_t count
) {
    return run_binary(
        context,
        a,
        b,
        output,
        count,
        [](
            const geo_cl20_t *device_a,
            const geo_cl20_t *device_b,
            geo_cl20_t *device_output,
            size_t element_count,
            unsigned int blocks,
            cudaStream_t stream
        ) {
            product_kernel<<<blocks, kThreadsPerBlock, 0u, stream>>>(
                device_a,
                device_b,
                device_output,
                element_count
            );
        }
    );
}

extern "C" geo_cuda_status_t geo_cuda_cl20_reverse_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *input,
    geo_cl20_t *output,
    const size_t count
) {
    return run_unary(
        context,
        input,
        output,
        count,
        [](
            const geo_cl20_t *device_input,
            geo_cl20_t *device_output,
            size_t element_count,
            unsigned int blocks,
            cudaStream_t stream
        ) {
            reverse_kernel<<<blocks, kThreadsPerBlock, 0u, stream>>>(
                device_input,
                device_output,
                element_count
            );
        }
    );
}

extern "C" geo_cuda_status_t geo_cuda_cl20_vector_dot_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_real_t *output,
    const size_t count
) {
    return run_binary(
        context,
        a,
        b,
        output,
        count,
        [](
            const geo_cl20_t *device_a,
            const geo_cl20_t *device_b,
            geo_real_t *device_output,
            size_t element_count,
            unsigned int blocks,
            cudaStream_t stream
        ) {
            dot_kernel<<<blocks, kThreadsPerBlock, 0u, stream>>>(
                device_a,
                device_b,
                device_output,
                element_count
            );
        }
    );
}

extern "C" geo_cuda_status_t geo_cuda_cl20_vector_wedge_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_real_t *output,
    const size_t count
) {
    return run_binary(
        context,
        a,
        b,
        output,
        count,
        [](
            const geo_cl20_t *device_a,
            const geo_cl20_t *device_b,
            geo_real_t *device_output,
            size_t element_count,
            unsigned int blocks,
            cudaStream_t stream
        ) {
            wedge_kernel<<<blocks, kThreadsPerBlock, 0u, stream>>>(
                device_a,
                device_b,
                device_output,
                element_count
            );
        }
    );
}

extern "C" geo_cuda_status_t geo_cuda_cl20_rotor_action_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *rotor,
    const geo_cl20_t *value,
    geo_cl20_t *output,
    const size_t count
) {
    return run_binary(
        context,
        rotor,
        value,
        output,
        count,
        [](
            const geo_cl20_t *device_rotor,
            const geo_cl20_t *device_value,
            geo_cl20_t *device_output,
            size_t element_count,
            unsigned int blocks,
            cudaStream_t stream
        ) {
            rotor_kernel<<<blocks, kThreadsPerBlock, 0u, stream>>>(
                device_rotor,
                device_value,
                device_output,
                element_count
            );
        }
    );
}
