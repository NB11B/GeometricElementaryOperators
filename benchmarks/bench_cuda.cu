#include "geo/cl20.h"
#include "geo/cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kThreadsPerBlock = 256u;

enum class Operation {
    Addition,
    GeometricProduct,
    Reverse,
    VectorDot,
    VectorWedge,
    RotorAction
};

struct Options {
    int device = 0;
    std::size_t batch = 262144u;
    unsigned int iterations = 100u;
    unsigned int warmup = 10u;
    std::string operation = "all";
    std::string csv_path;
};

struct BatchLayout {
    std::size_t mv_bytes = 0u;
    std::size_t scalar_bytes = 0u;
    unsigned int blocks = 0u;
};

struct Timing {
    float upload_ms = 0.0f;
    float kernel_total_ms = 0.0f;
    float download_ms = 0.0f;
    std::size_t upload_bytes = 0u;
    std::size_t download_bytes = 0u;
    std::size_t logical_kernel_bytes = 0u;
};

struct ErrorStats {
    double max_absolute = 0.0;
    double max_relative = 0.0;
    std::size_t mismatches = 0u;
};

struct Resources {
    int device = 0;
    geo_cuda_context_t *api_context = nullptr;
    cudaStream_t stream = nullptr;
    geo_cl20_t *device_a = nullptr;
    geo_cl20_t *device_b = nullptr;
    geo_cl20_t *device_rotor = nullptr;
    geo_cl20_t *device_mv_output = nullptr;
    geo_real_t *device_scalar_output = nullptr;

    ~Resources() {
        (void)cudaSetDevice(device);
        if (device_scalar_output != nullptr) (void)cudaFree(device_scalar_output);
        if (device_mv_output != nullptr) (void)cudaFree(device_mv_output);
        if (device_rotor != nullptr) (void)cudaFree(device_rotor);
        if (device_b != nullptr) (void)cudaFree(device_b);
        if (device_a != nullptr) (void)cudaFree(device_a);
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
        geo_cuda_context_destroy(api_context);
    }
};

struct Events {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~Events() {
        if (stop != nullptr) (void)cudaEventDestroy(stop);
        if (start != nullptr) (void)cudaEventDestroy(start);
    }
};

__device__ __forceinline__ geo_cl20_t device_add(
    geo_cl20_t left,
    geo_cl20_t right
) {
    geo_cl20_t output;
    output.scalar = left.scalar + right.scalar;
    output.e1 = left.e1 + right.e1;
    output.e2 = left.e2 + right.e2;
    output.e12 = left.e12 + right.e12;
    return output;
}

__device__ __forceinline__ geo_cl20_t device_product(
    geo_cl20_t left,
    geo_cl20_t right
) {
    geo_cl20_t output;
    output.scalar =
        left.scalar * right.scalar + left.e1 * right.e1 +
        left.e2 * right.e2 - left.e12 * right.e12;
    output.e1 =
        left.scalar * right.e1 + left.e1 * right.scalar -
        left.e2 * right.e12 + left.e12 * right.e2;
    output.e2 =
        left.scalar * right.e2 + left.e2 * right.scalar +
        left.e1 * right.e12 - left.e12 * right.e1;
    output.e12 =
        left.scalar * right.e12 + left.e12 * right.scalar +
        left.e1 * right.e2 - left.e2 * right.e1;
    return output;
}

__device__ __forceinline__ geo_cl20_t device_reverse(geo_cl20_t value) {
    value.e12 = -value.e12;
    return value;
}

__global__ void addition_kernel(
    const geo_cl20_t *left,
    const geo_cl20_t *right,
    geo_cl20_t *output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_add(left[index], right[index]);
}

__global__ void product_kernel(
    const geo_cl20_t *left,
    const geo_cl20_t *right,
    geo_cl20_t *output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_product(left[index], right[index]);
}

__global__ void reverse_kernel(
    const geo_cl20_t *input,
    geo_cl20_t *output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_reverse(input[index]);
}

__global__ void dot_kernel(
    const geo_cl20_t *left,
    const geo_cl20_t *right,
    geo_real_t *output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] =
            left[index].e1 * right[index].e1 +
            left[index].e2 * right[index].e2;
    }
}

__global__ void wedge_kernel(
    const geo_cl20_t *left,
    const geo_cl20_t *right,
    geo_real_t *output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] =
            left[index].e1 * right[index].e2 -
            left[index].e2 * right[index].e1;
    }
}

__global__ void rotor_kernel(
    const geo_cl20_t *rotor,
    const geo_cl20_t *value,
    geo_cl20_t *output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = device_product(
            device_product(rotor[index], value[index]),
            device_reverse(rotor[index])
        );
    }
}

const char *operation_name(Operation operation) {
    switch (operation) {
        case Operation::Addition:
            return "addition";
        case Operation::GeometricProduct:
            return "geometric_product";
        case Operation::Reverse:
            return "reverse";
        case Operation::VectorDot:
            return "vector_dot";
        case Operation::VectorWedge:
            return "vector_wedge";
        case Operation::RotorAction:
            return "rotor_action";
    }
    return "unknown";
}

bool operation_is_scalar(Operation operation) {
    return operation == Operation::VectorDot ||
        operation == Operation::VectorWedge;
}

bool parse_unsigned(const char *text, unsigned long long *value) {
    char *end = nullptr;
    unsigned long long parsed;

    if (text == nullptr || value == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    *value = parsed;
    return true;
}

void print_usage(const char *program) {
    std::printf(
        "Usage: %s [--device N] [--batch N] [--iterations N] [--warmup N] "
        "[--operation all|addition|geometric_product|reverse|vector_dot|"
        "vector_wedge|rotor_action] [--csv PATH]\n",
        program
    );
}

bool parse_options(int argc, char **argv, Options *options) {
    if (options == nullptr) return false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc) return false;

        const char *value = argv[++index];
        unsigned long long parsed = 0u;
        if (argument == "--device") {
            if (!parse_unsigned(value, &parsed) ||
                parsed > static_cast<unsigned long long>(INT_MAX)) {
                return false;
            }
            options->device = static_cast<int>(parsed);
        } else if (argument == "--batch") {
            if (!parse_unsigned(value, &parsed) || parsed == 0u ||
                parsed > static_cast<unsigned long long>(
                    std::numeric_limits<std::size_t>::max())) {
                return false;
            }
            options->batch = static_cast<std::size_t>(parsed);
        } else if (argument == "--iterations") {
            if (!parse_unsigned(value, &parsed) || parsed == 0u ||
                parsed > UINT_MAX) {
                return false;
            }
            options->iterations = static_cast<unsigned int>(parsed);
        } else if (argument == "--warmup") {
            if (!parse_unsigned(value, &parsed) || parsed > UINT_MAX) {
                return false;
            }
            options->warmup = static_cast<unsigned int>(parsed);
        } else if (argument == "--operation") {
            options->operation = value;
        } else if (argument == "--csv") {
            options->csv_path = value;
        } else {
            return false;
        }
    }
    return true;
}

bool selected(const Options &options, Operation operation) {
    if (options.operation == "all") return true;
    if (options.operation == operation_name(operation)) return true;
    if (options.operation == "add" && operation == Operation::Addition) return true;
    if (options.operation == "product" &&
        operation == Operation::GeometricProduct) return true;
    if (options.operation == "dot" && operation == Operation::VectorDot) return true;
    if (options.operation == "wedge" &&
        operation == Operation::VectorWedge) return true;
    if (options.operation == "rotor" &&
        operation == Operation::RotorAction) return true;
    return false;
}

bool checked_multiply(
    std::size_t left,
    std::size_t right,
    std::size_t *output
) {
    if (output == nullptr ||
        (right != 0u && left > std::numeric_limits<std::size_t>::max() / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

bool prepare_layout(
    std::size_t batch,
    int max_grid_x,
    BatchLayout *layout
) {
    if (layout == nullptr || batch == 0u || max_grid_x <= 0) return false;
    if (!checked_multiply(batch, sizeof(geo_cl20_t), &layout->mv_bytes) ||
        !checked_multiply(batch, sizeof(geo_real_t), &layout->scalar_bytes)) {
        return false;
    }

    const std::size_t threads = static_cast<std::size_t>(kThreadsPerBlock);
    const std::size_t block_count = batch / threads +
        (batch % threads == 0u ? 0u : 1u);
    if (block_count == 0u ||
        block_count > static_cast<std::size_t>(max_grid_x) ||
        block_count > static_cast<std::size_t>(UINT_MAX)) {
        return false;
    }
    layout->blocks = static_cast<unsigned int>(block_count);
    return true;
}

uint32_t next_random(uint32_t &state) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    return state;
}

geo_real_t sample(uint32_t &state) {
    const uint32_t bits = next_random(state) >> 8;
    const double unit = static_cast<double>(bits) /
        static_cast<double>(UINT32_C(0x00ffffff));
    return static_cast<geo_real_t>((unit * 2.0) - 1.0);
}

geo_cl20_t random_mv(uint32_t &state) {
    return geo_cl20_make(
        sample(state),
        sample(state),
        sample(state),
        sample(state)
    );
}

geo_cl20_t rotor_for(std::size_t index) {
    const double angle =
        0.0005 * static_cast<double>((index % 1009u) + 1u);
    return geo_cl20_make(
        static_cast<geo_real_t>(std::cos(angle)),
        static_cast<geo_real_t>(0),
        static_cast<geo_real_t>(0),
        static_cast<geo_real_t>(-std::sin(angle))
    );
}

bool cuda_ok(cudaError_t error, const char *stage) {
    if (error == cudaSuccess) return true;
    std::fprintf(
        stderr,
        "CUDA failure during %s: %s\n",
        stage,
        cudaGetErrorString(error)
    );
    return false;
}

bool allocate_device(void **pointer, std::size_t bytes, const char *stage) {
    if (pointer == nullptr || bytes == 0u) return false;
    return cuda_ok(cudaMalloc(pointer, bytes), stage);
}

bool create_events(Events *events) {
    if (events == nullptr) return false;
    if (!cuda_ok(cudaEventCreate(&events->start), "event creation")) return false;
    return cuda_ok(cudaEventCreate(&events->stop), "event creation");
}

bool record_elapsed(
    Events *events,
    cudaStream_t stream,
    float *milliseconds,
    const char *stage
) {
    return events != nullptr && milliseconds != nullptr &&
        cuda_ok(cudaEventRecord(events->stop, stream), stage) &&
        cuda_ok(cudaEventSynchronize(events->stop), stage) &&
        cuda_ok(
            cudaEventElapsedTime(milliseconds, events->start, events->stop),
            stage
        );
}

bool upload_inputs(
    Operation operation,
    const BatchLayout &layout,
    cudaStream_t stream,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    Resources *resources,
    Events *events,
    Timing *timing
) {
    if (resources == nullptr || events == nullptr || timing == nullptr) return false;
    if (!cuda_ok(cudaEventRecord(events->start, stream), "upload timing start")) {
        return false;
    }

    bool ok = true;
    switch (operation) {
        case Operation::Addition:
        case Operation::GeometricProduct:
        case Operation::VectorDot:
        case Operation::VectorWedge:
            timing->upload_bytes = layout.mv_bytes * 2u;
            ok = cuda_ok(
                cudaMemcpyAsync(
                    resources->device_a,
                    a.data(),
                    layout.mv_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload a"
            ) && cuda_ok(
                cudaMemcpyAsync(
                    resources->device_b,
                    b.data(),
                    layout.mv_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload b"
            );
            break;

        case Operation::Reverse:
            timing->upload_bytes = layout.mv_bytes;
            ok = cuda_ok(
                cudaMemcpyAsync(
                    resources->device_a,
                    a.data(),
                    layout.mv_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload reverse input"
            );
            break;

        case Operation::RotorAction:
            timing->upload_bytes = layout.mv_bytes * 2u;
            ok = cuda_ok(
                cudaMemcpyAsync(
                    resources->device_rotor,
                    rotor.data(),
                    layout.mv_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload rotor"
            ) && cuda_ok(
                cudaMemcpyAsync(
                    resources->device_a,
                    a.data(),
                    layout.mv_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ),
                "upload rotor value"
            );
            break;
    }

    return ok && record_elapsed(
        events,
        stream,
        &timing->upload_ms,
        "upload timing"
    );
}

bool launch(
    Operation operation,
    const Options &options,
    const BatchLayout &layout,
    const Resources &resources
) {
    switch (operation) {
        case Operation::Addition:
            addition_kernel<<<
                layout.blocks,
                kThreadsPerBlock,
                0u,
                resources.stream
            >>>(
                resources.device_a,
                resources.device_b,
                resources.device_mv_output,
                options.batch
            );
            break;

        case Operation::GeometricProduct:
            product_kernel<<<
                layout.blocks,
                kThreadsPerBlock,
                0u,
                resources.stream
            >>>(
                resources.device_a,
                resources.device_b,
                resources.device_mv_output,
                options.batch
            );
            break;

        case Operation::Reverse:
            reverse_kernel<<<
                layout.blocks,
                kThreadsPerBlock,
                0u,
                resources.stream
            >>>(
                resources.device_a,
                resources.device_mv_output,
                options.batch
            );
            break;

        case Operation::VectorDot:
            dot_kernel<<<
                layout.blocks,
                kThreadsPerBlock,
                0u,
                resources.stream
            >>>(
                resources.device_a,
                resources.device_b,
                resources.device_scalar_output,
                options.batch
            );
            break;

        case Operation::VectorWedge:
            wedge_kernel<<<
                layout.blocks,
                kThreadsPerBlock,
                0u,
                resources.stream
            >>>(
                resources.device_a,
                resources.device_b,
                resources.device_scalar_output,
                options.batch
            );
            break;

        case Operation::RotorAction:
            rotor_kernel<<<
                layout.blocks,
                kThreadsPerBlock,
                0u,
                resources.stream
            >>>(
                resources.device_rotor,
                resources.device_a,
                resources.device_mv_output,
                options.batch
            );
            break;
    }
    return cuda_ok(cudaGetLastError(), "kernel launch");
}

void update_error(ErrorStats *stats, geo_real_t actual, geo_real_t expected) {
    if (stats == nullptr) return;
    const double actual_double = static_cast<double>(actual);
    const double expected_double = static_cast<double>(expected);
    const double absolute = std::fabs(actual_double - expected_double);
    const double denominator = std::max(std::fabs(expected_double), 1.0e-30);
    const double relative = absolute / denominator;
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const double tolerance = 2.0e-12 + 2.0e-12 * std::fabs(expected_double);
#else
    const double tolerance = 4.0e-5 + 4.0e-5 * std::fabs(expected_double);
#endif
    stats->max_absolute = std::max(stats->max_absolute, absolute);
    stats->max_relative = std::max(stats->max_relative, relative);
    if (!std::isfinite(actual_double) || !std::isfinite(expected_double) ||
        absolute > tolerance) {
        ++stats->mismatches;
    }
}

ErrorStats compare_multivectors(
    Operation operation,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    const std::vector<geo_cl20_t> &actual
) {
    ErrorStats stats;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        geo_cl20_t expected;
        switch (operation) {
            case Operation::Addition:
                expected = geo_cl20_add(a[index], b[index]);
                break;
            case Operation::GeometricProduct:
                expected = geo_cl20_mul(a[index], b[index]);
                break;
            case Operation::Reverse:
                expected = geo_cl20_reverse(a[index]);
                break;
            case Operation::RotorAction:
                expected = geo_cl20_mul(
                    geo_cl20_mul(rotor[index], a[index]),
                    geo_cl20_reverse(rotor[index])
                );
                break;
            default:
                expected = geo_cl20_zero();
                break;
        }
        update_error(&stats, actual[index].scalar, expected.scalar);
        update_error(&stats, actual[index].e1, expected.e1);
        update_error(&stats, actual[index].e2, expected.e2);
        update_error(&stats, actual[index].e12, expected.e12);
    }
    return stats;
}

ErrorStats compare_scalars(
    Operation operation,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_real_t> &actual
) {
    ErrorStats stats;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const geo_real_t expected = operation == Operation::VectorDot
            ? geo_cl20_vector_dot(a[index], b[index])
            : geo_cl20_vector_wedge(a[index], b[index]);
        update_error(&stats, actual[index], expected);
    }
    return stats;
}

std::size_t logical_kernel_bytes(
    Operation operation,
    const BatchLayout &layout
) {
    switch (operation) {
        case Operation::Addition:
        case Operation::GeometricProduct:
        case Operation::RotorAction:
            return layout.mv_bytes * 3u;
        case Operation::Reverse:
            return layout.mv_bytes * 2u;
        case Operation::VectorDot:
        case Operation::VectorWedge:
            return layout.mv_bytes * 2u + layout.scalar_bytes;
    }
    return 0u;
}

bool run_operation(
    Operation operation,
    const Options &options,
    const BatchLayout &layout,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    std::vector<geo_cl20_t> *mv_output,
    std::vector<geo_real_t> *scalar_output,
    Resources *resources,
    Timing *timing,
    ErrorStats *errors
) {
    if (mv_output == nullptr || scalar_output == nullptr || resources == nullptr ||
        timing == nullptr || errors == nullptr) {
        return false;
    }

    Events events;
    if (!create_events(&events)) return false;
    timing->logical_kernel_bytes = logical_kernel_bytes(operation, layout);

    if (!upload_inputs(
            operation,
            layout,
            resources->stream,
            a,
            b,
            rotor,
            resources,
            &events,
            timing)) {
        return false;
    }

    for (unsigned int iteration = 0u;
         iteration < options.warmup;
         ++iteration) {
        if (!launch(operation, options, layout, *resources)) return false;
    }
    if (!cuda_ok(
            cudaStreamSynchronize(resources->stream),
            "warmup synchronization")) {
        return false;
    }

    if (!cuda_ok(
            cudaEventRecord(events.start, resources->stream),
            "kernel timing start")) {
        return false;
    }
    for (unsigned int iteration = 0u;
         iteration < options.iterations;
         ++iteration) {
        if (!launch(operation, options, layout, *resources)) return false;
    }
    if (!record_elapsed(
            &events,
            resources->stream,
            &timing->kernel_total_ms,
            "kernel timing")) {
        return false;
    }

    if (!cuda_ok(
            cudaEventRecord(events.start, resources->stream),
            "download timing start")) {
        return false;
    }
    if (operation_is_scalar(operation)) {
        timing->download_bytes = layout.scalar_bytes;
        if (!cuda_ok(
                cudaMemcpyAsync(
                    scalar_output->data(),
                    resources->device_scalar_output,
                    layout.scalar_bytes,
                    cudaMemcpyDeviceToHost,
                    resources->stream
                ),
                "download scalar output")) {
            return false;
        }
    } else {
        timing->download_bytes = layout.mv_bytes;
        if (!cuda_ok(
                cudaMemcpyAsync(
                    mv_output->data(),
                    resources->device_mv_output,
                    layout.mv_bytes,
                    cudaMemcpyDeviceToHost,
                    resources->stream
                ),
                "download multivector output")) {
            return false;
        }
    }
    if (!record_elapsed(
            &events,
            resources->stream,
            &timing->download_ms,
            "download timing")) {
        return false;
    }

    *errors = operation_is_scalar(operation)
        ? compare_scalars(operation, a, b, *scalar_output)
        : compare_multivectors(operation, a, b, rotor, *mv_output);
    return true;
}

void write_csv_header(std::ofstream *csv) {
    if (csv == nullptr || !csv->is_open()) return;
    *csv <<
        "operation,precision,batch,iterations,warmup,upload_bytes,download_bytes,"
        "logical_kernel_bytes,upload_ms,kernel_ms_per_batch,download_ms,"
        "total_ms_per_batch,items_per_second,kernel_effective_gbps,transfer_gbps,"
        "max_absolute_error,max_relative_error,mismatches\n";
}

void report(
    Operation operation,
    const Options &options,
    const Timing &timing,
    const ErrorStats &errors,
    std::ofstream *csv
) {
    const double kernel_ms = static_cast<double>(timing.kernel_total_ms) /
        static_cast<double>(options.iterations);
    const double total_ms = static_cast<double>(timing.upload_ms) + kernel_ms +
        static_cast<double>(timing.download_ms);
    const double kernel_seconds = kernel_ms * 1.0e-3;
    const double transfer_seconds =
        (static_cast<double>(timing.upload_ms) +
         static_cast<double>(timing.download_ms)) * 1.0e-3;
    const double items_per_second = kernel_seconds > 0.0
        ? static_cast<double>(options.batch) / kernel_seconds
        : 0.0;
    const double kernel_gbps = kernel_seconds > 0.0
        ? static_cast<double>(timing.logical_kernel_bytes) /
            kernel_seconds / 1.0e9
        : 0.0;
    const double transfer_gbps = transfer_seconds > 0.0
        ? static_cast<double>(timing.upload_bytes + timing.download_bytes) /
            transfer_seconds / 1.0e9
        : 0.0;
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif

    std::printf(
        "%-20s kernel=%9.4f ms total=%9.4f ms throughput=%12.3f Mitems/s "
        "kernel_bw=%8.3f GB/s transfer_bw=%8.3f GB/s max_abs=%10.3e "
        "max_rel=%10.3e mismatches=%zu\n",
        operation_name(operation),
        kernel_ms,
        total_ms,
        items_per_second / 1.0e6,
        kernel_gbps,
        transfer_gbps,
        errors.max_absolute,
        errors.max_relative,
        errors.mismatches
    );

    if (csv != nullptr && csv->is_open()) {
        *csv << operation_name(operation) << ',' << precision << ','
             << options.batch << ',' << options.iterations << ','
             << options.warmup << ',' << timing.upload_bytes << ','
             << timing.download_bytes << ',' << timing.logical_kernel_bytes << ','
             << std::setprecision(9) << timing.upload_ms << ',' << kernel_ms << ','
             << timing.download_ms << ',' << total_ms << ',' << items_per_second
             << ',' << kernel_gbps << ',' << transfer_gbps << ','
             << errors.max_absolute << ',' << errors.max_relative << ','
             << errors.mismatches << '\n';
    }
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    Resources resources;
    resources.device = options.device;
    const geo_cuda_status_t context_status =
        geo_cuda_context_create(options.device, &resources.api_context);
    if (context_status != GEO_CUDA_SUCCESS) {
        std::fprintf(
            stderr,
            "CUDA context creation failed: %s\n",
            geo_cuda_status_string(context_status)
        );
        return context_status == GEO_CUDA_NO_DEVICE ? 77 : EXIT_FAILURE;
    }

    geo_cuda_device_info_t info;
    const geo_cuda_status_t info_status =
        geo_cuda_get_device_info(resources.api_context, &info);
    if (info_status != GEO_CUDA_SUCCESS) {
        std::fprintf(
            stderr,
            "CUDA device query failed: %s\n",
            geo_cuda_status_string(info_status)
        );
        return EXIT_FAILURE;
    }

    if (!cuda_ok(cudaSetDevice(options.device), "device selection")) {
        return EXIT_FAILURE;
    }
    int max_grid_x = 0;
    if (!cuda_ok(
            cudaDeviceGetAttribute(
                &max_grid_x,
                cudaDevAttrMaxGridDimX,
                options.device
            ),
            "maximum grid query")) {
        return EXIT_FAILURE;
    }

    BatchLayout layout;
    if (!prepare_layout(options.batch, max_grid_x, &layout)) {
        std::fprintf(
            stderr,
            "Invalid batch size %zu for element sizes and device grid limit %d\n",
            options.batch,
            max_grid_x
        );
        return EXIT_FAILURE;
    }

    if (!cuda_ok(
            cudaStreamCreateWithFlags(
                &resources.stream,
                cudaStreamNonBlocking
            ),
            "stream creation")) {
        return EXIT_FAILURE;
    }

    if (!allocate_device(
            reinterpret_cast<void **>(&resources.device_a),
            layout.mv_bytes,
            "allocate a") ||
        !allocate_device(
            reinterpret_cast<void **>(&resources.device_b),
            layout.mv_bytes,
            "allocate b") ||
        !allocate_device(
            reinterpret_cast<void **>(&resources.device_rotor),
            layout.mv_bytes,
            "allocate rotor") ||
        !allocate_device(
            reinterpret_cast<void **>(&resources.device_mv_output),
            layout.mv_bytes,
            "allocate multivector output") ||
        !allocate_device(
            reinterpret_cast<void **>(&resources.device_scalar_output),
            layout.scalar_bytes,
            "allocate scalar output")) {
        return EXIT_FAILURE;
    }

    std::vector<geo_cl20_t> a;
    std::vector<geo_cl20_t> b;
    std::vector<geo_cl20_t> rotor;
    std::vector<geo_cl20_t> mv_output;
    std::vector<geo_real_t> scalar_output;
    try {
        a.resize(options.batch);
        b.resize(options.batch);
        rotor.resize(options.batch);
        mv_output.resize(options.batch);
        scalar_output.resize(options.batch);
    } catch (const std::bad_alloc &) {
        std::fprintf(
            stderr,
            "Host allocation failed for batch size %zu\n",
            options.batch
        );
        return EXIT_FAILURE;
    }

    uint32_t random_state = UINT32_C(0x243f6a88);
    for (std::size_t index = 0u; index < options.batch; ++index) {
        a[index] = random_mv(random_state);
        b[index] = random_mv(random_state);
        rotor[index] = rotor_for(index);
    }

    std::ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path, std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            std::fprintf(
                stderr,
                "Unable to open CSV output: %s\n",
                options.csv_path.c_str()
            );
            return EXIT_FAILURE;
        }
        write_csv_header(&csv);
    }

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif
    std::printf(
        "Geometric Elementary Operators CUDA 13.x harness\n"
        "device: %s (compute %d.%d, %d SMs, runtime %d, driver %d)\n"
        "precision: %s, batch: %zu, blocks: %u, threads: %u, "
        "iterations: %u, warmup: %u\n",
        info.name,
        info.compute_major,
        info.compute_minor,
        info.multiprocessor_count,
        info.runtime_version,
        info.driver_version,
        precision,
        options.batch,
        layout.blocks,
        kThreadsPerBlock,
        options.iterations,
        options.warmup
    );

    const Operation operations[] = {
        Operation::Addition,
        Operation::GeometricProduct,
        Operation::Reverse,
        Operation::VectorDot,
        Operation::VectorWedge,
        Operation::RotorAction
    };

    std::size_t selected_count = 0u;
    bool ok = true;
    for (Operation operation : operations) {
        if (!selected(options, operation)) continue;
        ++selected_count;
        Timing timing;
        ErrorStats errors;
        if (!run_operation(
                operation,
                options,
                layout,
                a,
                b,
                rotor,
                &mv_output,
                &scalar_output,
                &resources,
                &timing,
                &errors)) {
            ok = false;
            break;
        }
        report(
            operation,
            options,
            timing,
            errors,
            csv.is_open() ? &csv : nullptr
        );
        if (errors.mismatches != 0u) ok = false;
    }

    if (selected_count == 0u) {
        std::fprintf(
            stderr,
            "Unknown operation selection: %s\n",
            options.operation.c_str()
        );
        ok = false;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
