#include "geo/cl20.h"
#include "geo/cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

enum class Operation {
    Add,
    Product,
    Reverse,
    Dot,
    Wedge,
    Rotor
};

struct Options {
    int device = 0;
    std::size_t batch = 262144u;
    unsigned int iterations = 100u;
    unsigned int warmup = 10u;
    std::string operation = "all";
    std::string csv_path;
};

struct Timing {
    float upload_ms = 0.0f;
    float kernel_total_ms = 0.0f;
    float download_ms = 0.0f;
};

struct ErrorStats {
    double max_absolute = 0.0;
    double max_relative = 0.0;
    std::size_t mismatches = 0u;
};

__device__ __forceinline__ geo_cl20_t device_add(geo_cl20_t a, geo_cl20_t b) {
    geo_cl20_t output;
    output.scalar = a.scalar + b.scalar;
    output.e1 = a.e1 + b.e1;
    output.e2 = a.e2 + b.e2;
    output.e12 = a.e12 + b.e12;
    return output;
}

__device__ __forceinline__ geo_cl20_t device_product(geo_cl20_t a, geo_cl20_t b) {
    geo_cl20_t output;
    output.scalar = a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 - a.e12 * b.e12;
    output.e1 = a.scalar * b.e1 + a.e1 * b.scalar - a.e2 * b.e12 + a.e12 * b.e2;
    output.e2 = a.scalar * b.e2 + a.e2 * b.scalar + a.e1 * b.e12 - a.e12 * b.e1;
    output.e12 = a.scalar * b.e12 + a.e12 * b.scalar + a.e1 * b.e2 - a.e2 * b.e1;
    return output;
}

__device__ __forceinline__ geo_cl20_t device_reverse(geo_cl20_t value) {
    value.e12 = -value.e12;
    return value;
}

__global__ void add_kernel(const geo_cl20_t *a, const geo_cl20_t *b, geo_cl20_t *output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_add(a[index], b[index]);
}

__global__ void product_kernel(const geo_cl20_t *a, const geo_cl20_t *b, geo_cl20_t *output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_product(a[index], b[index]);
}

__global__ void reverse_kernel(const geo_cl20_t *input, geo_cl20_t *output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = device_reverse(input[index]);
}

__global__ void dot_kernel(const geo_cl20_t *a, const geo_cl20_t *b, geo_real_t *output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = a[index].e1 * b[index].e1 + a[index].e2 * b[index].e2;
}

__global__ void wedge_kernel(const geo_cl20_t *a, const geo_cl20_t *b, geo_real_t *output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = a[index].e1 * b[index].e2 - a[index].e2 * b[index].e1;
}

__global__ void rotor_kernel(
    const geo_cl20_t *rotor,
    const geo_cl20_t *value,
    geo_cl20_t *output,
    std::size_t count
) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = device_product(
            device_product(rotor[index], value[index]),
            device_reverse(rotor[index])
        );
    }
}

const char *operation_name(Operation operation) {
    switch (operation) {
        case Operation::Add: return "add";
        case Operation::Product: return "product";
        case Operation::Reverse: return "reverse";
        case Operation::Dot: return "vector_dot";
        case Operation::Wedge: return "vector_wedge";
        case Operation::Rotor: return "rotor_action";
    }
    return "unknown";
}

bool operation_is_scalar(Operation operation) {
    return operation == Operation::Dot || operation == Operation::Wedge;
}

bool parse_unsigned(const char *text, unsigned long long *value) {
    char *end = nullptr;
    if (text == nullptr || value == nullptr || text[0] == '\0') return false;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *value = parsed;
    return true;
}

void print_usage(const char *program) {
    std::printf(
        "Usage: %s [--device N] [--batch N] [--iterations N] [--warmup N] "
        "[--operation all|add|product|reverse|dot|wedge|rotor] [--csv PATH]\n",
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
            if (!parse_unsigned(value, &parsed) || parsed > static_cast<unsigned long long>(INT_MAX)) return false;
            options->device = static_cast<int>(parsed);
        } else if (argument == "--batch") {
            if (!parse_unsigned(value, &parsed) || parsed == 0u ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) return false;
            options->batch = static_cast<std::size_t>(parsed);
        } else if (argument == "--iterations") {
            if (!parse_unsigned(value, &parsed) || parsed == 0u || parsed > UINT_MAX) return false;
            options->iterations = static_cast<unsigned int>(parsed);
        } else if (argument == "--warmup") {
            if (!parse_unsigned(value, &parsed) || parsed > UINT_MAX) return false;
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
    const std::string name(operation_name(operation));
    if (options.operation == name) return true;
    if (options.operation == "dot" && operation == Operation::Dot) return true;
    if (options.operation == "wedge" && operation == Operation::Wedge) return true;
    if (options.operation == "rotor" && operation == Operation::Rotor) return true;
    return false;
}

uint32_t next_random(uint32_t &state) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    return state;
}

geo_real_t sample(uint32_t &state) {
    const uint32_t bits = next_random(state) >> 8;
    const double unit = static_cast<double>(bits) / static_cast<double>(UINT32_C(0x00ffffff));
    return static_cast<geo_real_t>((unit * 2.0) - 1.0);
}

geo_cl20_t random_mv(uint32_t &state) {
    return geo_cl20_make(sample(state), sample(state), sample(state), sample(state));
}

geo_cl20_t rotor_for(std::size_t index) {
    const double angle = 0.0005 * static_cast<double>((index % 1009u) + 1u);
    return geo_cl20_make(
        static_cast<geo_real_t>(std::cos(angle)),
        static_cast<geo_real_t>(0),
        static_cast<geo_real_t>(0),
        static_cast<geo_real_t>(-std::sin(angle))
    );
}

bool cuda_ok(cudaError_t error, const char *what) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA failure during %s: %s\n", what, cudaGetErrorString(error));
    return false;
}

unsigned int block_count(std::size_t count) {
    return static_cast<unsigned int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                     static_cast<std::size_t>(kThreadsPerBlock));
}

bool launch(
    Operation operation,
    const geo_cl20_t *device_a,
    const geo_cl20_t *device_b,
    const geo_cl20_t *device_rotor,
    geo_cl20_t *device_mv_output,
    geo_real_t *device_scalar_output,
    std::size_t count,
    cudaStream_t stream
) {
    const unsigned int blocks = block_count(count);
    switch (operation) {
        case Operation::Add:
            add_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_a, device_b, device_mv_output, count);
            break;
        case Operation::Product:
            product_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_a, device_b, device_mv_output, count);
            break;
        case Operation::Reverse:
            reverse_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_a, device_mv_output, count);
            break;
        case Operation::Dot:
            dot_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_a, device_b, device_scalar_output, count);
            break;
        case Operation::Wedge:
            wedge_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_a, device_b, device_scalar_output, count);
            break;
        case Operation::Rotor:
            rotor_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_rotor, device_a, device_mv_output, count);
            break;
    }
    return cuda_ok(cudaGetLastError(), "kernel launch");
}

void update_error(ErrorStats *stats, geo_real_t actual, geo_real_t expected) {
    const double a = static_cast<double>(actual);
    const double e = static_cast<double>(expected);
    const double absolute = std::fabs(a - e);
    const double denominator = std::max(std::fabs(e), 1.0e-30);
    const double relative = absolute / denominator;
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const double tolerance = 2.0e-12 + 2.0e-12 * std::fabs(e);
#else
    const double tolerance = 4.0e-5 + 4.0e-5 * std::fabs(e);
#endif
    stats->max_absolute = std::max(stats->max_absolute, absolute);
    stats->max_relative = std::max(stats->max_relative, relative);
    if (!std::isfinite(a) || !std::isfinite(e) || absolute > tolerance) ++stats->mismatches;
}

ErrorStats compare_mv(
    Operation operation,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    const std::vector<geo_cl20_t> &actual
) {
    ErrorStats stats;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        geo_cl20_t expected;
        switch (operation) {
            case Operation::Add:
                expected = geo_cl20_add(a[index], b[index]);
                break;
            case Operation::Product:
                expected = geo_cl20_mul(a[index], b[index]);
                break;
            case Operation::Reverse:
                expected = geo_cl20_reverse(a[index]);
                break;
            case Operation::Rotor:
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

ErrorStats compare_scalar(
    Operation operation,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_real_t> &actual
) {
    ErrorStats stats;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const geo_real_t expected = operation == Operation::Dot
            ? geo_cl20_vector_dot(a[index], b[index])
            : geo_cl20_vector_wedge(a[index], b[index]);
        update_error(&stats, actual[index], expected);
    }
    return stats;
}

bool run_operation(
    Operation operation,
    const Options &options,
    cudaStream_t stream,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    geo_cl20_t *device_a,
    geo_cl20_t *device_b,
    geo_cl20_t *device_rotor,
    geo_cl20_t *device_mv_output,
    geo_real_t *device_scalar_output,
    Timing *timing,
    ErrorStats *error_stats
) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "event creation") ||
        !cuda_ok(cudaEventCreate(&stop), "event creation")) {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        return false;
    }

    const std::size_t mv_bytes = options.batch * sizeof(geo_cl20_t);
    const std::size_t scalar_bytes = options.batch * sizeof(geo_real_t);
    bool ok = cuda_ok(cudaEventRecord(start, stream), "upload timing start") &&
        cuda_ok(cudaMemcpyAsync(device_a, a.data(), mv_bytes, cudaMemcpyHostToDevice, stream), "upload a") &&
        cuda_ok(cudaMemcpyAsync(device_b, b.data(), mv_bytes, cudaMemcpyHostToDevice, stream), "upload b") &&
        cuda_ok(cudaMemcpyAsync(device_rotor, rotor.data(), mv_bytes, cudaMemcpyHostToDevice, stream), "upload rotor") &&
        cuda_ok(cudaEventRecord(stop, stream), "upload timing stop") &&
        cuda_ok(cudaEventSynchronize(stop), "upload synchronization") &&
        cuda_ok(cudaEventElapsedTime(&timing->upload_ms, start, stop), "upload timing");

    for (unsigned int iteration = 0u; ok && iteration < options.warmup; ++iteration) {
        ok = launch(operation, device_a, device_b, device_rotor, device_mv_output,
                    device_scalar_output, options.batch, stream);
    }
    if (ok) ok = cuda_ok(cudaStreamSynchronize(stream), "warmup synchronization");

    if (ok) ok = cuda_ok(cudaEventRecord(start, stream), "kernel timing start");
    for (unsigned int iteration = 0u; ok && iteration < options.iterations; ++iteration) {
        ok = launch(operation, device_a, device_b, device_rotor, device_mv_output,
                    device_scalar_output, options.batch, stream);
    }
    if (ok) ok = cuda_ok(cudaEventRecord(stop, stream), "kernel timing stop");
    if (ok) ok = cuda_ok(cudaEventSynchronize(stop), "kernel synchronization");
    if (ok) ok = cuda_ok(cudaEventElapsedTime(&timing->kernel_total_ms, start, stop), "kernel timing");

    if (ok) ok = cuda_ok(cudaEventRecord(start, stream), "download timing start");
    if (operation_is_scalar(operation)) {
        std::vector<geo_real_t> output(options.batch);
        if (ok) ok = cuda_ok(cudaMemcpyAsync(output.data(), device_scalar_output, scalar_bytes,
                                              cudaMemcpyDeviceToHost, stream), "download scalar output");
        if (ok) ok = cuda_ok(cudaEventRecord(stop, stream), "download timing stop");
        if (ok) ok = cuda_ok(cudaEventSynchronize(stop), "download synchronization");
        if (ok) ok = cuda_ok(cudaEventElapsedTime(&timing->download_ms, start, stop), "download timing");
        if (ok) *error_stats = compare_scalar(operation, a, b, output);
    } else {
        std::vector<geo_cl20_t> output(options.batch);
        if (ok) ok = cuda_ok(cudaMemcpyAsync(output.data(), device_mv_output, mv_bytes,
                                              cudaMemcpyDeviceToHost, stream), "download multivector output");
        if (ok) ok = cuda_ok(cudaEventRecord(stop, stream), "download timing stop");
        if (ok) ok = cuda_ok(cudaEventSynchronize(stop), "download synchronization");
        if (ok) ok = cuda_ok(cudaEventElapsedTime(&timing->download_ms, start, stop), "download timing");
        if (ok) *error_stats = compare_mv(operation, a, b, rotor, output);
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ok;
}

void write_csv_header(std::ofstream *csv) {
    if (csv == nullptr || !csv->is_open()) return;
    *csv << "operation,precision,batch,iterations,warmup,upload_ms,kernel_ms_per_batch,"
            "download_ms,total_ms_per_batch,items_per_second,max_absolute_error,"
            "max_relative_error,mismatches\n";
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
    const double items_per_second = static_cast<double>(options.batch) / (kernel_ms * 1.0e-3);
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif

    std::printf(
        "%-16s kernel=%9.4f ms  total=%9.4f ms  throughput=%12.3f Mitems/s  "
        "max_abs=%10.3e  max_rel=%10.3e  mismatches=%zu\n",
        operation_name(operation),
        kernel_ms,
        total_ms,
        items_per_second / 1.0e6,
        errors.max_absolute,
        errors.max_relative,
        errors.mismatches
    );

    if (csv != nullptr && csv->is_open()) {
        *csv << operation_name(operation) << ',' << precision << ',' << options.batch << ','
             << options.iterations << ',' << options.warmup << ','
             << std::setprecision(9) << timing.upload_ms << ',' << kernel_ms << ','
             << timing.download_ms << ',' << total_ms << ',' << items_per_second << ','
             << errors.max_absolute << ',' << errors.max_relative << ',' << errors.mismatches << '\n';
    }
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    geo_cuda_context_t *api_context = nullptr;
    const geo_cuda_status_t context_status = geo_cuda_context_create(options.device, &api_context);
    if (context_status != GEO_CUDA_SUCCESS) {
        std::fprintf(stderr, "CUDA context creation failed: %s\n", geo_cuda_status_string(context_status));
        return context_status == GEO_CUDA_NO_DEVICE ? 77 : EXIT_FAILURE;
    }

    geo_cuda_device_info_t info;
    const geo_cuda_status_t info_status = geo_cuda_get_device_info(api_context, &info);
    if (info_status != GEO_CUDA_SUCCESS) {
        std::fprintf(stderr, "CUDA device query failed: %s\n", geo_cuda_status_string(info_status));
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }

    if (!cuda_ok(cudaSetDevice(options.device), "device selection")) {
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }

    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation")) {
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }

    const std::size_t mv_bytes = options.batch * sizeof(geo_cl20_t);
    const std::size_t scalar_bytes = options.batch * sizeof(geo_real_t);
    geo_cl20_t *device_a = nullptr;
    geo_cl20_t *device_b = nullptr;
    geo_cl20_t *device_rotor = nullptr;
    geo_cl20_t *device_mv_output = nullptr;
    geo_real_t *device_scalar_output = nullptr;

    bool ok = cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_a), mv_bytes), "allocate a") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_b), mv_bytes), "allocate b") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_rotor), mv_bytes), "allocate rotor") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_mv_output), mv_bytes), "allocate multivector output") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_scalar_output), scalar_bytes), "allocate scalar output");

    std::vector<geo_cl20_t> a(options.batch);
    std::vector<geo_cl20_t> b(options.batch);
    std::vector<geo_cl20_t> rotor(options.batch);
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
            std::fprintf(stderr, "Unable to open CSV output: %s\n", options.csv_path.c_str());
            ok = false;
        } else {
            write_csv_header(&csv);
        }
    }

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif
    std::printf(
        "Geometric Elementary Operators CUDA 13.x harness\n"
        "device: %s (compute %d.%d, %d SMs, runtime %d, driver %d)\n"
        "precision: %s, batch: %zu, iterations: %u, warmup: %u\n",
        info.name,
        info.compute_major,
        info.compute_minor,
        info.multiprocessor_count,
        info.runtime_version,
        info.driver_version,
        precision,
        options.batch,
        options.iterations,
        options.warmup
    );

    const Operation operations[] = {
        Operation::Add,
        Operation::Product,
        Operation::Reverse,
        Operation::Dot,
        Operation::Wedge,
        Operation::Rotor
    };

    std::size_t selected_count = 0u;
    for (Operation operation : operations) {
        if (!selected(options, operation)) continue;
        ++selected_count;
        Timing timing;
        ErrorStats errors;
        if (!ok || !run_operation(
                operation,
                options,
                stream,
                a,
                b,
                rotor,
                device_a,
                device_b,
                device_rotor,
                device_mv_output,
                device_scalar_output,
                &timing,
                &errors
            )) {
            ok = false;
            break;
        }
        report(operation, options, timing, errors, csv.is_open() ? &csv : nullptr);
        if (errors.mismatches != 0u) ok = false;
    }
    if (selected_count == 0u) {
        std::fprintf(stderr, "Unknown operation selection: %s\n", options.operation.c_str());
        ok = false;
    }

    if (device_scalar_output != nullptr) cudaFree(device_scalar_output);
    if (device_mv_output != nullptr) cudaFree(device_mv_output);
    if (device_rotor != nullptr) cudaFree(device_rotor);
    if (device_b != nullptr) cudaFree(device_b);
    if (device_a != nullptr) cudaFree(device_a);
    cudaStreamDestroy(stream);
    geo_cuda_context_destroy(api_context);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
