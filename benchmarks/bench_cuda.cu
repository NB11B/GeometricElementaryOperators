#include "geo/cl20.h"
#include "geo/cuda.h"

#if defined(GEO_HAVE_GENERATED_CUDA_SCHEDULES)
#include "geo_cuda_schedule_addition.h"
#include "geo_cuda_schedule_geometric_product.h"
#include "geo_cuda_schedule_rotor_action.h"
#include "geo_cuda_schedule_vector_dot.h"
#include "geo_cuda_schedule_vector_wedge.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
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

constexpr Operation kOperations[] = {
    Operation::Addition,
    Operation::GeometricProduct,
    Operation::Reverse,
    Operation::VectorDot,
    Operation::VectorWedge,
    Operation::RotorAction
};

struct Options {
    int device = 0;
    std::size_t batch = 262144u;
    unsigned int iterations = 100u;
    unsigned int warmup = 10u;
    uint32_t seed = UINT32_C(0x243f6a88);
    std::string operation = "all";
    std::string csv_path;
};

struct BatchLayout {
    std::size_t mv_bytes = 0u;
    std::size_t scalar_bytes = 0u;
    unsigned int blocks = 0u;
};

struct ByteCounts {
    std::size_t upload_bytes = 0u;
    std::size_t download_bytes = 0u;
    std::size_t logical_kernel_bytes = 0u;
};

struct ErrorStats {
    double max_absolute = 0.0;
    double max_relative = 0.0;
    std::size_t mismatches = 0u;
};

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~EventPair() {
        if (stop != nullptr) (void)cudaEventDestroy(stop);
        if (start != nullptr) (void)cudaEventDestroy(start);
    }
};

struct GeneratedResources {
    int device = 0;
    cudaStream_t stream = nullptr;
    geo_cl20_t *device_a = nullptr;
    geo_cl20_t *device_b = nullptr;
    geo_cl20_t *device_reverse = nullptr;
    geo_cl20_t *device_output = nullptr;

    ~GeneratedResources() {
        (void)cudaSetDevice(device);
        if (device_output != nullptr) (void)cudaFree(device_output);
        if (device_reverse != nullptr) (void)cudaFree(device_reverse);
        if (device_b != nullptr) (void)cudaFree(device_b);
        if (device_a != nullptr) (void)cudaFree(device_a);
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
    }
};

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

bool operation_is_public_scalar(Operation operation) {
    return operation == Operation::VectorDot ||
        operation == Operation::VectorWedge;
}

#if defined(GEO_HAVE_GENERATED_CUDA_SCHEDULES)
bool operation_has_generated_schedule(Operation operation) {
    return operation != Operation::Reverse;
}
#endif

std::string canonical_operation(const std::string &value) {
    if (value == "all") return value;
    if (value == "addition" || value == "add") return "addition";
    if (value == "geometric_product" || value == "product") {
        return "geometric_product";
    }
    if (value == "reverse") return value;
    if (value == "vector_dot" || value == "dot") return "vector_dot";
    if (value == "vector_wedge" || value == "wedge") {
        return "vector_wedge";
    }
    if (value == "rotor_action" || value == "rotor") {
        return "rotor_action";
    }
    return std::string();
}

bool parse_unsigned(const char *text, unsigned long long *value) {
    char *end = nullptr;
    unsigned long long parsed;
    if (text == nullptr || value == nullptr || text[0] == '\0' ||
        text[0] == '-') {
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
        "[--seed N] [--operation all|addition|geometric_product|reverse|"
        "vector_dot|vector_wedge|rotor_action] [--csv PATH]\n",
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
                parsed > static_cast<unsigned long long>(
                    std::numeric_limits<int>::max())) {
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
                parsed > std::numeric_limits<unsigned int>::max()) {
                return false;
            }
            options->iterations = static_cast<unsigned int>(parsed);
        } else if (argument == "--warmup") {
            if (!parse_unsigned(value, &parsed) ||
                parsed > std::numeric_limits<unsigned int>::max()) {
                return false;
            }
            options->warmup = static_cast<unsigned int>(parsed);
        } else if (argument == "--seed") {
            if (!parse_unsigned(value, &parsed) || parsed > UINT32_MAX) {
                return false;
            }
            options->seed = static_cast<uint32_t>(parsed);
        } else if (argument == "--operation") {
            options->operation = canonical_operation(value);
            if (options->operation.empty()) return false;
        } else if (argument == "--csv") {
            options->csv_path = value;
        } else {
            return false;
        }
    }
    return true;
}

bool selected(const Options &options, Operation operation) {
    return options.operation == "all" ||
        options.operation == operation_name(operation);
}

bool checked_multiply(
    std::size_t left,
    std::size_t right,
    std::size_t *output
) {
    if (output == nullptr ||
        (right != 0u &&
         left > std::numeric_limits<std::size_t>::max() / right)) {
        return false;
    }
    *output = left * right;
    return true;
}

bool checked_add(
    std::size_t left,
    std::size_t right,
    std::size_t *output
) {
    if (output == nullptr ||
        left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    *output = left + right;
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
        block_count > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    layout->blocks = static_cast<unsigned int>(block_count);
    return true;
}

bool byte_counts(
    Operation operation,
    bool generated,
    const BatchLayout &layout,
    ByteCounts *counts
) {
    if (counts == nullptr) return false;
    std::size_t two_mv = 0u;
    std::size_t three_mv = 0u;
    std::size_t four_mv = 0u;
    if (!checked_multiply(layout.mv_bytes, 2u, &two_mv) ||
        !checked_multiply(layout.mv_bytes, 3u, &three_mv) ||
        !checked_multiply(layout.mv_bytes, 4u, &four_mv)) {
        return false;
    }
    if (generated) {
        counts->upload_bytes = operation == Operation::RotorAction
            ? three_mv : two_mv;
        counts->download_bytes = layout.mv_bytes;
        counts->logical_kernel_bytes = operation == Operation::RotorAction
            ? four_mv : three_mv;
        return operation != Operation::Reverse;
    }
    switch (operation) {
        case Operation::Addition:
        case Operation::GeometricProduct:
        case Operation::RotorAction:
            counts->upload_bytes = two_mv;
            counts->download_bytes = layout.mv_bytes;
            counts->logical_kernel_bytes = three_mv;
            return true;
        case Operation::Reverse:
            counts->upload_bytes = layout.mv_bytes;
            counts->download_bytes = layout.mv_bytes;
            counts->logical_kernel_bytes = two_mv;
            return true;
        case Operation::VectorDot:
        case Operation::VectorWedge:
            counts->upload_bytes = two_mv;
            counts->download_bytes = layout.scalar_bytes;
            return checked_add(
                two_mv, layout.scalar_bytes,
                &counts->logical_kernel_bytes
            );
    }
    return false;
}

uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

geo_real_t sample(uint32_t *state) {
    const uint32_t bits = next_random(state) >> 8;
    const double unit = static_cast<double>(bits) /
        static_cast<double>(UINT32_C(0x00ffffff));
    return static_cast<geo_real_t>((unit * 2.0) - 1.0);
}

geo_cl20_t random_mv(uint32_t *state) {
    return geo_cl20_make(
        sample(state), sample(state), sample(state), sample(state)
    );
}

geo_cl20_t random_rotor(uint32_t *state) {
    const double angle = static_cast<double>(sample(state));
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
        stderr, "CUDA failure during %s: %s\n", stage,
        cudaGetErrorString(error)
    );
    return false;
}

template <typename T>
bool allocate_device(T **pointer, std::size_t bytes, const char *stage) {
    if (pointer == nullptr || bytes == 0u) return false;
    return cuda_ok(
        cudaMalloc(reinterpret_cast<void **>(pointer), bytes), stage
    );
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

geo_cl20_t expected_multivector(
    Operation operation,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t rotor,
    bool generated
) {
    switch (operation) {
        case Operation::Addition:
            return geo_cl20_add(a, b);
        case Operation::GeometricProduct:
            return geo_cl20_mul(a, b);
        case Operation::Reverse:
            return geo_cl20_reverse(a);
        case Operation::VectorDot:
            return geo_cl20_make(
                geo_cl20_vector_dot(a, b), 0, 0, 0
            );
        case Operation::VectorWedge:
            return geo_cl20_make(
                0, 0, 0,
                generated ? geo_cl20_vector_wedge(a, b) : 0
            );
        case Operation::RotorAction:
            return geo_cl20_mul(
                geo_cl20_mul(rotor, a), geo_cl20_reverse(rotor)
            );
    }
    return geo_cl20_zero();
}

ErrorStats compare_multivectors(
    Operation operation,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    const std::vector<geo_cl20_t> &actual,
    bool generated
) {
    ErrorStats stats;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const geo_cl20_t expected = expected_multivector(
            operation, a[index], b[index], rotor[index], generated
        );
        update_error(&stats, actual[index].scalar, expected.scalar);
        update_error(&stats, actual[index].e1, expected.e1);
        update_error(&stats, actual[index].e2, expected.e2);
        update_error(&stats, actual[index].e12, expected.e12);
    }
    return stats;
}

ErrorStats compare_public_scalars(
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

geo_cuda_status_t run_public_once(
    Operation operation,
    geo_cuda_context_t *context,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    std::vector<geo_cl20_t> *mv_output,
    std::vector<geo_real_t> *scalar_output
) {
    switch (operation) {
        case Operation::Addition:
            return geo_cuda_cl20_add_batch(
                context, a.data(), b.data(), mv_output->data(), a.size()
            );
        case Operation::GeometricProduct:
            return geo_cuda_cl20_product_batch(
                context, a.data(), b.data(), mv_output->data(), a.size()
            );
        case Operation::Reverse:
            return geo_cuda_cl20_reverse_batch(
                context, a.data(), mv_output->data(), a.size()
            );
        case Operation::VectorDot:
            return geo_cuda_cl20_vector_dot_batch(
                context, a.data(), b.data(), scalar_output->data(), a.size()
            );
        case Operation::VectorWedge:
            return geo_cuda_cl20_vector_wedge_batch(
                context, a.data(), b.data(), scalar_output->data(), a.size()
            );
        case Operation::RotorAction:
            return geo_cuda_cl20_rotor_action_batch(
                context, rotor.data(), a.data(), mv_output->data(), a.size()
            );
    }
    return GEO_CUDA_INVALID_ARGUMENT;
}

bool run_public(
    Operation operation,
    const Options &options,
    geo_cuda_context_t *context,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    std::vector<geo_cl20_t> *mv_output,
    std::vector<geo_real_t> *scalar_output,
    double *ns_per_item,
    ErrorStats *errors
) {
    for (unsigned int index = 0u; index < options.warmup; ++index) {
        const geo_cuda_status_t status = run_public_once(
            operation, context, a, b, rotor, mv_output, scalar_output
        );
        if (status != GEO_CUDA_SUCCESS) {
            std::fprintf(
                stderr, "Public CUDA warmup failed for %s: %s\n",
                operation_name(operation), geo_cuda_status_string(status)
            );
            return false;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    for (unsigned int index = 0u; index < options.iterations; ++index) {
        const geo_cuda_status_t status = run_public_once(
            operation, context, a, b, rotor, mv_output, scalar_output
        );
        if (status != GEO_CUDA_SUCCESS) {
            std::fprintf(
                stderr, "Public CUDA execution failed for %s: %s\n",
                operation_name(operation), geo_cuda_status_string(status)
            );
            return false;
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ns = std::chrono::duration<double, std::nano>(
        stop - start
    ).count();
    *ns_per_item = elapsed_ns /
        (static_cast<double>(options.iterations) * options.batch);
    *errors = operation_is_public_scalar(operation)
        ? compare_public_scalars(operation, a, b, *scalar_output)
        : compare_multivectors(
            operation, a, b, rotor, *mv_output, false
        );
    return true;
}

#if defined(GEO_HAVE_GENERATED_CUDA_SCHEDULES)
bool upload_generated(
    Operation operation,
    const BatchLayout &layout,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    const std::vector<geo_cl20_t> &reverse,
    GeneratedResources *resources
) {
    const geo_cl20_t *input_a = operation == Operation::RotorAction
        ? rotor.data() : a.data();
    const geo_cl20_t *input_b = operation == Operation::RotorAction
        ? a.data() : b.data();
    if (!cuda_ok(cudaMemcpyAsync(
            resources->device_a, input_a, layout.mv_bytes,
            cudaMemcpyHostToDevice, resources->stream), "generated upload r0") ||
        !cuda_ok(cudaMemcpyAsync(
            resources->device_b, input_b, layout.mv_bytes,
            cudaMemcpyHostToDevice, resources->stream), "generated upload r1")) {
        return false;
    }
    if (operation == Operation::RotorAction &&
        !cuda_ok(cudaMemcpyAsync(
            resources->device_reverse, reverse.data(), layout.mv_bytes,
            cudaMemcpyHostToDevice, resources->stream),
            "generated upload r2")) {
        return false;
    }
    return cuda_ok(
        cudaStreamSynchronize(resources->stream),
        "generated upload synchronization"
    );
}

int launch_generated(
    Operation operation,
    const Options &options,
    const GeneratedResources &resources
) {
    void *stream = reinterpret_cast<void *>(resources.stream);
    switch (operation) {
        case Operation::Addition:
            return geo_cuda_schedule_addition_launch(
                resources.device_a, resources.device_b,
                resources.device_output, options.batch,
                kThreadsPerBlock, stream
            );
        case Operation::GeometricProduct:
            return geo_cuda_schedule_geometric_product_launch(
                resources.device_a, resources.device_b,
                resources.device_output, options.batch,
                kThreadsPerBlock, stream
            );
        case Operation::VectorDot:
            return geo_cuda_schedule_vector_dot_launch(
                resources.device_a, resources.device_b,
                resources.device_output, options.batch,
                kThreadsPerBlock, stream
            );
        case Operation::VectorWedge:
            return geo_cuda_schedule_vector_wedge_launch(
                resources.device_a, resources.device_b,
                resources.device_output, options.batch,
                kThreadsPerBlock, stream
            );
        case Operation::RotorAction:
            return geo_cuda_schedule_rotor_action_launch(
                resources.device_a, resources.device_b,
                resources.device_reverse, resources.device_output,
                options.batch, kThreadsPerBlock, stream
            );
        case Operation::Reverse:
            return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

bool run_generated(
    Operation operation,
    const Options &options,
    const BatchLayout &layout,
    const std::vector<geo_cl20_t> &a,
    const std::vector<geo_cl20_t> &b,
    const std::vector<geo_cl20_t> &rotor,
    const std::vector<geo_cl20_t> &reverse,
    std::vector<geo_cl20_t> *output,
    GeneratedResources *resources,
    double *ns_per_item,
    ErrorStats *errors
) {
    if (!upload_generated(
            operation, layout, a, b, rotor, reverse, resources)) {
        return false;
    }
    for (unsigned int index = 0u; index < options.warmup; ++index) {
        const int status = launch_generated(operation, options, *resources);
        if (status != static_cast<int>(cudaSuccess)) {
            std::fprintf(
                stderr, "Generated CUDA warmup failed for %s: %s\n",
                operation_name(operation),
                cudaGetErrorString(static_cast<cudaError_t>(status))
            );
            return false;
        }
    }
    if (!cuda_ok(cudaStreamSynchronize(resources->stream),
            "generated warmup synchronization")) {
        return false;
    }
    EventPair events;
    if (!cuda_ok(cudaEventCreate(&events.start), "generated event creation") ||
        !cuda_ok(cudaEventCreate(&events.stop), "generated event creation") ||
        !cuda_ok(cudaEventRecord(events.start, resources->stream),
            "generated timing start")) {
        return false;
    }
    for (unsigned int index = 0u; index < options.iterations; ++index) {
        const int status = launch_generated(operation, options, *resources);
        if (status != static_cast<int>(cudaSuccess)) {
            std::fprintf(
                stderr, "Generated CUDA execution failed for %s: %s\n",
                operation_name(operation),
                cudaGetErrorString(static_cast<cudaError_t>(status))
            );
            return false;
        }
    }
    float elapsed_ms = 0.0f;
    if (!cuda_ok(cudaEventRecord(events.stop, resources->stream),
            "generated timing stop") ||
        !cuda_ok(cudaEventSynchronize(events.stop),
            "generated timing synchronization") ||
        !cuda_ok(cudaEventElapsedTime(
            &elapsed_ms, events.start, events.stop),
            "generated elapsed time")) {
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(
            output->data(), resources->device_output, layout.mv_bytes,
            cudaMemcpyDeviceToHost, resources->stream),
            "generated output download") ||
        !cuda_ok(cudaStreamSynchronize(resources->stream),
            "generated output synchronization")) {
        return false;
    }
    *ns_per_item = static_cast<double>(elapsed_ms) * 1.0e6 /
        (static_cast<double>(options.iterations) * options.batch);
    *errors = compare_multivectors(
        operation, a, b, rotor, *output, true
    );
    return true;
}
#endif

void write_csv_header(std::ofstream *csv) {
    if (csv == nullptr || !csv->is_open()) return;
    *csv <<
        "operation,backend,timing_scope,precision,batch,iterations,warmup,seed,"
        "upload_bytes,download_bytes,logical_kernel_bytes,ns_per_item,"
        "max_absolute_error,max_relative_error,mismatches\n";
}

void report(
    Operation operation,
    const char *backend,
    const char *timing_scope,
    const Options &options,
    const ByteCounts &bytes,
    double ns_per_item,
    const ErrorStats &errors,
    std::ofstream *csv
) {
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif
    std::printf(
        "RESULT operation=%s backend=%s timing_scope=%s "
        "ns_per_item=%.6f max_absolute=%.17g max_relative=%.17g "
        "mismatches=%zu\n",
        operation_name(operation), backend, timing_scope, ns_per_item,
        errors.max_absolute, errors.max_relative, errors.mismatches
    );
    if (csv != nullptr && csv->is_open()) {
        *csv << operation_name(operation) << ',' << backend << ','
             << timing_scope << ',' << precision << ',' << options.batch << ','
             << options.iterations << ',' << options.warmup << ','
             << options.seed << ',' << bytes.upload_bytes << ','
             << bytes.download_bytes << ',' << bytes.logical_kernel_bytes
             << ',' << std::setprecision(17) << ns_per_item << ','
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

    geo_cuda_context_t *api_context = nullptr;
    const geo_cuda_status_t context_status =
        geo_cuda_context_create(options.device, &api_context);
    if (context_status != GEO_CUDA_SUCCESS) {
        std::fprintf(
            stderr, "CUDA context creation failed: %s\n",
            geo_cuda_status_string(context_status)
        );
        return context_status == GEO_CUDA_NO_DEVICE ? 77 : EXIT_FAILURE;
    }

    geo_cuda_device_info_t info;
    const geo_cuda_status_t info_status =
        geo_cuda_get_device_info(api_context, &info);
    if (info_status != GEO_CUDA_SUCCESS) {
        std::fprintf(
            stderr, "CUDA device query failed: %s\n",
            geo_cuda_status_string(info_status)
        );
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }
    if (!cuda_ok(cudaSetDevice(options.device), "device selection")) {
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }
    int max_grid_x = 0;
    if (!cuda_ok(cudaDeviceGetAttribute(
            &max_grid_x, cudaDevAttrMaxGridDimX, options.device),
            "maximum grid query")) {
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }
    BatchLayout layout;
    if (!prepare_layout(options.batch, max_grid_x, &layout)) {
        std::fprintf(
            stderr, "Invalid batch size %zu for element sizes and device grid limit %d\n",
            options.batch, max_grid_x
        );
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }

    std::vector<geo_cl20_t> a;
    std::vector<geo_cl20_t> b;
    std::vector<geo_cl20_t> rotor;
    std::vector<geo_cl20_t> reverse;
    std::vector<geo_cl20_t> mv_output;
    std::vector<geo_real_t> scalar_output;
    try {
        a.resize(options.batch);
        b.resize(options.batch);
        rotor.resize(options.batch);
        reverse.resize(options.batch);
        mv_output.resize(options.batch);
        scalar_output.resize(options.batch);
    } catch (const std::bad_alloc &) {
        std::fprintf(
            stderr, "Host allocation failed for batch size %zu\n",
            options.batch
        );
        geo_cuda_context_destroy(api_context);
        return EXIT_FAILURE;
    }
    uint32_t random_state = options.seed;
    for (std::size_t index = 0u; index < options.batch; ++index) {
        a[index] = random_mv(&random_state);
        b[index] = random_mv(&random_state);
        rotor[index] = random_rotor(&random_state);
        reverse[index] = geo_cl20_reverse(rotor[index]);
    }

    std::ofstream csv;
    if (!options.csv_path.empty()) {
        csv.open(options.csv_path, std::ios::out | std::ios::trunc);
        if (!csv.is_open()) {
            std::fprintf(
                stderr, "Unable to open CSV output: %s\n",
                options.csv_path.c_str()
            );
            geo_cuda_context_destroy(api_context);
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
        "Geometric Elementary Operators CUDA selected-path harness\n"
        "device: %s (compute %d.%d, %d SMs, runtime %d, driver %d)\n"
        "configuration: precision=%s batch=%zu iterations=%u warmup=%u "
        "seed=%u operation=%s\n",
        info.name, info.compute_major, info.compute_minor,
        info.multiprocessor_count, info.runtime_version, info.driver_version,
        precision, options.batch, options.iterations, options.warmup,
        options.seed, options.operation.c_str()
    );

    bool ok = true;
    for (Operation operation : kOperations) {
        if (!selected(options, operation)) continue;
        double ns_per_item = 0.0;
        ErrorStats errors;
        ByteCounts bytes;
        if (!byte_counts(operation, false, layout, &bytes)) {
            ok = false;
            break;
        }
        if (!run_public(
                operation, options, api_context, a, b, rotor,
                &mv_output, &scalar_output, &ns_per_item, &errors)) {
            ok = false;
            break;
        }
        report(
            operation, "cuda_public_api", "host_end_to_end", options,
            bytes, ns_per_item, errors, csv.is_open() ? &csv : nullptr
        );
        if (errors.mismatches != 0u) ok = false;
    }
    geo_cuda_context_destroy(api_context);

#if defined(GEO_HAVE_GENERATED_CUDA_SCHEDULES)
    bool needs_generated_resources = false;
    bool needs_generated_reverse = false;
    for (Operation operation : kOperations) {
        if (selected(options, operation) &&
            operation_has_generated_schedule(operation)) {
            needs_generated_resources = true;
            if (operation == Operation::RotorAction) {
                needs_generated_reverse = true;
            }
        }
    }
    GeneratedResources generated;
    generated.device = options.device;
    if (ok && needs_generated_resources &&
        (!cuda_ok(cudaStreamCreateWithFlags(
                    &generated.stream, cudaStreamNonBlocking),
                    "generated stream creation") ||
               !allocate_device(
                    &generated.device_a, layout.mv_bytes,
                    "generated allocate r0") ||
               !allocate_device(
                    &generated.device_b, layout.mv_bytes,
                    "generated allocate r1") ||
               !allocate_device(
                    &generated.device_output, layout.mv_bytes,
                    "generated allocate output"))) {
        ok = false;
    }
    if (ok && needs_generated_reverse &&
        !allocate_device(
            &generated.device_reverse, layout.mv_bytes,
            "generated allocate r2")) {
        ok = false;
    }
    if (ok && needs_generated_resources) {
        for (Operation operation : kOperations) {
            if (!selected(options, operation) ||
                !operation_has_generated_schedule(operation)) {
                continue;
            }
            double ns_per_item = 0.0;
            ErrorStats errors;
            ByteCounts bytes;
            if (!byte_counts(operation, true, layout, &bytes)) {
                ok = false;
                break;
            }
            if (!run_generated(
                    operation, options, layout, a, b, rotor, reverse,
                    &mv_output, &generated, &ns_per_item, &errors)) {
                ok = false;
                break;
            }
            report(
                operation, "cuda_generated_schedule", "device_kernel",
                options, bytes, ns_per_item, errors,
                csv.is_open() ? &csv : nullptr
            );
            if (errors.mismatches != 0u) ok = false;
        }
    }
#endif

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
