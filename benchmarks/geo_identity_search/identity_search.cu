#include "geo_identity_corpus.cuh"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kDefaultAssignments = UINT64_C(1) << 20U;
constexpr uint32_t kDefaultBlockSize = 256U;
constexpr uint64_t kDefaultCpuChecks = UINT64_C(4096);

struct Options {
    int device = 0;
    uint64_t assignments = kDefaultAssignments;
    uint32_t block_size = kDefaultBlockSize;
    uint64_t cpu_checks = kDefaultCpuChecks;
    std::string csv_path;
};

struct SearchResult {
    const char *name = nullptr;
    const char *expected = nullptr;
    int dimension = 0;
    std::string signature;
    int prime = 0;
    int variable_count = 0;
    int node_count = 0;
    uint64_t assignments = 0;
    uint64_t cpu_checks = 0;
    float kernel_us = 0.0f;
    double assignments_per_second = 0.0;
    bool found_counterexample = false;
    uint64_t witness_assignment = UINT64_MAX;
    uint16_t witness_blade = 0;
    int32_t witness_lhs = 0;
    int32_t witness_rhs = 0;
    bool passed = false;
};

bool cuda_ok(cudaError_t status, const char *stage) {
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr,
        "CUDA failure during %s: %s\n",
        stage,
        cudaGetErrorString(status)
    );
    return false;
}

uint64_t parse_u64(const char *text, const char *name) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        std::fprintf(stderr, "Invalid %s\n", name);
        std::exit(EXIT_FAILURE);
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        std::fprintf(stderr, "Invalid %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint64_t>(value);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s [--device N] [--assignments N] [--block-size N] "
                "[--cpu-checks N] [--csv PATH]\n",
                argv[0]
            );
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value after %s\n", argument.c_str());
            std::exit(EXIT_FAILURE);
        }
        const char *value = argv[++index];
        if (argument == "--device") {
            const uint64_t parsed = parse_u64(value, "device");
            if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                std::fprintf(stderr, "Device index is too large\n");
                std::exit(EXIT_FAILURE);
            }
            options.device = static_cast<int>(parsed);
        } else if (argument == "--assignments") {
            options.assignments = parse_u64(value, "assignments");
            if (options.assignments == 0U) {
                std::fprintf(stderr, "Assignments must be nonzero\n");
                std::exit(EXIT_FAILURE);
            }
        } else if (argument == "--block-size") {
            const uint64_t parsed = parse_u64(value, "block-size");
            if (parsed == 0U || parsed > 1024U) {
                std::fprintf(stderr, "Block size must be in [1,1024]\n");
                std::exit(EXIT_FAILURE);
            }
            options.block_size = static_cast<uint32_t>(parsed);
        } else if (argument == "--cpu-checks") {
            options.cpu_checks = parse_u64(value, "cpu-checks");
        } else if (argument == "--csv") {
            options.csv_path = value;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argument.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    return options;
}

template <int ID>
__global__ void search_kernel(
    uint64_t assignments,
    unsigned long long *first_counterexample
) {
    const uint64_t assignment =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (assignment >= assignments) return;

    const auto witness =
        geo_identity_generated::identity<ID>::evaluate(assignment);
    if (!witness.equal) {
        atomicMin(first_counterexample, static_cast<unsigned long long>(assignment));
    }
}

template <int ID>
std::string signature_string() {
    using identity_t = geo_identity_generated::identity<ID>;
    std::string result;
    result.reserve(static_cast<std::size_t>(identity_t::DIMENSION) * 3U);
    for (int index = 0; index < identity_t::DIMENSION; ++index) {
        if (index != 0) result.push_back(',');
        result += identity_t::signature_at(index) > 0 ? "+1" : "-1";
    }
    return result;
}

template <int ID>
bool run_identity(const Options &options, SearchResult *result) {
    using identity_t = geo_identity_generated::identity<ID>;

    result->name = identity_t::NAME;
    result->expected =
        identity_t::EXPECT_COUNTEREXAMPLE ? "counterexample" : "identity";
    result->dimension = identity_t::DIMENSION;
    result->signature = signature_string<ID>();
    result->prime = identity_t::PRIME;
    result->variable_count = identity_t::VARIABLE_COUNT;
    result->node_count = identity_t::NODE_COUNT;
    result->assignments = options.assignments;
    result->cpu_checks =
        options.cpu_checks < options.assignments
        ? options.cpu_checks
        : options.assignments;

    bool cpu_found_counterexample = false;
    uint64_t cpu_first = UINT64_MAX;
    for (uint64_t assignment = 0; assignment < result->cpu_checks; ++assignment) {
        const auto witness = identity_t::evaluate(assignment);
        if (!witness.equal) {
            cpu_found_counterexample = true;
            cpu_first = assignment;
            if (!identity_t::EXPECT_COUNTEREXAMPLE) {
                std::fprintf(
                    stderr,
                    "CPU exact check rejected identity %s at assignment %llu, "
                    "blade %u: lhs=%d rhs=%d\n",
                    identity_t::NAME,
                    static_cast<unsigned long long>(assignment),
                    static_cast<unsigned int>(witness.blade),
                    witness.lhs,
                    witness.rhs
                );
                return false;
            }
            break;
        }
    }

    if (
        identity_t::EXPECT_COUNTEREXAMPLE &&
        result->cpu_checks > 0U &&
        !cpu_found_counterexample
    ) {
        std::fprintf(
            stderr,
            "CPU exact check found no counterexample for %s in %llu assignments\n",
            identity_t::NAME,
            static_cast<unsigned long long>(result->cpu_checks)
        );
        return false;
    }

    unsigned long long *device_first = nullptr;
    if (!cuda_ok(
            cudaMalloc(reinterpret_cast<void **>(&device_first),
                       sizeof(unsigned long long)),
            "counterexample allocation")) {
        return false;
    }

    const unsigned long long sentinel = ULLONG_MAX;
    if (!cuda_ok(
            cudaMemcpy(
                device_first,
                &sentinel,
                sizeof(sentinel),
                cudaMemcpyHostToDevice),
            "counterexample initialization")) {
        cudaFree(device_first);
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "event create start") ||
        !cuda_ok(cudaEventCreate(&stop), "event create stop")) {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        cudaFree(device_first);
        return false;
    }

    const uint64_t block_count_u64 =
        options.assignments / options.block_size +
        (options.assignments % options.block_size == 0U ? 0U : 1U);
    if (block_count_u64 > static_cast<uint64_t>(UINT_MAX)) {
        std::fprintf(stderr, "Grid is too large for %s\n", identity_t::NAME);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(device_first);
        return false;
    }

    if (!cuda_ok(cudaEventRecord(start), "event record start")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(device_first);
        return false;
    }

    search_kernel<ID><<<
        static_cast<unsigned int>(block_count_u64),
        options.block_size
    >>>(options.assignments, device_first);

    if (!cuda_ok(cudaGetLastError(), "identity search launch") ||
        !cuda_ok(cudaEventRecord(stop), "event record stop") ||
        !cuda_ok(cudaEventSynchronize(stop), "event synchronize stop")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(device_first);
        return false;
    }

    float elapsed_ms = 0.0f;
    if (!cuda_ok(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "event elapsed time")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(device_first);
        return false;
    }

    unsigned long long first = ULLONG_MAX;
    if (!cuda_ok(
            cudaMemcpy(
                &first,
                device_first,
                sizeof(first),
                cudaMemcpyDeviceToHost),
            "counterexample download")) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(device_first);
        return false;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_first);

    result->kernel_us = elapsed_ms * 1000.0f;
    result->assignments_per_second =
        static_cast<double>(options.assignments) /
        (static_cast<double>(result->kernel_us) * 1.0e-6);
    result->found_counterexample = first != ULLONG_MAX;

    if (result->found_counterexample) {
        result->witness_assignment = static_cast<uint64_t>(first);
        const auto witness = identity_t::evaluate(result->witness_assignment);
        if (witness.equal) {
            std::fprintf(
                stderr,
                "GPU reported a counterexample for %s that host evaluation "
                "could not reproduce\n",
                identity_t::NAME
            );
            return false;
        }
        result->witness_blade = witness.blade;
        result->witness_lhs = witness.lhs;
        result->witness_rhs = witness.rhs;
    }

    result->passed =
        identity_t::EXPECT_COUNTEREXAMPLE
        ? result->found_counterexample
        : !result->found_counterexample;

    if (!result->passed) {
        std::fprintf(
            stderr,
            "%s: expected %s but GPU found_counterexample=%s\n",
            identity_t::NAME,
            result->expected,
            result->found_counterexample ? "true" : "false"
        );
        return false;
    }

    if (
        identity_t::EXPECT_COUNTEREXAMPLE &&
        cpu_found_counterexample &&
        result->witness_assignment > cpu_first
    ) {
        std::fprintf(
            stderr,
            "%s: GPU first witness %llu is later than CPU witness %llu\n",
            identity_t::NAME,
            static_cast<unsigned long long>(result->witness_assignment),
            static_cast<unsigned long long>(cpu_first)
        );
        return false;
    }

    return true;
}

bool dispatch_identity(
    int identity_index,
    const Options &options,
    SearchResult *result
) {
    switch (identity_index) {
#define GEO_IDENTITY_CASE(ID) \
        case ID: return run_identity<ID>(options, result);
        GEO_IDENTITY_FOR_EACH(GEO_IDENTITY_CASE)
#undef GEO_IDENTITY_CASE
        default:
            return false;
    }
}

void print_csv_header(FILE *stream) {
    std::fprintf(
        stream,
        "CSV_IDENTITY,identity,expected,dimension,signature,prime,variables,"
        "nodes,assignments,cpu_checks,kernel_us,assignments_per_second,"
        "found_counterexample,witness_assignment,witness_blade,witness_lhs,"
        "witness_rhs,result\n"
    );
}

void print_csv_row(FILE *stream, const SearchResult &result) {
    std::fprintf(
        stream,
        "CSV_IDENTITY,%s,%s,%d,\"%s\",%d,%d,%d,%llu,%llu,%.6f,%.3f,%s,",
        result.name,
        result.expected,
        result.dimension,
        result.signature.c_str(),
        result.prime,
        result.variable_count,
        result.node_count,
        static_cast<unsigned long long>(result.assignments),
        static_cast<unsigned long long>(result.cpu_checks),
        static_cast<double>(result.kernel_us),
        result.assignments_per_second,
        result.found_counterexample ? "true" : "false"
    );
    if (result.found_counterexample) {
        std::fprintf(
            stream,
            "%llu,%u,%d,%d,",
            static_cast<unsigned long long>(result.witness_assignment),
            static_cast<unsigned int>(result.witness_blade),
            result.witness_lhs,
            result.witness_rhs
        );
    } else {
        std::fprintf(stream, ",,,,");
    }
    std::fprintf(stream, "%s\n", result.passed ? "pass" : "fail");
}

}  // namespace

int main(int argc, char **argv) {
    const Options options = parse_options(argc, argv);
    if (!cuda_ok(cudaSetDevice(options.device), "device selection")) {
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(
            cudaGetDeviceProperties(&properties, options.device),
            "device properties")) {
        return EXIT_FAILURE;
    }

    int runtime_version = 0;
    int driver_version = 0;
    if (!cuda_ok(cudaRuntimeGetVersion(&runtime_version), "runtime version") ||
        !cuda_ok(cudaDriverGetVersion(&driver_version), "driver version")) {
        return EXIT_FAILURE;
    }

    std::printf(
        "GEO_IDENTITY_SEARCH,device=%d,name=%s,cc=%d.%d,runtime=%d,driver=%d,"
        "identities=%d,assignments=%llu,block_size=%u,cpu_checks=%llu\n",
        options.device,
        properties.name,
        properties.major,
        properties.minor,
        runtime_version,
        driver_version,
        geo_identity_generated::IDENTITY_COUNT,
        static_cast<unsigned long long>(options.assignments),
        options.block_size,
        static_cast<unsigned long long>(options.cpu_checks)
    );

    FILE *csv = nullptr;
    if (!options.csv_path.empty()) {
        csv = std::fopen(options.csv_path.c_str(), "wb");
        if (csv == nullptr) {
            std::fprintf(
                stderr,
                "Unable to open CSV path: %s\n",
                options.csv_path.c_str()
            );
            return EXIT_FAILURE;
        }
    }

    print_csv_header(stdout);
    if (csv != nullptr) print_csv_header(csv);

    bool all_passed = true;
    std::vector<SearchResult> results(
        static_cast<std::size_t>(geo_identity_generated::IDENTITY_COUNT)
    );
    for (
        int identity_index = 0;
        identity_index < geo_identity_generated::IDENTITY_COUNT;
        ++identity_index
    ) {
        const bool passed =
            dispatch_identity(identity_index, options, &results[identity_index]);
        all_passed = all_passed && passed;
        print_csv_row(stdout, results[identity_index]);
        if (csv != nullptr) print_csv_row(csv, results[identity_index]);
    }

    if (csv != nullptr) std::fclose(csv);

    std::printf(
        "GEO_IDENTITY_SEARCH,status=%s\n",
        all_passed ? "complete" : "fail"
    );
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
