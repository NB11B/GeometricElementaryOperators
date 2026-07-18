#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kMaxDimension = 6;
constexpr int kMaxBlades = 64;
constexpr int kMaxTerms = 3;
constexpr int kMaxContributions = kMaxBlades * kMaxTerms;
constexpr int kPrime = 65521;
constexpr int kSideRight = 0;
constexpr int kSideLeft = 1;

struct OperatorCase {
    int dimension;
    int signature[kMaxDimension];
    int side;
    int term_count;
    int blades[kMaxTerms];
    int coefficients[kMaxTerms];
    int contribution_count;
    int sources[kMaxContributions];
    int targets[kMaxContributions];
    int factors[kMaxContributions];
};

__host__ __device__ int popcount_u32(std::uint32_t value) {
    int count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

__host__ __device__ int gp_sign(int left, int right, const OperatorCase &test_case) {
    int sign = 1;
    for (int index = 0; index < test_case.dimension; ++index) {
        const int bit = 1 << index;
        if ((left & bit) != 0) {
            if ((popcount_u32(static_cast<std::uint32_t>(right & (bit - 1))) & 1) != 0) {
                sign = -sign;
            }
            if ((right & bit) != 0) sign *= test_case.signature[index];
        }
    }
    return sign;
}

__host__ __device__ int normalize(long long value) {
    value %= kPrime;
    if (value < 0) value += kPrime;
    return static_cast<int>(value);
}

__host__ __device__ int input_coefficient(std::uint64_t assignment, int source) {
    std::uint64_t value = assignment + 1U;
    value ^= static_cast<std::uint64_t>(source + 1) * UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31U;
    return static_cast<int>(value % 7U) - 3;
}

__global__ void compare_kernel(
    const OperatorCase *cases,
    int case_count,
    std::uint64_t assignments,
    unsigned long long *mismatches
) {
    const std::uint64_t global =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(case_count) * assignments;
    if (global >= total) return;

    const int case_index = static_cast<int>(global / assignments);
    const std::uint64_t assignment = global % assignments;
    const OperatorCase test_case = cases[case_index];
    const int count = 1 << test_case.dimension;
    int generic[kMaxBlades];
    int specialized[kMaxBlades];
    for (int blade = 0; blade < count; ++blade) {
        generic[blade] = 0;
        specialized[blade] = 0;
    }

    for (int source = 0; source < count; ++source) {
        const int input = input_coefficient(assignment, source);
        if (input == 0) continue;
        for (int term = 0; term < test_case.term_count; ++term) {
            const int fixed_blade = test_case.blades[term];
            const int fixed_coefficient = test_case.coefficients[term];
            const int target = source ^ fixed_blade;
            const int sign = test_case.side == kSideRight
                ? gp_sign(source, fixed_blade, test_case)
                : gp_sign(fixed_blade, source, test_case);
            generic[target] = normalize(
                static_cast<long long>(generic[target]) +
                static_cast<long long>(sign) * fixed_coefficient * input
            );
        }
    }

    for (int row = 0; row < test_case.contribution_count; ++row) {
        const int source = test_case.sources[row];
        const int input = input_coefficient(assignment, source);
        if (input == 0) continue;
        const int target = test_case.targets[row];
        specialized[target] = normalize(
            static_cast<long long>(specialized[target]) +
            static_cast<long long>(test_case.factors[row]) * input
        );
    }

    for (int blade = 0; blade < count; ++blade) {
        if (generic[blade] != specialized[blade]) {
            atomicAdd(mismatches, 1ULL);
            return;
        }
    }
}

void check_cuda(cudaError_t status, const char *context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", context, cudaGetErrorString(status));
        std::exit(2);
    }
}

OperatorCase make_case(int dimension, int q, int side, bool sparse) {
    OperatorCase output{};
    output.dimension = dimension;
    output.side = side;
    for (int index = 0; index < kMaxDimension; ++index) {
        output.signature[index] =
            index < dimension - q ? 1 : (index < dimension ? -1 : 0);
    }
    const int pseudoscalar = (1 << dimension) - 1;
    if (sparse) {
        output.term_count = 3;
        output.blades[0] = 0;
        output.coefficients[0] = 1;
        output.blades[1] = 1;
        output.coefficients[1] = -2;
        output.blades[2] = pseudoscalar;
        output.coefficients[2] = 1;
    } else {
        output.term_count = 1;
        output.blades[0] = pseudoscalar;
        output.coefficients[0] = 1;
    }
    const int count = 1 << dimension;
    for (int source = 0; source < count; ++source) {
        for (int term = 0; term < output.term_count; ++term) {
            const int fixed_blade = output.blades[term];
            const int index = output.contribution_count++;
            output.sources[index] = source;
            output.targets[index] = source ^ fixed_blade;
            output.factors[index] = output.coefficients[term] * (
                side == kSideRight
                    ? gp_sign(source, fixed_blade, output)
                    : gp_sign(fixed_blade, source, output)
            );
        }
    }
    return output;
}

}  // namespace

int main(int argc, char **argv) {
    std::uint64_t assignments = 1024;
    int device = 0;
    if (argc > 1) assignments = static_cast<std::uint64_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2) device = std::atoi(argv[2]);
    if (assignments == 0) {
        std::fprintf(stderr, "assignments must be positive\n");
        return 2;
    }

    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    std::vector<OperatorCase> cases;
    for (int dimension = 2; dimension <= 6; ++dimension) {
        for (int q = 0; q <= dimension; ++q) {
            for (int side = kSideRight; side <= kSideLeft; ++side) {
                cases.push_back(make_case(dimension, q, side, false));
                cases.push_back(make_case(dimension, q, side, true));
            }
        }
    }

    OperatorCase *device_cases = nullptr;
    unsigned long long *device_mismatches = nullptr;
    check_cuda(cudaMalloc(&device_cases, cases.size() * sizeof(OperatorCase)), "cudaMalloc cases");
    check_cuda(cudaMalloc(&device_mismatches, sizeof(unsigned long long)), "cudaMalloc mismatches");
    check_cuda(
        cudaMemcpy(
            device_cases,
            cases.data(),
            cases.size() * sizeof(OperatorCase),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy cases"
    );
    check_cuda(cudaMemset(device_mismatches, 0, sizeof(unsigned long long)), "cudaMemset mismatches");

    const std::uint64_t total = static_cast<std::uint64_t>(cases.size()) * assignments;
    const int block_size = 128;
    const int block_count = static_cast<int>((total + block_size - 1U) / block_size);
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    compare_kernel<<<block_count, block_size>>>(
        device_cases,
        static_cast<int>(cases.size()),
        assignments,
        device_mismatches
    );
    check_cuda(cudaGetLastError(), "compare_kernel launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

    unsigned long long mismatches = 0;
    check_cuda(
        cudaMemcpy(
            &mismatches,
            device_mismatches,
            sizeof(mismatches),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy mismatches"
    );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_cases);
    cudaFree(device_mismatches);

    const double seconds = static_cast<double>(milliseconds) / 1000.0;
    const double assignments_per_second = seconds > 0.0
        ? static_cast<double>(total) / seconds
        : 0.0;
    std::printf(
        "GEO_OPERATOR_CUDA,device=%d,name=%s,cc=%d.%d,cases=%zu,assignments=%llu,total=%llu,kernel_ms=%.6f,assignments_per_second=%.3f,mismatches=%llu\n",
        device,
        properties.name,
        properties.major,
        properties.minor,
        cases.size(),
        static_cast<unsigned long long>(assignments),
        static_cast<unsigned long long>(total),
        static_cast<double>(milliseconds),
        assignments_per_second,
        mismatches
    );
    if (mismatches != 0) {
        std::printf("GEO_OPERATOR_V5_1_CUDA: FAIL\n");
        return 1;
    }
    std::printf(
        "GEO_OPERATOR_V5_1_CUDA: PASS dimensions=2-6 signatures=25 cases=%zu\n",
        cases.size()
    );
    return 0;
}
