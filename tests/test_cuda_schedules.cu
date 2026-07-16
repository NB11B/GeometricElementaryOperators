#include "geo/cl20.h"
#include "geo/cuda.h"
#include "geo_cuda_schedule_addition.h"
#include "geo_cuda_schedule_geometric_product.h"
#include "geo_cuda_schedule_rotor_action.h"
#include "geo_cuda_schedule_vector_dot.h"
#include "geo_cuda_schedule_vector_wedge.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kCount = 257u;
constexpr unsigned int kBlockSize = 128u;

struct Resources {
    geo_cuda_context_t *context = nullptr;
    geo_cl20_t *d0 = nullptr;
    geo_cl20_t *d1 = nullptr;
    geo_cl20_t *d2 = nullptr;
    geo_cl20_t *output = nullptr;

    ~Resources() {
        if (output != nullptr) cudaFree(output);
        if (d2 != nullptr) cudaFree(d2);
        if (d1 != nullptr) cudaFree(d1);
        if (d0 != nullptr) cudaFree(d0);
        geo_cuda_context_destroy(context);
    }
};

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
    const double angle = 0.001 * static_cast<double>((index % 101u) + 1u);
    return geo_cl20_make(
        static_cast<geo_real_t>(std::cos(angle)),
        static_cast<geo_real_t>(0),
        static_cast<geo_real_t>(0),
        static_cast<geo_real_t>(-std::sin(angle))
    );
}

bool scalar_near(geo_real_t actual, geo_real_t expected) {
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const double absolute_tolerance = 2.0e-12;
    const double relative_tolerance = 2.0e-12;
#else
    const double absolute_tolerance = 4.0e-5;
    const double relative_tolerance = 4.0e-5;
#endif
    const double a = static_cast<double>(actual);
    const double e = static_cast<double>(expected);
    const double error = std::fabs(a - e);
    return std::isfinite(a) && std::isfinite(e) &&
        error <= absolute_tolerance + relative_tolerance * std::fabs(e);
}

bool mv_near(geo_cl20_t actual, geo_cl20_t expected) {
    return scalar_near(actual.scalar, expected.scalar) &&
        scalar_near(actual.e1, expected.e1) &&
        scalar_near(actual.e2, expected.e2) &&
        scalar_near(actual.e12, expected.e12);
}

int fail_cuda(const char *stage, cudaError_t error) {
    std::fprintf(stderr, "FAIL: %s: %s\n", stage, cudaGetErrorString(error));
    return EXIT_FAILURE;
}

int fail_launch(const char *stage, int status) {
    return fail_cuda(stage, static_cast<cudaError_t>(status));
}

int allocate(geo_cl20_t **pointer, std::size_t bytes, const char *stage) {
    const cudaError_t error = cudaMalloc(reinterpret_cast<void **>(pointer), bytes);
    if (error == cudaSuccess) return EXIT_SUCCESS;
    return fail_cuda(stage, error);
}

int upload(
    geo_cl20_t *destination,
    const std::vector<geo_cl20_t> &source,
    const char *stage
) {
    const cudaError_t error = cudaMemcpy(
        destination,
        source.data(),
        source.size() * sizeof(geo_cl20_t),
        cudaMemcpyHostToDevice
    );
    if (error == cudaSuccess) return EXIT_SUCCESS;
    return fail_cuda(stage, error);
}

int download(
    std::vector<geo_cl20_t> *destination,
    const geo_cl20_t *source,
    const char *stage
) {
    const cudaError_t error = cudaMemcpy(
        destination->data(),
        source,
        destination->size() * sizeof(geo_cl20_t),
        cudaMemcpyDeviceToHost
    );
    if (error == cudaSuccess) return EXIT_SUCCESS;
    return fail_cuda(stage, error);
}

int synchronize(const char *stage) {
    const cudaError_t error = cudaDeviceSynchronize();
    if (error == cudaSuccess) return EXIT_SUCCESS;
    return fail_cuda(stage, error);
}

int check_output(
    const char *operation,
    const std::vector<geo_cl20_t> &actual,
    const std::vector<geo_cl20_t> &expected
) {
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        if (!mv_near(actual[index], expected[index])) {
            std::fprintf(
                stderr,
                "FAIL: %s mismatch at index %zu\n",
                operation,
                index
            );
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main() {
    Resources resources;
    geo_cuda_status_t context_status = geo_cuda_context_create(0, &resources.context);
    if (context_status == GEO_CUDA_NO_DEVICE) {
        std::puts("SKIP: no CUDA device or compatible driver");
        return 77;
    }
    if (context_status != GEO_CUDA_SUCCESS) {
        std::fprintf(
            stderr,
            "FAIL: context creation: %s\n",
            geo_cuda_status_string(context_status)
        );
        return EXIT_FAILURE;
    }

    const std::size_t bytes = kCount * sizeof(geo_cl20_t);
    if (allocate(&resources.d0, bytes, "allocate input 0") != EXIT_SUCCESS ||
        allocate(&resources.d1, bytes, "allocate input 1") != EXIT_SUCCESS ||
        allocate(&resources.d2, bytes, "allocate input 2") != EXIT_SUCCESS ||
        allocate(&resources.output, bytes, "allocate output") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    std::vector<geo_cl20_t> a(kCount);
    std::vector<geo_cl20_t> b(kCount);
    std::vector<geo_cl20_t> rotor(kCount);
    std::vector<geo_cl20_t> reverse(kCount);
    std::vector<geo_cl20_t> actual(kCount);
    std::vector<geo_cl20_t> expected(kCount);
    uint32_t random_state = UINT32_C(0x9e3779b9);

    for (std::size_t index = 0u; index < kCount; ++index) {
        a[index] = random_mv(random_state);
        b[index] = random_mv(random_state);
        rotor[index] = rotor_for(index);
        reverse[index] = geo_cl20_reverse(rotor[index]);
    }

    if (upload(resources.d0, a, "upload a") != EXIT_SUCCESS ||
        upload(resources.d1, b, "upload b") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    int launch_status = geo_cuda_schedule_addition_launch(
        resources.d0,
        resources.d1,
        resources.output,
        kCount,
        kBlockSize,
        nullptr
    );
    if (launch_status != static_cast<int>(cudaSuccess)) {
        return fail_launch("addition launch", launch_status);
    }
    if (synchronize("addition synchronize") != EXIT_SUCCESS ||
        download(&actual, resources.output, "addition download") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        expected[index] = geo_cl20_add(a[index], b[index]);
    }
    if (check_output("addition", actual, expected) != EXIT_SUCCESS) return EXIT_FAILURE;

    launch_status = geo_cuda_schedule_geometric_product_launch(
        resources.d0,
        resources.d1,
        resources.output,
        kCount,
        kBlockSize,
        nullptr
    );
    if (launch_status != static_cast<int>(cudaSuccess)) {
        return fail_launch("product launch", launch_status);
    }
    if (synchronize("product synchronize") != EXIT_SUCCESS ||
        download(&actual, resources.output, "product download") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        expected[index] = geo_cl20_mul(a[index], b[index]);
    }
    if (check_output("geometric product", actual, expected) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    launch_status = geo_cuda_schedule_vector_dot_launch(
        resources.d0,
        resources.d1,
        resources.output,
        kCount,
        kBlockSize,
        nullptr
    );
    if (launch_status != static_cast<int>(cudaSuccess)) {
        return fail_launch("dot launch", launch_status);
    }
    if (synchronize("dot synchronize") != EXIT_SUCCESS ||
        download(&actual, resources.output, "dot download") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        expected[index] = geo_cl20_make(
            geo_cl20_vector_dot(a[index], b[index]),
            static_cast<geo_real_t>(0),
            static_cast<geo_real_t>(0),
            static_cast<geo_real_t>(0)
        );
    }
    if (check_output("vector dot", actual, expected) != EXIT_SUCCESS) return EXIT_FAILURE;

    launch_status = geo_cuda_schedule_vector_wedge_launch(
        resources.d0,
        resources.d1,
        resources.output,
        kCount,
        kBlockSize,
        nullptr
    );
    if (launch_status != static_cast<int>(cudaSuccess)) {
        return fail_launch("wedge launch", launch_status);
    }
    if (synchronize("wedge synchronize") != EXIT_SUCCESS ||
        download(&actual, resources.output, "wedge download") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        expected[index] = geo_cl20_make(
            static_cast<geo_real_t>(0),
            static_cast<geo_real_t>(0),
            static_cast<geo_real_t>(0),
            geo_cl20_vector_wedge(a[index], b[index])
        );
    }
    if (check_output("vector wedge", actual, expected) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    if (upload(resources.d0, rotor, "upload rotor") != EXIT_SUCCESS ||
        upload(resources.d1, a, "upload rotor value") != EXIT_SUCCESS ||
        upload(resources.d2, reverse, "upload rotor reverse") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    launch_status = geo_cuda_schedule_rotor_action_launch(
        resources.d0,
        resources.d1,
        resources.d2,
        resources.output,
        kCount,
        kBlockSize,
        nullptr
    );
    if (launch_status != static_cast<int>(cudaSuccess)) {
        return fail_launch("rotor launch", launch_status);
    }
    if (synchronize("rotor synchronize") != EXIT_SUCCESS ||
        download(&actual, resources.output, "rotor download") != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        expected[index] = geo_cl20_mul(
            geo_cl20_mul(rotor[index], a[index]),
            reverse[index]
        );
    }
    if (check_output("rotor action", actual, expected) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    if (geo_cuda_schedule_addition_launch(
            resources.d0,
            resources.d1,
            resources.output,
            1u,
            0u,
            nullptr) != static_cast<int>(cudaErrorInvalidValue)) {
        std::fputs("FAIL: generated launcher must reject a zero block size\n", stderr);
        return EXIT_FAILURE;
    }

    std::puts("PASS generated CUDA schedule equivalence");
    return EXIT_SUCCESS;
}
