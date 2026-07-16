#include "geo/cl20.h"
#include "geo/cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kCount = 257u;

uint32_t next_random(uint32_t &state) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    return state;
}

geo_real_t sample(uint32_t &state) {
    const uint32_t bits = next_random(state) >> 8;
    const double unit = static_cast<double>(bits) / static_cast<double>(UINT32_C(0x00ffffff));
    return static_cast<geo_real_t>((unit * 4.0) - 2.0);
}

geo_cl20_t random_mv(uint32_t &state) {
    return geo_cl20_make(sample(state), sample(state), sample(state), sample(state));
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
    const double absolute_tolerance = 1.0e-12;
    const double relative_tolerance = 1.0e-12;
#else
    const double absolute_tolerance = 3.0e-5;
    const double relative_tolerance = 3.0e-5;
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

int fail_status(const char *operation, geo_cuda_status_t status) {
    std::fprintf(stderr, "FAIL: %s: %s\n", operation, geo_cuda_status_string(status));
    return EXIT_FAILURE;
}

int fail_index(const char *operation, std::size_t index) {
    std::fprintf(stderr, "FAIL: %s mismatch at index %zu\n", operation, index);
    return EXIT_FAILURE;
}

}  // namespace

int main() {
    geo_cuda_context_t *context = nullptr;
    geo_cuda_status_t status = geo_cuda_context_create(0, &context);
    if (status == GEO_CUDA_NO_DEVICE) {
        std::puts("SKIP: no CUDA device or compatible driver");
        return 77;
    }
    if (status != GEO_CUDA_SUCCESS) return fail_status("context creation", status);

    geo_cuda_device_info_t info;
    status = geo_cuda_get_device_info(context, &info);
    if (status != GEO_CUDA_SUCCESS) {
        geo_cuda_context_destroy(context);
        return fail_status("device query", status);
    }

    std::vector<geo_cl20_t> a(kCount);
    std::vector<geo_cl20_t> b(kCount);
    std::vector<geo_cl20_t> rotor(kCount);
    std::vector<geo_cl20_t> output(kCount);
    std::vector<geo_real_t> scalar_output(kCount);
    uint32_t random_state = UINT32_C(0x6d2b79f5);

    for (std::size_t i = 0; i < kCount; ++i) {
        a[i] = random_mv(random_state);
        b[i] = random_mv(random_state);
        rotor[i] = rotor_for(i);
    }

    status = geo_cuda_cl20_add_batch(context, a.data(), b.data(), output.data(), kCount);
    if (status != GEO_CUDA_SUCCESS) return fail_status("addition", status);
    for (std::size_t i = 0; i < kCount; ++i) {
        if (!mv_near(output[i], geo_cl20_add(a[i], b[i]))) return fail_index("addition", i);
    }

    status = geo_cuda_cl20_product_batch(context, a.data(), b.data(), output.data(), kCount);
    if (status != GEO_CUDA_SUCCESS) return fail_status("product", status);
    for (std::size_t i = 0; i < kCount; ++i) {
        if (!mv_near(output[i], geo_cl20_mul(a[i], b[i]))) return fail_index("product", i);
    }

    status = geo_cuda_cl20_reverse_batch(context, a.data(), output.data(), kCount);
    if (status != GEO_CUDA_SUCCESS) return fail_status("reverse", status);
    for (std::size_t i = 0; i < kCount; ++i) {
        if (!mv_near(output[i], geo_cl20_reverse(a[i]))) return fail_index("reverse", i);
    }

    status = geo_cuda_cl20_vector_dot_batch(context, a.data(), b.data(), scalar_output.data(), kCount);
    if (status != GEO_CUDA_SUCCESS) return fail_status("vector dot", status);
    for (std::size_t i = 0; i < kCount; ++i) {
        if (!scalar_near(scalar_output[i], geo_cl20_vector_dot(a[i], b[i]))) {
            return fail_index("vector dot", i);
        }
    }

    status = geo_cuda_cl20_vector_wedge_batch(context, a.data(), b.data(), scalar_output.data(), kCount);
    if (status != GEO_CUDA_SUCCESS) return fail_status("vector wedge", status);
    for (std::size_t i = 0; i < kCount; ++i) {
        if (!scalar_near(scalar_output[i], geo_cl20_vector_wedge(a[i], b[i]))) {
            return fail_index("vector wedge", i);
        }
    }

    status = geo_cuda_cl20_rotor_action_batch(context, rotor.data(), a.data(), output.data(), kCount);
    if (status != GEO_CUDA_SUCCESS) return fail_status("rotor action", status);
    for (std::size_t i = 0; i < kCount; ++i) {
        const geo_cl20_t expected = geo_cl20_mul(
            geo_cl20_mul(rotor[i], a[i]),
            geo_cl20_reverse(rotor[i])
        );
        if (!mv_near(output[i], expected)) return fail_index("rotor action", i);
    }

    if (geo_cuda_cl20_add_batch(context, nullptr, nullptr, nullptr, 0u) != GEO_CUDA_SUCCESS) {
        std::fputs("FAIL: zero-length batch must be accepted\n", stderr);
        geo_cuda_context_destroy(context);
        return EXIT_FAILURE;
    }
    if (geo_cuda_cl20_add_batch(context, nullptr, b.data(), output.data(), 1u) !=
        GEO_CUDA_INVALID_ARGUMENT) {
        std::fputs("FAIL: non-empty null input must be rejected\n", stderr);
        geo_cuda_context_destroy(context);
        return EXIT_FAILURE;
    }

    std::printf(
        "PASS CUDA equivalence on %s (compute %d.%d, runtime %d, driver %d)\n",
        info.name,
        info.compute_major,
        info.compute_minor,
        info.runtime_version,
        info.driver_version
    );
    geo_cuda_context_destroy(context);
    return EXIT_SUCCESS;
}
