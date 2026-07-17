#include "geo/cl20.h"
#include "geo/cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kCount = 257u;

uint32_t next_random(uint32_t &state) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    return state;
}

geo_real_t sample(uint32_t &state) {
    const uint32_t bits = next_random(state) >> 8;
    const double unit = static_cast<double>(bits) /
        static_cast<double>(UINT32_C(0x00ffffff));
    return static_cast<geo_real_t>((unit * 4.0) - 2.0);
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

int fail_status(
    const char *operation,
    geo_cuda_status_t status,
    geo_cuda_context_t *context
) {
    std::fprintf(
        stderr,
        "FAIL: %s: %s\n",
        operation,
        geo_cuda_status_string(status)
    );
    geo_cuda_context_destroy(context);
    return EXIT_FAILURE;
}

int fail_index(
    const char *operation,
    std::size_t index,
    geo_cuda_context_t *context
) {
    std::fprintf(stderr, "FAIL: %s mismatch at index %zu\n", operation, index);
    geo_cuda_context_destroy(context);
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
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("context creation", status, context);
    }

    geo_cuda_device_info_t info;
    status = geo_cuda_get_device_info(context, &info);
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("device query", status, context);
    }

    std::vector<geo_cl20_t> a(kCount);
    std::vector<geo_cl20_t> b(kCount);
    std::vector<geo_cl20_t> rotor(kCount);
    std::vector<geo_cl20_t> output(kCount);
    std::vector<geo_real_t> scalar_output(kCount);
    uint32_t random_state = UINT32_C(0x6d2b79f5);

    for (std::size_t index = 0u; index < kCount; ++index) {
        a[index] = random_mv(random_state);
        b[index] = random_mv(random_state);
        rotor[index] = rotor_for(index);
    }

    status = geo_cuda_cl20_add_batch(
        context,
        a.data(),
        b.data(),
        output.data(),
        kCount
    );
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("addition", status, context);
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        if (!mv_near(output[index], geo_cl20_add(a[index], b[index]))) {
            return fail_index("addition", index, context);
        }
    }

    status = geo_cuda_cl20_product_batch(
        context,
        a.data(),
        b.data(),
        output.data(),
        kCount
    );
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("product", status, context);
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        if (!mv_near(output[index], geo_cl20_mul(a[index], b[index]))) {
            return fail_index("product", index, context);
        }
    }

    status = geo_cuda_cl20_reverse_batch(
        context,
        a.data(),
        output.data(),
        kCount
    );
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("reverse", status, context);
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        if (!mv_near(output[index], geo_cl20_reverse(a[index]))) {
            return fail_index("reverse", index, context);
        }
    }

    status = geo_cuda_cl20_vector_dot_batch(
        context,
        a.data(),
        b.data(),
        scalar_output.data(),
        kCount
    );
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("vector dot", status, context);
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        if (!scalar_near(
                scalar_output[index],
                geo_cl20_vector_dot(a[index], b[index]))) {
            return fail_index("vector dot", index, context);
        }
    }

    status = geo_cuda_cl20_vector_wedge_batch(
        context,
        a.data(),
        b.data(),
        scalar_output.data(),
        kCount
    );
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("vector wedge", status, context);
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        if (!scalar_near(
                scalar_output[index],
                geo_cl20_vector_wedge(a[index], b[index]))) {
            return fail_index("vector wedge", index, context);
        }
    }

    status = geo_cuda_cl20_rotor_action_batch(
        context,
        rotor.data(),
        a.data(),
        output.data(),
        kCount
    );
    if (status != GEO_CUDA_SUCCESS) {
        return fail_status("rotor action", status, context);
    }
    for (std::size_t index = 0u; index < kCount; ++index) {
        const geo_cl20_t expected = geo_cl20_mul(
            geo_cl20_mul(rotor[index], a[index]),
            geo_cl20_reverse(rotor[index])
        );
        if (!mv_near(output[index], expected)) {
            return fail_index("rotor action", index, context);
        }
    }

    if (geo_cuda_cl20_add_batch(
            context,
            nullptr,
            nullptr,
            nullptr,
            0u) != GEO_CUDA_SUCCESS) {
        std::fputs("FAIL: zero-length batch must be accepted\n", stderr);
        geo_cuda_context_destroy(context);
        return EXIT_FAILURE;
    }
    if (geo_cuda_cl20_add_batch(
            context,
            nullptr,
            b.data(),
            output.data(),
            1u) != GEO_CUDA_INVALID_ARGUMENT) {
        std::fputs("FAIL: non-empty null input must be rejected\n", stderr);
        geo_cuda_context_destroy(context);
        return EXIT_FAILURE;
    }
    if (geo_cuda_cl20_add_batch(
            context,
            a.data(),
            b.data(),
            output.data(),
            std::numeric_limits<std::size_t>::max()) !=
        GEO_CUDA_INVALID_ARGUMENT) {
        std::fputs("FAIL: overflowing batch dimensions must be rejected\n", stderr);
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
