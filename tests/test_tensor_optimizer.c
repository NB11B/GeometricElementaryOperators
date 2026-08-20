#include "geo/tensor_optimizer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOL ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOL ((geo_real_t)2e-6f)
#endif

static void assert_close(geo_real_t actual, geo_real_t expected) {
    const geo_real_t error = (geo_real_t)fabs((double)(actual - expected));
    assert(error <= GEO_TEST_TOL);
}

static void test_global_clip_scale(void) {
    const geo_real_t gradient_a[2] = {(geo_real_t)3, (geo_real_t)4};
    const geo_real_t gradient_b[2] = {(geo_real_t)0, (geo_real_t)0};
    geo_real_t sum_square = (geo_real_t)0;
    assert(geo_tensor_grad_square_accumulate(
        gradient_a, 2u, &sum_square
    ) == GEO_TENSOR_OK);
    assert(geo_tensor_grad_square_accumulate(
        gradient_b, 2u, &sum_square
    ) == GEO_TENSOR_OK);
    assert_close(sum_square, (geo_real_t)25);

    geo_real_t clip_scale;
    assert(geo_tensor_grad_clip_scale(
        sum_square, (geo_real_t)1, &clip_scale
    ) == GEO_TENSOR_OK);
    assert_close(
        clip_scale,
        (geo_real_t)1 / ((geo_real_t)5 + (geo_real_t)1e-6)
    );

    assert(geo_tensor_grad_clip_scale(
        sum_square, (geo_real_t)0, &clip_scale
    ) == GEO_TENSOR_OK);
    assert_close(clip_scale, (geo_real_t)1);
}

static void test_first_adamw_step(void) {
    geo_real_t parameter[3] = {
        (geo_real_t)1.0, (geo_real_t)-2.0, (geo_real_t)0.5
    };
    const geo_real_t original[3] = {
        parameter[0], parameter[1], parameter[2]
    };
    const geo_real_t gradient[3] = {
        (geo_real_t)0.2, (geo_real_t)-0.4, (geo_real_t)0.1
    };
    geo_real_t first_moment[3] = {0};
    geo_real_t second_moment[3] = {0};
    const geo_tensor_adamw_config config = {
        (geo_real_t)0.01,
        (geo_real_t)0.9,
        (geo_real_t)0.999,
        (geo_real_t)1e-8,
        (geo_real_t)0.1,
        (geo_real_t)0,
        1u
    };

    assert(geo_tensor_adamw_step(
        parameter, gradient, first_moment, second_moment,
        3u, (geo_real_t)1, config
    ) == GEO_TENSOR_OK);

    for (size_t index = 0u; index < 3u; ++index) {
        const geo_real_t expected_m = (geo_real_t)0.1 * gradient[index];
        const geo_real_t expected_v = (geo_real_t)0.001 * gradient[index] * gradient[index];
        const geo_real_t corrected_m = gradient[index];
        const geo_real_t corrected_v = gradient[index] * gradient[index];
        const geo_real_t expected_parameter =
            original[index] * ((geo_real_t)1 - (geo_real_t)0.001) -
            (geo_real_t)0.01 * corrected_m /
            ((geo_real_t)sqrt((double)corrected_v) + (geo_real_t)1e-8);
        assert_close(first_moment[index], expected_m);
        assert_close(second_moment[index], expected_v);
        assert_close(parameter[index], expected_parameter);
    }
}

static void test_clipped_update(void) {
    geo_real_t parameter[1] = {(geo_real_t)1};
    const geo_real_t gradient[1] = {(geo_real_t)10};
    geo_real_t first_moment[1] = {0};
    geo_real_t second_moment[1] = {0};
    const geo_tensor_adamw_config config = {
        (geo_real_t)0.1,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)1e-8,
        (geo_real_t)0,
        (geo_real_t)1,
        1u
    };
    assert(geo_tensor_adamw_step(
        parameter, gradient, first_moment, second_moment,
        1u, (geo_real_t)0.1, config
    ) == GEO_TENSOR_OK);
    assert_close(first_moment[0], (geo_real_t)1);
    assert_close(second_moment[0], (geo_real_t)1);
    assert_close(parameter[0], (geo_real_t)0.9);
}

int main(void) {
    test_global_clip_scale();
    test_first_adamw_step();
    test_clipped_update();
    puts("tensor_optimizer: ok");
    return 0;
}
