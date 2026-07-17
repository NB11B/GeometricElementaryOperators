#include "geo_filter.h"

#include "conventional_filter.h"

#include <math.h>
#include <stddef.h>

static int output_is_finite(const benchmark_output_t *output)
{
    return output != NULL &&
        isfinite(output->qw) &&
        isfinite(output->qx) &&
        isfinite(output->qy) &&
        isfinite(output->qz) &&
        isfinite(output->gravity_x) &&
        isfinite(output->gravity_y) &&
        isfinite(output->gravity_z) &&
        isfinite(output->confidence);
}

int geo_filter_self_test(void)
{
    geo_float_filter_t geo_float_state;
    geo_fixed_filter_t geo_fixed_state;
    conventional_filter_t conventional_state;
    const imu_sample_t sample = {
        .ax = 0.01f,
        .ay = -0.02f,
        .az = BENCH_GRAVITY_MPS2,
        .gx = 0.10f,
        .gy = -0.20f,
        .gz = 0.05f,
    };
    const float dt = 1.0f / (float)BENCH_SAMPLE_RATE_HZ;
    benchmark_output_t geo_float_output = {0};
    benchmark_output_t geo_fixed_output = {0};
    benchmark_output_t conventional_output = {0};

    geo_float_filter_reset(&geo_float_state);
    geo_fixed_filter_reset(&geo_fixed_state);
    conventional_filter_reset(&conventional_state);

    for (size_t index = 0; index < 512U; ++index) {
        geo_float_output = geo_float_filter_step(
            &geo_float_state,
            &sample,
            dt
        );
        geo_fixed_output = geo_fixed_filter_step(
            &geo_fixed_state,
            &sample,
            dt
        );
        conventional_output = conventional_filter_step(
            &conventional_state,
            &sample,
            dt
        );

        if (!output_is_finite(&geo_float_output) ||
            !output_is_finite(&geo_fixed_output) ||
            !output_is_finite(&conventional_output)) {
            return 0;
        }
    }

    if (geo_fixed_state.arithmetic_failures != 0U) {
        return 0;
    }

    if (bench_quaternion_error_deg(
            &geo_float_output,
            &conventional_output) > 0.01f) {
        return 0;
    }

    if (bench_quaternion_error_deg(
            &geo_fixed_output,
            &conventional_output) > 1.0f) {
        return 0;
    }

    return 1;
}
