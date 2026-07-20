#include "geo_filter.h"
#include "geo_filter_fused.h"
#include "geo_imu_generated_schedule.h"

#include "conventional_filter.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

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

static int generated_schedule_self_test(void)
{
    const float qw = 0.9f;
    const float qx = 0.1f;
    const float qy = -0.2f;
    const float qz = 0.3f;
    const float wx = 0.4f;
    const float wy = -0.5f;
    const float wz = 0.6f;
    float gx;
    float gy;
    float gz;
    float dw;
    float dx;
    float dy;
    float dz;

    geo_generated_float_gravity(qw, qx, qy, qz, &gx, &gy, &gz);
    if (fabsf(gx - 2.0f * (qx * qz - qw * qy)) > 1.0e-6f ||
        fabsf(gy - 2.0f * (qw * qx + qy * qz)) > 1.0e-6f ||
        fabsf(gz - (qw * qw - qx * qx - qy * qy + qz * qz)) > 1.0e-6f) {
        return 0;
    }

    geo_generated_float_q_times_vector_quaternion(
        qw,
        qx,
        qy,
        qz,
        wx,
        wy,
        wz,
        &dw,
        &dx,
        &dy,
        &dz
    );
    if (fabsf(dw - (-qx * wx - qy * wy - qz * wz)) > 1.0e-6f ||
        fabsf(dx - (qw * wx + qy * wz - qz * wy)) > 1.0e-6f ||
        fabsf(dy - (qw * wy - qx * wz + qz * wx)) > 1.0e-6f ||
        fabsf(dz - (qw * wz + qx * wy - qy * wx)) > 1.0e-6f) {
        return 0;
    }

    {
        const int32_t iqw = 58982;
        const int32_t iqx = 6554;
        const int32_t iqy = -13107;
        const int32_t iqz = 19661;
        const int32_t iwx = 26214;
        const int32_t iwy = -32768;
        const int32_t iwz = 39322;
        int64_t igx;
        int64_t igy;
        int64_t igz;
        int64_t idw;
        int64_t idx;
        int64_t idy;
        int64_t idz;

        geo_generated_q32_gravity(
            iqw,
            iqx,
            iqy,
            iqz,
            &igx,
            &igy,
            &igz
        );
        if (igx != INT64_C(2) * (
                (int64_t)iqx * iqz - (int64_t)iqw * iqy) ||
            igy != INT64_C(2) * (
                (int64_t)iqw * iqx + (int64_t)iqy * iqz) ||
            igz != (int64_t)iqw * iqw -
                (int64_t)iqx * iqx -
                (int64_t)iqy * iqy +
                (int64_t)iqz * iqz) {
            return 0;
        }

        geo_generated_q32_q_times_vector_quaternion(
            iqw,
            iqx,
            iqy,
            iqz,
            iwx,
            iwy,
            iwz,
            &idw,
            &idx,
            &idy,
            &idz
        );
        if (idw != -(int64_t)iqx * iwx -
                (int64_t)iqy * iwy -
                (int64_t)iqz * iwz ||
            idx != (int64_t)iqw * iwx +
                (int64_t)iqy * iwz -
                (int64_t)iqz * iwy ||
            idy != (int64_t)iqw * iwy -
                (int64_t)iqx * iwz +
                (int64_t)iqz * iwx ||
            idz != (int64_t)iqw * iwz +
                (int64_t)iqx * iwy -
                (int64_t)iqy * iwx) {
            return 0;
        }
    }

    return 1;
}

int geo_filter_self_test(void)
{
    geo_float_filter_t geo_float_state;
    geo_fixed_filter_t geo_fixed_state;
    geo_float_fused_filter_t geo_float_fused_state;
    geo_fixed_fused_filter_t geo_fixed_fused_state;
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
    benchmark_output_t geo_float_fused_output = {0};
    benchmark_output_t geo_fixed_fused_output = {0};
    benchmark_output_t conventional_output = {0};

    if (!generated_schedule_self_test()) {
        return 0;
    }

    geo_float_filter_reset(&geo_float_state);
    geo_fixed_filter_reset(&geo_fixed_state);
    geo_float_fused_filter_reset(&geo_float_fused_state);
    geo_fixed_fused_filter_reset(&geo_fixed_fused_state);
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
        geo_float_fused_output = geo_float_fused_filter_step(
            &geo_float_fused_state,
            &sample,
            dt
        );
        geo_fixed_fused_output = geo_fixed_fused_filter_step(
            &geo_fixed_fused_state,
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
            !output_is_finite(&geo_float_fused_output) ||
            !output_is_finite(&geo_fixed_fused_output) ||
            !output_is_finite(&conventional_output)) {
            return 0;
        }
    }

    if (geo_fixed_state.arithmetic_failures != 0U ||
        geo_fixed_fused_state.arithmetic_failures != 0U) {
        return 0;
    }

    if (bench_quaternion_error_deg(
            &geo_float_output,
            &conventional_output) > 0.01f) {
        return 0;
    }

    if (bench_quaternion_error_deg(
            &geo_float_fused_output,
            &conventional_output) > 0.01f) {
        return 0;
    }

    if (bench_quaternion_error_deg(
            &geo_fixed_output,
            &conventional_output) > 1.0f) {
        return 0;
    }

    if (bench_quaternion_error_deg(
            &geo_fixed_fused_output,
            &conventional_output) > 1.0f) {
        return 0;
    }

    return 1;
}
