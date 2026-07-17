#include "geo_filter_fused.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#if GEO_FIXED_FRACTION_BITS != 16
#error "B1 fused replay schedule currently requires GEO_FIXED_FRACTION_BITS == 16"
#endif

/*
 * A1/B1 are sparse schedules lowered from the paired-Cl(2,0) quaternion
 * representation used by A0/B0.
 *
 * q = w + x i + y j + z k
 * A = w + y e12
 * B = x e1 + z e2
 * q -> A + iB
 *
 * (A+iB)(C+iD) = (AC-BD) + i(AD+BC)
 *
 * The generic A0/B0 path evaluates four complete Cl(2,0) products per
 * quaternion product.  This file emits only the nonzero products surviving
 * the known sparse grades.  It is the manually frozen target schedule for the
 * compiler/lowering stage, not a change in filter policy or task semantics.
 */

#define Q16_SCALE_I64 INT64_C(65536)
#define Q16_HALF_SCALE_I64 INT64_C(32768)
#define Q16_SCALE_F 65536.0f
#define Q16_INV_SCALE_F (1.0f / Q16_SCALE_F)
#define Q16_DT ((geo_fixed_t)328)
#define Q16_HALF_DT ((geo_fixed_t)164)
#define Q16_KP ((geo_fixed_t)144179)
#define Q16_KI ((geo_fixed_t)2294)
#define Q16_ZERO ((geo_fixed_t)0)
#define Q16_ONE ((geo_fixed_t)65536)

static inline void fused_float_gravity(
    float qw,
    float qx,
    float qy,
    float qz,
    float *gx,
    float *gy,
    float *gz
)
{
    *gx = 2.0f * (qx * qz - qw * qy);
    *gy = 2.0f * (qw * qx + qy * qz);
    *gz = qw * qw - qx * qx - qy * qy + qz * qz;
}

static inline void fused_float_q_times_vector_quaternion(
    float qw,
    float qx,
    float qy,
    float qz,
    float wx,
    float wy,
    float wz,
    float *dw,
    float *dx,
    float *dy,
    float *dz
)
{
    /* Sparse lowering of q * (0 + wx i + wy j + wz k). */
    *dw = -qx * wx - qy * wy - qz * wz;
    *dx = qw * wx + qy * wz - qz * wy;
    *dy = qw * wy - qx * wz + qz * wx;
    *dz = qw * wz + qx * wy - qy * wx;
}

void geo_float_fused_filter_reset(void *opaque)
{
    geo_float_fused_filter_t *state =
        (geo_float_fused_filter_t *)opaque;
    memset(state, 0, sizeof(*state));
    state->qw = 1.0f;
}

benchmark_output_t geo_float_fused_filter_step(
    void *opaque,
    const imu_sample_t *sample,
    float dt
)
{
    geo_float_fused_filter_t *state =
        (geo_float_fused_filter_t *)opaque;
    const float kp = 2.2f;
    const float ki = 0.035f;
    float ax = sample->ax;
    float ay = sample->ay;
    float az = sample->az;
    float wx = sample->gx;
    float wy = sample->gy;
    float wz = sample->gz;
    const float accel_norm = sqrtf(ax * ax + ay * ay + az * az);
    float confidence = 0.0f;

    if (accel_norm > 1.0e-6f) {
        float vx;
        float vy;
        float vz;
        float ex;
        float ey;
        float ez;

        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;
        confidence = bench_clampf(
            1.0f - fabsf(accel_norm - BENCH_GRAVITY_MPS2) / 2.0f,
            0.0f,
            1.0f
        );

        fused_float_gravity(
            state->qw,
            state->qx,
            state->qy,
            state->qz,
            &vx,
            &vy,
            &vz
        );

        ex = ay * vz - az * vy;
        ey = az * vx - ax * vz;
        ez = ax * vy - ay * vx;

        state->integral_x += ki * confidence * ex * dt;
        state->integral_y += ki * confidence * ey * dt;
        state->integral_z += ki * confidence * ez * dt;
        wx += kp * confidence * ex + state->integral_x;
        wy += kp * confidence * ey + state->integral_y;
        wz += kp * confidence * ez + state->integral_z;
    }

    {
        const float half_dt = 0.5f * dt;
        const float qw = state->qw;
        const float qx = state->qx;
        const float qy = state->qy;
        const float qz = state->qz;
        float dw;
        float dx;
        float dy;
        float dz;

        fused_float_q_times_vector_quaternion(
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

        state->qw += dw * half_dt;
        state->qx += dx * half_dt;
        state->qy += dy * half_dt;
        state->qz += dz * half_dt;
        bench_normalize_quaternion(
            &state->qw,
            &state->qx,
            &state->qy,
            &state->qz
        );
    }

    {
        benchmark_output_t output;
        output.qw = state->qw;
        output.qx = state->qx;
        output.qy = state->qy;
        output.qz = state->qz;
        fused_float_gravity(
            state->qw,
            state->qx,
            state->qy,
            state->qz,
            &output.gravity_x,
            &output.gravity_y,
            &output.gravity_z
        );
        output.confidence = confidence;
        output.motion_state =
            (fabsf(wx) + fabsf(wy) + fabsf(wz) < 0.08f) ? 0U : 1U;
        return output;
    }
}

static inline int q16_in_range(int64_t value)
{
    return value >= INT32_MIN && value <= INT32_MAX;
}

static int q16_from_float_fast(float value, geo_fixed_t *output)
{
    float scaled;
    int64_t rounded;

    if (output == NULL || !isfinite(value)) {
        return 0;
    }

    scaled = value * Q16_SCALE_F;
    if (scaled > (float)INT32_MAX || scaled < (float)INT32_MIN) {
        return 0;
    }

    rounded = scaled >= 0.0f
        ? (int64_t)(scaled + 0.5f)
        : (int64_t)(scaled - 0.5f);
    if (!q16_in_range(rounded)) {
        return 0;
    }

    *output = (geo_fixed_t)rounded;
    return 1;
}

static inline float q16_to_float_fast(geo_fixed_t value)
{
    return (float)value * Q16_INV_SCALE_F;
}

static int q16_round_accumulator(int64_t accumulator, geo_fixed_t *output)
{
    int64_t quotient;
    int64_t remainder;
    uint64_t abs_remainder;

    if (output == NULL) {
        return 0;
    }

    quotient = accumulator / Q16_SCALE_I64;
    remainder = accumulator % Q16_SCALE_I64;
    abs_remainder = remainder < 0
        ? (uint64_t)(-remainder)
        : (uint64_t)remainder;

    if (abs_remainder >= (uint64_t)Q16_HALF_SCALE_I64) {
        quotient += accumulator < 0 ? -1 : 1;
    }

    if (!q16_in_range(quotient)) {
        return 0;
    }
    *output = (geo_fixed_t)quotient;
    return 1;
}

static int q16_add_checked(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
)
{
    const int64_t value = (int64_t)left + (int64_t)right;
    if (output == NULL || !q16_in_range(value)) {
        return 0;
    }
    *output = (geo_fixed_t)value;
    return 1;
}

static int q16_mul_checked(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
)
{
    return q16_round_accumulator((int64_t)left * (int64_t)right, output);
}

static int q16_mul3_checked(
    geo_fixed_t first,
    geo_fixed_t second,
    geo_fixed_t third,
    geo_fixed_t *output
)
{
    geo_fixed_t intermediate;
    return q16_mul_checked(first, second, &intermediate) &&
        q16_mul_checked(intermediate, third, output);
}

static int q16_cross_component(
    geo_fixed_t a0,
    geo_fixed_t b0,
    geo_fixed_t a1,
    geo_fixed_t b1,
    geo_fixed_t *output
)
{
    const int64_t accumulator =
        (int64_t)a0 * (int64_t)b0 -
        (int64_t)a1 * (int64_t)b1;
    return q16_round_accumulator(accumulator, output);
}

static int q16_fused_gravity(
    geo_fixed_t qw,
    geo_fixed_t qx,
    geo_fixed_t qy,
    geo_fixed_t qz,
    geo_fixed_t *gx,
    geo_fixed_t *gy,
    geo_fixed_t *gz
)
{
    const int64_t gx_accumulator = INT64_C(2) * (
        (int64_t)qx * (int64_t)qz -
        (int64_t)qw * (int64_t)qy
    );
    const int64_t gy_accumulator = INT64_C(2) * (
        (int64_t)qw * (int64_t)qx +
        (int64_t)qy * (int64_t)qz
    );
    const int64_t gz_accumulator =
        (int64_t)qw * (int64_t)qw -
        (int64_t)qx * (int64_t)qx -
        (int64_t)qy * (int64_t)qy +
        (int64_t)qz * (int64_t)qz;

    return q16_round_accumulator(gx_accumulator, gx) &&
        q16_round_accumulator(gy_accumulator, gy) &&
        q16_round_accumulator(gz_accumulator, gz);
}

static int q16_fused_q_times_vector_quaternion(
    geo_fixed_t qw,
    geo_fixed_t qx,
    geo_fixed_t qy,
    geo_fixed_t qz,
    geo_fixed_t wx,
    geo_fixed_t wy,
    geo_fixed_t wz,
    geo_fixed_t *dw,
    geo_fixed_t *dx,
    geo_fixed_t *dy,
    geo_fixed_t *dz
)
{
    const int64_t dw_accumulator =
        -(int64_t)qx * (int64_t)wx -
        (int64_t)qy * (int64_t)wy -
        (int64_t)qz * (int64_t)wz;
    const int64_t dx_accumulator =
        (int64_t)qw * (int64_t)wx +
        (int64_t)qy * (int64_t)wz -
        (int64_t)qz * (int64_t)wy;
    const int64_t dy_accumulator =
        (int64_t)qw * (int64_t)wy -
        (int64_t)qx * (int64_t)wz +
        (int64_t)qz * (int64_t)wx;
    const int64_t dz_accumulator =
        (int64_t)qw * (int64_t)wz +
        (int64_t)qx * (int64_t)wy -
        (int64_t)qy * (int64_t)wx;

    return q16_round_accumulator(dw_accumulator, dw) &&
        q16_round_accumulator(dx_accumulator, dx) &&
        q16_round_accumulator(dy_accumulator, dy) &&
        q16_round_accumulator(dz_accumulator, dz);
}

static int q16_normalize_state(geo_fixed_fused_filter_t *state)
{
    const float qw = q16_to_float_fast(state->qw);
    const float qx = q16_to_float_fast(state->qx);
    const float qy = q16_to_float_fast(state->qy);
    const float qz = q16_to_float_fast(state->qz);
    const float norm = sqrtf(
        qw * qw + qx * qx + qy * qy + qz * qz
    );
    geo_fixed_t normalized_qw;
    geo_fixed_t normalized_qx;
    geo_fixed_t normalized_qy;
    geo_fixed_t normalized_qz;

    if (!isfinite(norm) || norm <= 1.0e-12f ||
        !q16_from_float_fast(qw / norm, &normalized_qw) ||
        !q16_from_float_fast(qx / norm, &normalized_qx) ||
        !q16_from_float_fast(qy / norm, &normalized_qy) ||
        !q16_from_float_fast(qz / norm, &normalized_qz)) {
        return 0;
    }

    state->qw = normalized_qw;
    state->qx = normalized_qx;
    state->qy = normalized_qy;
    state->qz = normalized_qz;
    return 1;
}

static benchmark_output_t q16_failure_output(
    geo_fixed_fused_filter_t *state
)
{
    benchmark_output_t output;
    ++state->arithmetic_failures;
    output.qw = NAN;
    output.qx = NAN;
    output.qy = NAN;
    output.qz = NAN;
    output.gravity_x = NAN;
    output.gravity_y = NAN;
    output.gravity_z = NAN;
    output.confidence = 0.0f;
    output.motion_state = UINT8_MAX;
    return output;
}

void geo_fixed_fused_filter_reset(void *opaque)
{
    geo_fixed_fused_filter_t *state =
        (geo_fixed_fused_filter_t *)opaque;
    memset(state, 0, sizeof(*state));
    state->qw = Q16_ONE;
}

benchmark_output_t geo_fixed_fused_filter_step(
    void *opaque,
    const imu_sample_t *sample,
    float dt
)
{
    geo_fixed_fused_filter_t *state =
        (geo_fixed_fused_filter_t *)opaque;
    geo_fixed_fused_filter_t next = *state;
    float ax_float = sample->ax;
    float ay_float = sample->ay;
    float az_float = sample->az;
    const float accel_norm = sqrtf(
        ax_float * ax_float +
        ay_float * ay_float +
        az_float * az_float
    );
    float confidence_float = 0.0f;
    geo_fixed_t wx;
    geo_fixed_t wy;
    geo_fixed_t wz;

    (void)dt;

    if (!q16_from_float_fast(sample->gx, &wx) ||
        !q16_from_float_fast(sample->gy, &wy) ||
        !q16_from_float_fast(sample->gz, &wz)) {
        return q16_failure_output(state);
    }

    if (accel_norm > 1.0e-6f) {
        geo_fixed_t ax;
        geo_fixed_t ay;
        geo_fixed_t az;
        geo_fixed_t confidence;
        geo_fixed_t vx;
        geo_fixed_t vy;
        geo_fixed_t vz;
        geo_fixed_t ex;
        geo_fixed_t ey;
        geo_fixed_t ez;
        geo_fixed_t corrected_x;
        geo_fixed_t corrected_y;
        geo_fixed_t corrected_z;
        geo_fixed_t proportional_x;
        geo_fixed_t proportional_y;
        geo_fixed_t proportional_z;
        geo_fixed_t integral_delta_x;
        geo_fixed_t integral_delta_y;
        geo_fixed_t integral_delta_z;
        geo_fixed_t temporary;

        ax_float /= accel_norm;
        ay_float /= accel_norm;
        az_float /= accel_norm;
        confidence_float = bench_clampf(
            1.0f - fabsf(accel_norm - BENCH_GRAVITY_MPS2) / 2.0f,
            0.0f,
            1.0f
        );

        if (!q16_from_float_fast(ax_float, &ax) ||
            !q16_from_float_fast(ay_float, &ay) ||
            !q16_from_float_fast(az_float, &az) ||
            !q16_from_float_fast(confidence_float, &confidence) ||
            !q16_fused_gravity(
                next.qw,
                next.qx,
                next.qy,
                next.qz,
                &vx,
                &vy,
                &vz) ||
            !q16_cross_component(ay, vz, az, vy, &ex) ||
            !q16_cross_component(az, vx, ax, vz, &ey) ||
            !q16_cross_component(ax, vy, ay, vx, &ez) ||
            !q16_mul_checked(confidence, ex, &corrected_x) ||
            !q16_mul_checked(confidence, ey, &corrected_y) ||
            !q16_mul_checked(confidence, ez, &corrected_z) ||
            !q16_mul_checked(Q16_KP, corrected_x, &proportional_x) ||
            !q16_mul_checked(Q16_KP, corrected_y, &proportional_y) ||
            !q16_mul_checked(Q16_KP, corrected_z, &proportional_z) ||
            !q16_mul3_checked(
                Q16_KI,
                corrected_x,
                Q16_DT,
                &integral_delta_x) ||
            !q16_mul3_checked(
                Q16_KI,
                corrected_y,
                Q16_DT,
                &integral_delta_y) ||
            !q16_mul3_checked(
                Q16_KI,
                corrected_z,
                Q16_DT,
                &integral_delta_z) ||
            !q16_add_checked(
                next.integral_x,
                integral_delta_x,
                &next.integral_x) ||
            !q16_add_checked(
                next.integral_y,
                integral_delta_y,
                &next.integral_y) ||
            !q16_add_checked(
                next.integral_z,
                integral_delta_z,
                &next.integral_z) ||
            !q16_add_checked(wx, proportional_x, &temporary) ||
            !q16_add_checked(temporary, next.integral_x, &wx) ||
            !q16_add_checked(wy, proportional_y, &temporary) ||
            !q16_add_checked(temporary, next.integral_y, &wy) ||
            !q16_add_checked(wz, proportional_z, &temporary) ||
            !q16_add_checked(temporary, next.integral_z, &wz)) {
            return q16_failure_output(state);
        }
    }

    {
        geo_fixed_t dw;
        geo_fixed_t dx;
        geo_fixed_t dy;
        geo_fixed_t dz;
        geo_fixed_t delta_w;
        geo_fixed_t delta_x;
        geo_fixed_t delta_y;
        geo_fixed_t delta_z;

        if (!q16_fused_q_times_vector_quaternion(
                next.qw,
                next.qx,
                next.qy,
                next.qz,
                wx,
                wy,
                wz,
                &dw,
                &dx,
                &dy,
                &dz) ||
            !q16_mul_checked(dw, Q16_HALF_DT, &delta_w) ||
            !q16_mul_checked(dx, Q16_HALF_DT, &delta_x) ||
            !q16_mul_checked(dy, Q16_HALF_DT, &delta_y) ||
            !q16_mul_checked(dz, Q16_HALF_DT, &delta_z) ||
            !q16_add_checked(next.qw, delta_w, &next.qw) ||
            !q16_add_checked(next.qx, delta_x, &next.qx) ||
            !q16_add_checked(next.qy, delta_y, &next.qy) ||
            !q16_add_checked(next.qz, delta_z, &next.qz) ||
            !q16_normalize_state(&next)) {
            return q16_failure_output(state);
        }
    }

    {
        geo_fixed_t gx;
        geo_fixed_t gy;
        geo_fixed_t gz;
        benchmark_output_t output;
        const float wx_float = q16_to_float_fast(wx);
        const float wy_float = q16_to_float_fast(wy);
        const float wz_float = q16_to_float_fast(wz);

        if (!q16_fused_gravity(
                next.qw,
                next.qx,
                next.qy,
                next.qz,
                &gx,
                &gy,
                &gz)) {
            return q16_failure_output(state);
        }

        next.arithmetic_failures = state->arithmetic_failures;
        output.qw = q16_to_float_fast(next.qw);
        output.qx = q16_to_float_fast(next.qx);
        output.qy = q16_to_float_fast(next.qy);
        output.qz = q16_to_float_fast(next.qz);
        output.gravity_x = q16_to_float_fast(gx);
        output.gravity_y = q16_to_float_fast(gy);
        output.gravity_z = q16_to_float_fast(gz);
        output.confidence = confidence_float;
        output.motion_state =
            (fabsf(wx_float) + fabsf(wy_float) + fabsf(wz_float) < 0.08f)
            ? 0U
            : 1U;
        *state = next;
        return output;
    }
}
