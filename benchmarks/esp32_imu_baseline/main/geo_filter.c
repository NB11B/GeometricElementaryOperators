#include "geo_filter.h"

#include "geo/cl20.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/*
 * A quaternion is embedded in M2(C), with each real 2x2 matrix represented by
 * one Cl(2,0) value.  For q = w + xi + yj + zk:
 *
 *   q -> A + iB
 *   A = w + y e12
 *   B = x e1 + z e2
 *
 * Complex matrix multiplication then becomes four GEO Cl(2,0) products:
 *
 *   (A+iB)(C+iD) = (AC-BD) + i(AD+BC).
 *
 * This gives an exact quaternion product while using the existing GEO kernel,
 * rather than duplicating scalar quaternion multiplication in the A/B paths.
 */
typedef struct {
    geo_cl20_t real;
    geo_cl20_t imaginary;
} geo_float_quaternion_pair_t;

typedef struct {
    geo_fixed_cl20_t real;
    geo_fixed_cl20_t imaginary;
} geo_fixed_quaternion_pair_t;

static geo_float_quaternion_pair_t geo_float_quaternion_encode(
    float w,
    float x,
    float y,
    float z
)
{
    geo_float_quaternion_pair_t encoded;
    encoded.real = geo_cl20_make(
        (geo_real_t)w,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)y
    );
    encoded.imaginary = geo_cl20_make(
        (geo_real_t)0,
        (geo_real_t)x,
        (geo_real_t)z,
        (geo_real_t)0
    );
    return encoded;
}

static void geo_float_quaternion_decode(
    geo_float_quaternion_pair_t encoded,
    float *w,
    float *x,
    float *y,
    float *z
)
{
    *w = (float)encoded.real.scalar;
    *x = (float)encoded.imaginary.e1;
    *y = (float)encoded.real.e12;
    *z = (float)encoded.imaginary.e2;
}

static geo_float_quaternion_pair_t geo_float_quaternion_pair_mul(
    geo_float_quaternion_pair_t left,
    geo_float_quaternion_pair_t right
)
{
    const geo_cl20_t ac = geo_cl20_mul(left.real, right.real);
    const geo_cl20_t bd = geo_cl20_mul(left.imaginary, right.imaginary);
    const geo_cl20_t ad = geo_cl20_mul(left.real, right.imaginary);
    const geo_cl20_t bc = geo_cl20_mul(left.imaginary, right.real);
    geo_float_quaternion_pair_t result;

    result.real = geo_cl20_sub(ac, bd);
    result.imaginary = geo_cl20_add(ad, bc);
    return result;
}

static void geo_float_quaternion_mul(
    float aw,
    float ax,
    float ay,
    float az,
    float bw,
    float bx,
    float by,
    float bz,
    float *ow,
    float *ox,
    float *oy,
    float *oz
)
{
    const geo_float_quaternion_pair_t left =
        geo_float_quaternion_encode(aw, ax, ay, az);
    const geo_float_quaternion_pair_t right =
        geo_float_quaternion_encode(bw, bx, by, bz);
    const geo_float_quaternion_pair_t product =
        geo_float_quaternion_pair_mul(left, right);

    geo_float_quaternion_decode(product, ow, ox, oy, oz);
}

static void geo_float_world_to_body_vector(
    float qw,
    float qx,
    float qy,
    float qz,
    float vx,
    float vy,
    float vz,
    float *ox,
    float *oy,
    float *oz
)
{
    float tw;
    float tx;
    float ty;
    float tz;
    float rw;

    geo_float_quaternion_mul(
        qw, -qx, -qy, -qz,
        0.0f, vx, vy, vz,
        &tw, &tx, &ty, &tz
    );
    geo_float_quaternion_mul(
        tw, tx, ty, tz,
        qw, qx, qy, qz,
        &rw, ox, oy, oz
    );
    (void)rw;
}

static benchmark_output_t geo_float_output(
    const geo_float_filter_t *state,
    float confidence,
    float wx,
    float wy,
    float wz
)
{
    benchmark_output_t output;

    output.qw = state->qw;
    output.qx = state->qx;
    output.qy = state->qy;
    output.qz = state->qz;
    geo_float_world_to_body_vector(
        state->qw,
        state->qx,
        state->qy,
        state->qz,
        0.0f,
        0.0f,
        1.0f,
        &output.gravity_x,
        &output.gravity_y,
        &output.gravity_z
    );
    output.confidence = confidence;
    output.motion_state =
        (fabsf(wx) + fabsf(wy) + fabsf(wz) < 0.08f) ? 0U : 1U;
    return output;
}

void geo_float_filter_reset(void *opaque)
{
    geo_float_filter_t *state = (geo_float_filter_t *)opaque;
    memset(state, 0, sizeof(*state));
    state->qw = 1.0f;
}

benchmark_output_t geo_float_filter_step(
    void *opaque,
    const imu_sample_t *sample,
    float dt
)
{
    geo_float_filter_t *state = (geo_float_filter_t *)opaque;
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

        geo_float_world_to_body_vector(
            state->qw,
            state->qx,
            state->qy,
            state->qz,
            0.0f,
            0.0f,
            1.0f,
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
        float dw;
        float dx;
        float dy;
        float dz;
        const float half_dt = 0.5f * dt;

        geo_float_quaternion_mul(
            state->qw,
            state->qx,
            state->qy,
            state->qz,
            0.0f,
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

    return geo_float_output(state, confidence, wx, wy, wz);
}

static int geo_fixed_from_float(float value, geo_fixed_t *output)
{
    return geo_fixed_from_double((double)value, output) == GEO_FIXED_OK;
}

static int geo_fixed_checked_add(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
)
{
    const int64_t value = (int64_t)left + (int64_t)right;
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) {
        return 0;
    }
    *output = (geo_fixed_t)value;
    return 1;
}

static int geo_fixed_checked_sub(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
)
{
    const int64_t value = (int64_t)left - (int64_t)right;
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) {
        return 0;
    }
    *output = (geo_fixed_t)value;
    return 1;
}

static int geo_fixed_cl20_add_checked(
    geo_fixed_cl20_t left,
    geo_fixed_cl20_t right,
    geo_fixed_cl20_t *output
)
{
    geo_fixed_cl20_t result;
    if (output == NULL ||
        !geo_fixed_checked_add(left.scalar, right.scalar, &result.scalar) ||
        !geo_fixed_checked_add(left.e1, right.e1, &result.e1) ||
        !geo_fixed_checked_add(left.e2, right.e2, &result.e2) ||
        !geo_fixed_checked_add(left.e12, right.e12, &result.e12)) {
        return 0;
    }
    *output = result;
    return 1;
}

static int geo_fixed_cl20_sub_checked(
    geo_fixed_cl20_t left,
    geo_fixed_cl20_t right,
    geo_fixed_cl20_t *output
)
{
    geo_fixed_cl20_t result;
    if (output == NULL ||
        !geo_fixed_checked_sub(left.scalar, right.scalar, &result.scalar) ||
        !geo_fixed_checked_sub(left.e1, right.e1, &result.e1) ||
        !geo_fixed_checked_sub(left.e2, right.e2, &result.e2) ||
        !geo_fixed_checked_sub(left.e12, right.e12, &result.e12)) {
        return 0;
    }
    *output = result;
    return 1;
}

static geo_fixed_quaternion_pair_t geo_fixed_quaternion_encode(
    geo_fixed_t w,
    geo_fixed_t x,
    geo_fixed_t y,
    geo_fixed_t z
)
{
    geo_fixed_quaternion_pair_t encoded;
    encoded.real = (geo_fixed_cl20_t){w, 0, 0, y};
    encoded.imaginary = (geo_fixed_cl20_t){0, x, z, 0};
    return encoded;
}

static void geo_fixed_quaternion_decode(
    geo_fixed_quaternion_pair_t encoded,
    geo_fixed_t *w,
    geo_fixed_t *x,
    geo_fixed_t *y,
    geo_fixed_t *z
)
{
    *w = encoded.real.scalar;
    *x = encoded.imaginary.e1;
    *y = encoded.real.e12;
    *z = encoded.imaginary.e2;
}

static int geo_fixed_quaternion_pair_mul(
    geo_fixed_quaternion_pair_t left,
    geo_fixed_quaternion_pair_t right,
    geo_fixed_quaternion_pair_t *output
)
{
    geo_fixed_cl20_t ac;
    geo_fixed_cl20_t bd;
    geo_fixed_cl20_t ad;
    geo_fixed_cl20_t bc;
    geo_fixed_quaternion_pair_t result;

    if (output == NULL ||
        geo_fixed_cl20_mul(left.real, right.real, &ac) != GEO_FIXED_OK ||
        geo_fixed_cl20_mul(left.imaginary, right.imaginary, &bd) != GEO_FIXED_OK ||
        geo_fixed_cl20_mul(left.real, right.imaginary, &ad) != GEO_FIXED_OK ||
        geo_fixed_cl20_mul(left.imaginary, right.real, &bc) != GEO_FIXED_OK ||
        !geo_fixed_cl20_sub_checked(ac, bd, &result.real) ||
        !geo_fixed_cl20_add_checked(ad, bc, &result.imaginary)) {
        return 0;
    }

    *output = result;
    return 1;
}

static int geo_fixed_quaternion_mul(
    geo_fixed_t aw,
    geo_fixed_t ax,
    geo_fixed_t ay,
    geo_fixed_t az,
    geo_fixed_t bw,
    geo_fixed_t bx,
    geo_fixed_t by,
    geo_fixed_t bz,
    geo_fixed_t *ow,
    geo_fixed_t *ox,
    geo_fixed_t *oy,
    geo_fixed_t *oz
)
{
    geo_fixed_quaternion_pair_t product;
    if (!geo_fixed_quaternion_pair_mul(
            geo_fixed_quaternion_encode(aw, ax, ay, az),
            geo_fixed_quaternion_encode(bw, bx, by, bz),
            &product)) {
        return 0;
    }
    geo_fixed_quaternion_decode(product, ow, ox, oy, oz);
    return 1;
}

static int geo_fixed_world_to_body_vector(
    geo_fixed_t qw,
    geo_fixed_t qx,
    geo_fixed_t qy,
    geo_fixed_t qz,
    geo_fixed_t vx,
    geo_fixed_t vy,
    geo_fixed_t vz,
    geo_fixed_t *ox,
    geo_fixed_t *oy,
    geo_fixed_t *oz
)
{
    geo_fixed_t tw;
    geo_fixed_t tx;
    geo_fixed_t ty;
    geo_fixed_t tz;
    geo_fixed_t rw;

    if (qx == INT32_MIN || qy == INT32_MIN || qz == INT32_MIN) {
        return 0;
    }
    if (!geo_fixed_quaternion_mul(
            qw, -qx, -qy, -qz,
            0, vx, vy, vz,
            &tw, &tx, &ty, &tz) ||
        !geo_fixed_quaternion_mul(
            tw, tx, ty, tz,
            qw, qx, qy, qz,
            &rw, ox, oy, oz)) {
        return 0;
    }
    (void)rw;
    return 1;
}

static int geo_fixed_mul_checked(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
)
{
    return geo_fixed_mul(left, right, output) == GEO_FIXED_OK;
}

static int geo_fixed_mul3_checked(
    geo_fixed_t first,
    geo_fixed_t second,
    geo_fixed_t third,
    geo_fixed_t *output
)
{
    geo_fixed_t intermediate;
    return geo_fixed_mul_checked(first, second, &intermediate) &&
        geo_fixed_mul_checked(intermediate, third, output);
}

static int geo_fixed_cross_component(
    geo_fixed_t a0,
    geo_fixed_t b0,
    geo_fixed_t a1,
    geo_fixed_t b1,
    geo_fixed_t *output
)
{
    geo_fixed_t first;
    geo_fixed_t second;
    return geo_fixed_mul_checked(a0, b0, &first) &&
        geo_fixed_mul_checked(a1, b1, &second) &&
        geo_fixed_checked_sub(first, second, output);
}

static int geo_fixed_normalize_state(geo_fixed_filter_t *state)
{
    const double w = geo_fixed_to_double(state->qw);
    const double x = geo_fixed_to_double(state->qx);
    const double y = geo_fixed_to_double(state->qy);
    const double z = geo_fixed_to_double(state->qz);
    const double norm = sqrt(w * w + x * x + y * y + z * z);
    geo_fixed_t qw;
    geo_fixed_t qx;
    geo_fixed_t qy;
    geo_fixed_t qz;

    if (!isfinite(norm) || norm <= 1.0e-12 ||
        geo_fixed_from_double(w / norm, &qw) != GEO_FIXED_OK ||
        geo_fixed_from_double(x / norm, &qx) != GEO_FIXED_OK ||
        geo_fixed_from_double(y / norm, &qy) != GEO_FIXED_OK ||
        geo_fixed_from_double(z / norm, &qz) != GEO_FIXED_OK) {
        return 0;
    }

    state->qw = qw;
    state->qx = qx;
    state->qy = qy;
    state->qz = qz;
    return 1;
}

static benchmark_output_t geo_fixed_failure_output(geo_fixed_filter_t *state)
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

static int geo_fixed_make_output(
    const geo_fixed_filter_t *state,
    float confidence,
    geo_fixed_t wx,
    geo_fixed_t wy,
    geo_fixed_t wz,
    benchmark_output_t *output
)
{
    geo_fixed_t zero;
    geo_fixed_t one;
    geo_fixed_t gx;
    geo_fixed_t gy;
    geo_fixed_t gz;
    const float wx_float = (float)geo_fixed_to_double(wx);
    const float wy_float = (float)geo_fixed_to_double(wy);
    const float wz_float = (float)geo_fixed_to_double(wz);

    if (output == NULL ||
        !geo_fixed_from_float(0.0f, &zero) ||
        !geo_fixed_from_float(1.0f, &one) ||
        !geo_fixed_world_to_body_vector(
            state->qw,
            state->qx,
            state->qy,
            state->qz,
            zero,
            zero,
            one,
            &gx,
            &gy,
            &gz)) {
        return 0;
    }

    output->qw = (float)geo_fixed_to_double(state->qw);
    output->qx = (float)geo_fixed_to_double(state->qx);
    output->qy = (float)geo_fixed_to_double(state->qy);
    output->qz = (float)geo_fixed_to_double(state->qz);
    output->gravity_x = (float)geo_fixed_to_double(gx);
    output->gravity_y = (float)geo_fixed_to_double(gy);
    output->gravity_z = (float)geo_fixed_to_double(gz);
    output->confidence = confidence;
    output->motion_state =
        (fabsf(wx_float) + fabsf(wy_float) + fabsf(wz_float) < 0.08f)
        ? 0U
        : 1U;
    return 1;
}

void geo_fixed_filter_reset(void *opaque)
{
    geo_fixed_filter_t *state = (geo_fixed_filter_t *)opaque;
    memset(state, 0, sizeof(*state));
    if (geo_fixed_from_double(1.0, &state->qw) != GEO_FIXED_OK) {
        state->arithmetic_failures = 1U;
    }
}

benchmark_output_t geo_fixed_filter_step(
    void *opaque,
    const imu_sample_t *sample,
    float dt
)
{
    geo_fixed_filter_t *state = (geo_fixed_filter_t *)opaque;
    geo_fixed_filter_t next = *state;
    float ax_float = sample->ax;
    float ay_float = sample->ay;
    float az_float = sample->az;
    const float accel_norm = sqrtf(
        ax_float * ax_float + ay_float * ay_float + az_float * az_float
    );
    float confidence_float = 0.0f;
    geo_fixed_t wx;
    geo_fixed_t wy;
    geo_fixed_t wz;
    geo_fixed_t dt_fixed;
    geo_fixed_t half_dt;
    geo_fixed_t kp;
    geo_fixed_t ki;

    if (!geo_fixed_from_float(sample->gx, &wx) ||
        !geo_fixed_from_float(sample->gy, &wy) ||
        !geo_fixed_from_float(sample->gz, &wz) ||
        !geo_fixed_from_float(dt, &dt_fixed) ||
        !geo_fixed_from_float(0.5f * dt, &half_dt) ||
        !geo_fixed_from_float(2.2f, &kp) ||
        !geo_fixed_from_float(0.035f, &ki)) {
        return geo_fixed_failure_output(state);
    }

    if (accel_norm > 1.0e-6f) {
        geo_fixed_t ax;
        geo_fixed_t ay;
        geo_fixed_t az;
        geo_fixed_t confidence;
        geo_fixed_t zero;
        geo_fixed_t one;
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

        if (!geo_fixed_from_float(ax_float, &ax) ||
            !geo_fixed_from_float(ay_float, &ay) ||
            !geo_fixed_from_float(az_float, &az) ||
            !geo_fixed_from_float(confidence_float, &confidence) ||
            !geo_fixed_from_float(0.0f, &zero) ||
            !geo_fixed_from_float(1.0f, &one) ||
            !geo_fixed_world_to_body_vector(
                next.qw,
                next.qx,
                next.qy,
                next.qz,
                zero,
                zero,
                one,
                &vx,
                &vy,
                &vz) ||
            !geo_fixed_cross_component(ay, vz, az, vy, &ex) ||
            !geo_fixed_cross_component(az, vx, ax, vz, &ey) ||
            !geo_fixed_cross_component(ax, vy, ay, vx, &ez) ||
            !geo_fixed_mul_checked(confidence, ex, &corrected_x) ||
            !geo_fixed_mul_checked(confidence, ey, &corrected_y) ||
            !geo_fixed_mul_checked(confidence, ez, &corrected_z) ||
            !geo_fixed_mul_checked(kp, corrected_x, &proportional_x) ||
            !geo_fixed_mul_checked(kp, corrected_y, &proportional_y) ||
            !geo_fixed_mul_checked(kp, corrected_z, &proportional_z) ||
            !geo_fixed_mul3_checked(ki, corrected_x, dt_fixed, &integral_delta_x) ||
            !geo_fixed_mul3_checked(ki, corrected_y, dt_fixed, &integral_delta_y) ||
            !geo_fixed_mul3_checked(ki, corrected_z, dt_fixed, &integral_delta_z) ||
            !geo_fixed_checked_add(
                next.integral_x,
                integral_delta_x,
                &next.integral_x) ||
            !geo_fixed_checked_add(
                next.integral_y,
                integral_delta_y,
                &next.integral_y) ||
            !geo_fixed_checked_add(
                next.integral_z,
                integral_delta_z,
                &next.integral_z) ||
            !geo_fixed_checked_add(wx, proportional_x, &temporary) ||
            !geo_fixed_checked_add(temporary, next.integral_x, &wx) ||
            !geo_fixed_checked_add(wy, proportional_y, &temporary) ||
            !geo_fixed_checked_add(temporary, next.integral_y, &wy) ||
            !geo_fixed_checked_add(wz, proportional_z, &temporary) ||
            !geo_fixed_checked_add(temporary, next.integral_z, &wz)) {
            return geo_fixed_failure_output(state);
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

        if (!geo_fixed_quaternion_mul(
                next.qw,
                next.qx,
                next.qy,
                next.qz,
                0,
                wx,
                wy,
                wz,
                &dw,
                &dx,
                &dy,
                &dz) ||
            !geo_fixed_mul_checked(dw, half_dt, &delta_w) ||
            !geo_fixed_mul_checked(dx, half_dt, &delta_x) ||
            !geo_fixed_mul_checked(dy, half_dt, &delta_y) ||
            !geo_fixed_mul_checked(dz, half_dt, &delta_z) ||
            !geo_fixed_checked_add(next.qw, delta_w, &next.qw) ||
            !geo_fixed_checked_add(next.qx, delta_x, &next.qx) ||
            !geo_fixed_checked_add(next.qy, delta_y, &next.qy) ||
            !geo_fixed_checked_add(next.qz, delta_z, &next.qz) ||
            !geo_fixed_normalize_state(&next)) {
            return geo_fixed_failure_output(state);
        }
    }

    {
        benchmark_output_t output;
        next.arithmetic_failures = state->arithmetic_failures;
        if (!geo_fixed_make_output(
                &next,
                confidence_float,
                wx,
                wy,
                wz,
                &output)) {
            return geo_fixed_failure_output(state);
        }
        *state = next;
        return output;
    }
}
