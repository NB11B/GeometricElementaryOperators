#include "conventional_filter.h"

#include <math.h>
#include <string.h>

void conventional_filter_reset(void *opaque)
{
    conventional_filter_t *state = (conventional_filter_t *)opaque;
    memset(state, 0, sizeof(*state));
    state->qw = 1.0f;
}

benchmark_output_t conventional_filter_step(void *opaque,
                                            const imu_sample_t *sample,
                                            float dt)
{
    conventional_filter_t *state = (conventional_filter_t *)opaque;
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
        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;
        confidence = bench_clampf(1.0f - fabsf(accel_norm - BENCH_GRAVITY_MPS2) / 2.0f, 0.0f, 1.0f);

        const float vx = 2.0f * (state->qx * state->qz - state->qw * state->qy);
        const float vy = 2.0f * (state->qw * state->qx + state->qy * state->qz);
        const float vz = state->qw * state->qw - state->qx * state->qx - state->qy * state->qy + state->qz * state->qz;
        const float ex = ay * vz - az * vy;
        const float ey = az * vx - ax * vz;
        const float ez = ax * vy - ay * vx;

        state->integral_x += ki * confidence * ex * dt;
        state->integral_y += ki * confidence * ey * dt;
        state->integral_z += ki * confidence * ez * dt;
        wx += kp * confidence * ex + state->integral_x;
        wy += kp * confidence * ey + state->integral_y;
        wz += kp * confidence * ez + state->integral_z;
    }

    const float half_dt = 0.5f * dt;
    const float qw = state->qw;
    const float qx = state->qx;
    const float qy = state->qy;
    const float qz = state->qz;
    state->qw += (-qx * wx - qy * wy - qz * wz) * half_dt;
    state->qx += ( qw * wx + qy * wz - qz * wy) * half_dt;
    state->qy += ( qw * wy - qx * wz + qz * wx) * half_dt;
    state->qz += ( qw * wz + qx * wy - qy * wx) * half_dt;
    bench_normalize_quaternion(&state->qw, &state->qx, &state->qy, &state->qz);

    benchmark_output_t output;
    output.qw = state->qw;
    output.qx = state->qx;
    output.qy = state->qy;
    output.qz = state->qz;
    output.gravity_x = 2.0f * (state->qx * state->qz - state->qw * state->qy);
    output.gravity_y = 2.0f * (state->qw * state->qx + state->qy * state->qz);
    output.gravity_z = state->qw * state->qw - state->qx * state->qx - state->qy * state->qy + state->qz * state->qz;
    output.confidence = confidence;
    output.motion_state = (fabsf(wx) + fabsf(wy) + fabsf(wz) < 0.08f) ? 0U : 1U;
    return output;
}
