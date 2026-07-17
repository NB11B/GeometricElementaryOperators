#include "replay_trace.h"

#include <math.h>

static float noise(uint32_t x)
{
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return ((float)(x & 0xffffU) / 32767.5f) - 1.0f;
}

static void euler_to_quaternion(float roll, float pitch, float yaw,
                                float *qw, float *qx, float *qy, float *qz)
{
    const float cr = cosf(0.5f * roll);
    const float sr = sinf(0.5f * roll);
    const float cp = cosf(0.5f * pitch);
    const float sp = sinf(0.5f * pitch);
    const float cy = cosf(0.5f * yaw);
    const float sy = sinf(0.5f * yaw);
    *qw = cr * cp * cy + sr * sp * sy;
    *qx = sr * cp * cy - cr * sp * sy;
    *qy = cr * sp * cy + sr * cp * sy;
    *qz = cr * cp * sy - sr * sp * cy;
}

void replay_trace_sample(uint32_t index,
                         imu_sample_t *sample,
                         benchmark_output_t *reference)
{
    const float t = (float)index / (float)BENCH_SAMPLE_RATE_HZ;
    const float roll = 0.45f * sinf(0.63f * t) + 0.10f * sinf(2.7f * t);
    const float pitch = 0.35f * sinf(0.41f * t + 0.7f);
    const float yaw = 0.55f * sinf(0.23f * t) + 0.012f * t;
    const float roll_rate = 0.45f * 0.63f * cosf(0.63f * t) + 0.10f * 2.7f * cosf(2.7f * t);
    const float pitch_rate = 0.35f * 0.41f * cosf(0.41f * t + 0.7f);
    const float yaw_rate = 0.55f * 0.23f * cosf(0.23f * t) + 0.012f;

    euler_to_quaternion(roll, pitch, yaw,
                        &reference->qw, &reference->qx,
                        &reference->qy, &reference->qz);

    const float qw = reference->qw;
    const float qx = reference->qx;
    const float qy = reference->qy;
    const float qz = reference->qz;
    const float gx_body = 2.0f * (qx * qz - qw * qy);
    const float gy_body = 2.0f * (qw * qx + qy * qz);
    const float gz_body = qw * qw - qx * qx - qy * qy + qz * qz;
    const float disturbance = ((index / 1200U) % 3U == 1U) ? 0.45f * sinf(7.0f * t) : 0.0f;

    sample->ax = BENCH_GRAVITY_MPS2 * gx_body + disturbance + 0.025f * noise(index * 17U + 1U);
    sample->ay = BENCH_GRAVITY_MPS2 * gy_body + 0.5f * disturbance + 0.025f * noise(index * 17U + 2U);
    sample->az = BENCH_GRAVITY_MPS2 * gz_body + 0.025f * noise(index * 17U + 3U);
    sample->gx = roll_rate + 0.0015f * noise(index * 17U + 4U);
    sample->gy = pitch_rate + 0.0015f * noise(index * 17U + 5U);
    sample->gz = yaw_rate + 0.0015f * noise(index * 17U + 6U);

    reference->gravity_x = gx_body;
    reference->gravity_y = gy_body;
    reference->gravity_z = gz_body;
    reference->confidence = disturbance == 0.0f ? 1.0f : 0.75f;
    reference->motion_state = (fabsf(roll_rate) + fabsf(pitch_rate) + fabsf(yaw_rate) < 0.08f) ? 0U : 1U;
}
