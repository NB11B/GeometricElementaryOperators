#include "tinyml_baseline.h"

#include <math.h>
#include <string.h>

static int8_t quantize(float value, float scale)
{
    const float scaled = value * scale;
    if (scaled > 127.0f) {
        return 127;
    }
    if (scaled < -127.0f) {
        return -127;
    }
    return (int8_t)lrintf(scaled);
}

static void initialize_sparse_identity_network(tinyml_baseline_t *state)
{
    memset(state->weights1, 0, sizeof(state->weights1));
    memset(state->weights2, 0, sizeof(state->weights2));
    memset(state->bias1, 0, sizeof(state->bias1));
    memset(state->bias2, 0, sizeof(state->bias2));

    const uint32_t newest_base = (TINYML_WINDOW - 1U) * TINYML_CHANNELS;
    for (uint32_t channel = 0U; channel < TINYML_CHANNELS; ++channel) {
        const uint32_t positive = channel * 2U;
        const uint32_t negative = positive + 1U;
        state->weights1[positive * TINYML_INPUTS + newest_base + channel] = 1;
        state->weights1[negative * TINYML_INPUTS + newest_base + channel] = -1;
        state->weights2[channel * TINYML_HIDDEN + positive] = 1;
        state->weights2[channel * TINYML_HIDDEN + negative] = -1;
    }

    for (uint32_t i = 0U; i < 4U; ++i) {
        state->weights2[6U * TINYML_HIDDEN + (6U + i)] = 1;
    }
}

void tinyml_baseline_reset(void *opaque)
{
    tinyml_baseline_t *state = (tinyml_baseline_t *)opaque;
    memset(state, 0, sizeof(*state));
    state->qw = 1.0f;
    initialize_sparse_identity_network(state);
}

static void push_sample(tinyml_baseline_t *state, const imu_sample_t *sample)
{
    const uint32_t base = state->cursor * TINYML_CHANNELS;
    state->ring[base + 0U] = quantize(sample->ax, 12.0f);
    state->ring[base + 1U] = quantize(sample->ay, 12.0f);
    state->ring[base + 2U] = quantize(sample->az, 12.0f);
    state->ring[base + 3U] = quantize(sample->gx, 96.0f);
    state->ring[base + 4U] = quantize(sample->gy, 96.0f);
    state->ring[base + 5U] = quantize(sample->gz, 96.0f);
    state->cursor = (state->cursor + 1U) % TINYML_WINDOW;
    if (state->filled < TINYML_WINDOW) {
        ++state->filled;
    }

    for (uint32_t age = 0U; age < TINYML_WINDOW; ++age) {
        const uint32_t source_slot = (state->cursor + age) % TINYML_WINDOW;
        memcpy(&state->input[age * TINYML_CHANNELS],
               &state->ring[source_slot * TINYML_CHANNELS],
               TINYML_CHANNELS);
    }
}

benchmark_output_t tinyml_baseline_step(void *opaque,
                                       const imu_sample_t *sample,
                                       float dt)
{
    tinyml_baseline_t *state = (tinyml_baseline_t *)opaque;
    int32_t hidden[TINYML_HIDDEN];
    int32_t output_q[TINYML_OUTPUTS];
    push_sample(state, sample);

    for (uint32_t h = 0U; h < TINYML_HIDDEN; ++h) {
        int32_t acc = state->bias1[h];
        const int8_t *weights = &state->weights1[h * TINYML_INPUTS];
        for (uint32_t i = 0U; i < TINYML_INPUTS; ++i) {
            acc += (int32_t)state->input[i] * (int32_t)weights[i];
        }
        hidden[h] = acc > 0 ? acc : 0;
    }

    for (uint32_t o = 0U; o < TINYML_OUTPUTS; ++o) {
        int32_t acc = state->bias2[o];
        const int8_t *weights = &state->weights2[o * TINYML_HIDDEN];
        for (uint32_t h = 0U; h < TINYML_HIDDEN; ++h) {
            acc += hidden[h] * (int32_t)weights[h];
        }
        output_q[o] = acc;
    }

    float ax = (float)output_q[0] / 12.0f;
    float ay = (float)output_q[1] / 12.0f;
    float az = (float)output_q[2] / 12.0f;
    const float wx = (float)output_q[3] / 96.0f;
    const float wy = (float)output_q[4] / 96.0f;
    const float wz = (float)output_q[5] / 96.0f;
    const float accel_norm = sqrtf(ax * ax + ay * ay + az * az);
    float confidence = 0.0f;

    if (accel_norm > 1.0e-6f) {
        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;
        confidence = bench_clampf(1.0f - fabsf(accel_norm - BENCH_GRAVITY_MPS2) / 2.0f, 0.0f, 1.0f);
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
