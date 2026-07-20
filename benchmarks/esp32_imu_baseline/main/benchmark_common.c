#include "benchmark_common.h"
#include "replay_trace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"

static uint32_t latency_us[BENCH_SAMPLE_COUNT];
static float orientation_error_deg[BENCH_SAMPLE_COUNT];

float bench_clampf(float value, float lo, float hi)
{
    return value < lo ? lo : (value > hi ? hi : value);
}

void bench_normalize_quaternion(float *qw, float *qx, float *qy, float *qz)
{
    const float norm = sqrtf(
        (*qw) * (*qw) +
        (*qx) * (*qx) +
        (*qy) * (*qy) +
        (*qz) * (*qz)
    );
    if (norm > 1.0e-12f) {
        *qw /= norm;
        *qx /= norm;
        *qy /= norm;
        *qz /= norm;
    } else {
        *qw = 1.0f;
        *qx = 0.0f;
        *qy = 0.0f;
        *qz = 0.0f;
    }
}

float bench_quaternion_error_deg(
    const benchmark_output_t *a,
    const benchmark_output_t *b
)
{
    float dot;

    if (a == NULL || b == NULL ||
        !isfinite(a->qw) || !isfinite(a->qx) ||
        !isfinite(a->qy) || !isfinite(a->qz) ||
        !isfinite(b->qw) || !isfinite(b->qx) ||
        !isfinite(b->qy) || !isfinite(b->qz)) {
        return 180.0f;
    }

    dot = fabsf(
        a->qw * b->qw +
        a->qx * b->qx +
        a->qy * b->qy +
        a->qz * b->qz
    );
    dot = bench_clampf(dot, 0.0f, 1.0f);
    return 2.0f * acosf(dot) * 57.29577951308232f;
}

static uint32_t bench_hash_bytes(
    uint32_t hash,
    const void *data,
    size_t size
)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 16777619U;
    }
    return hash;
}

uint32_t bench_hash_output(uint32_t hash, const benchmark_output_t *output)
{
    if (output == NULL) {
        return bench_hash_bytes(hash, "null", 4U);
    }

    /* Hash named fields only. Hashing sizeof(struct) would include padding. */
    hash = bench_hash_bytes(hash, &output->qw, sizeof(output->qw));
    hash = bench_hash_bytes(hash, &output->qx, sizeof(output->qx));
    hash = bench_hash_bytes(hash, &output->qy, sizeof(output->qy));
    hash = bench_hash_bytes(hash, &output->qz, sizeof(output->qz));
    hash = bench_hash_bytes(
        hash,
        &output->gravity_x,
        sizeof(output->gravity_x)
    );
    hash = bench_hash_bytes(
        hash,
        &output->gravity_y,
        sizeof(output->gravity_y)
    );
    hash = bench_hash_bytes(
        hash,
        &output->gravity_z,
        sizeof(output->gravity_z)
    );
    hash = bench_hash_bytes(
        hash,
        &output->confidence,
        sizeof(output->confidence)
    );
    return bench_hash_bytes(
        hash,
        &output->motion_state,
        sizeof(output->motion_state)
    );
}

static int compare_u32(const void *a, const void *b)
{
    const uint32_t x = *(const uint32_t *)a;
    const uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int compare_float(const void *a, const void *b)
{
    const float x = *(const float *)a;
    const float y = *(const float *)b;
    return (x > y) - (x < y);
}

benchmark_result_t benchmark_run(
    const benchmark_impl_t *impl,
    uint32_t run_index
)
{
    benchmark_result_t result;
    const float dt = 1.0f / (float)BENCH_SAMPLE_RATE_HZ;
    double sum = 0.0;
    double sum_sq = 0.0;
    double error_sum = 0.0;
    uint32_t hash = 2166136261U;
    const uint32_t deadline_us = 1000000U / BENCH_SAMPLE_RATE_HZ;

    (void)run_index;
    memset(&result, 0, sizeof(result));
    impl->reset(impl->state);

    for (uint32_t i = 0; i < BENCH_WARMUP_SAMPLES; ++i) {
        imu_sample_t sample;
        benchmark_output_t reference;
        replay_trace_sample(i, &sample, &reference);
        (void)impl->step(impl->state, &sample, dt);
    }
    impl->reset(impl->state);

    for (uint32_t i = 0; i < BENCH_SAMPLE_COUNT; ++i) {
        imu_sample_t sample;
        benchmark_output_t reference;
        benchmark_output_t output;
        int64_t start;
        uint32_t elapsed;

        replay_trace_sample(i, &sample, &reference);
        start = esp_timer_get_time();
        output = impl->step(impl->state, &sample, dt);
        elapsed = (uint32_t)(esp_timer_get_time() - start);

        latency_us[i] = elapsed;
        orientation_error_deg[i] =
            bench_quaternion_error_deg(&output, &reference);
        sum += elapsed;
        sum_sq += (double)elapsed * elapsed;
        error_sum += orientation_error_deg[i];
        if (elapsed > deadline_us) {
            ++result.deadline_misses;
        }
        if (!isfinite(output.qw) || !isfinite(output.qx) ||
            !isfinite(output.qy) || !isfinite(output.qz)) {
            ++result.nan_count;
        }
        hash = bench_hash_output(hash, &output);
    }

    qsort(
        latency_us,
        BENCH_SAMPLE_COUNT,
        sizeof(latency_us[0]),
        compare_u32
    );
    qsort(
        orientation_error_deg,
        BENCH_SAMPLE_COUNT,
        sizeof(orientation_error_deg[0]),
        compare_float
    );

    result.mean_latency_us = sum / BENCH_SAMPLE_COUNT;
    result.stddev_latency_us = sqrt(fmax(
        0.0,
        sum_sq / BENCH_SAMPLE_COUNT -
            result.mean_latency_us * result.mean_latency_us
    ));
    result.min_latency_us = latency_us[0];
    result.p50_latency_us =
        latency_us[(BENCH_SAMPLE_COUNT * 50U) / 100U];
    result.p95_latency_us =
        latency_us[(BENCH_SAMPLE_COUNT * 95U) / 100U];
    result.p99_latency_us =
        latency_us[(BENCH_SAMPLE_COUNT * 99U) / 100U];
    result.max_latency_us = latency_us[BENCH_SAMPLE_COUNT - 1U];
    result.mean_orientation_error_deg = error_sum / BENCH_SAMPLE_COUNT;
    result.p95_orientation_error_deg =
        orientation_error_deg[(BENCH_SAMPLE_COUNT * 95U) / 100U];
    result.max_orientation_error_deg =
        orientation_error_deg[BENCH_SAMPLE_COUNT - 1U];
    result.output_hash = hash;
    return result;
}

void benchmark_print_csv_header(void)
{
    printf(
        "CSV,implementation,run,state_bytes,mean_us,stddev_us,min_us,p50_us,"
        "p95_us,p99_us,max_us,deadline_misses,mean_error_deg,p95_error_deg,"
        "max_error_deg,nan_count,output_hash,min_free_heap,largest_free_block\n"
    );
}

void benchmark_print_csv_row(
    const benchmark_impl_t *impl,
    uint32_t run_index,
    const benchmark_result_t *r
)
{
    printf(
        "CSV,%s,%lu,%u,%.6f,%.6f,%lu,%lu,%lu,%lu,%lu,%lu,%.6f,%.6f,"
        "%.6f,%lu,%08lx,%u,%u\n",
        impl->name,
        (unsigned long)run_index,
        (unsigned)impl->state_bytes,
        r->mean_latency_us,
        r->stddev_latency_us,
        (unsigned long)r->min_latency_us,
        (unsigned long)r->p50_latency_us,
        (unsigned long)r->p95_latency_us,
        (unsigned long)r->p99_latency_us,
        (unsigned long)r->max_latency_us,
        (unsigned long)r->deadline_misses,
        r->mean_orientation_error_deg,
        r->p95_orientation_error_deg,
        r->max_orientation_error_deg,
        (unsigned long)r->nan_count,
        (unsigned long)r->output_hash,
        (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT)
    );
}
