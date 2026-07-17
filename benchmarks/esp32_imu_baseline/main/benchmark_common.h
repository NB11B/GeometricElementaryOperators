#ifndef GEO_ESP32_IMU_BENCHMARK_COMMON_H
#define GEO_ESP32_IMU_BENCHMARK_COMMON_H

#include <stddef.h>
#include <stdint.h>

#define BENCH_SAMPLE_RATE_HZ 200U
#define BENCH_SAMPLE_COUNT 12000U
#define BENCH_WARMUP_SAMPLES 256U
#define BENCH_RUNS 30U
#define BENCH_GRAVITY_MPS2 9.80665f

typedef struct {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
} imu_sample_t;

typedef struct {
    float qw;
    float qx;
    float qy;
    float qz;
    float gravity_x;
    float gravity_y;
    float gravity_z;
    float confidence;
    uint8_t motion_state;
} benchmark_output_t;

typedef struct {
    const char *name;
    void *state;
    void (*reset)(void *state);
    benchmark_output_t (*step)(void *state, const imu_sample_t *sample, float dt);
    size_t state_bytes;
} benchmark_impl_t;

typedef struct {
    double mean_latency_us;
    uint32_t min_latency_us;
    uint32_t p50_latency_us;
    uint32_t p95_latency_us;
    uint32_t p99_latency_us;
    uint32_t max_latency_us;
    double stddev_latency_us;
    double mean_orientation_error_deg;
    double p95_orientation_error_deg;
    double max_orientation_error_deg;
    uint32_t deadline_misses;
    uint32_t nan_count;
    uint32_t output_hash;
} benchmark_result_t;

float bench_clampf(float value, float lo, float hi);
void bench_normalize_quaternion(float *qw, float *qx, float *qy, float *qz);
float bench_quaternion_error_deg(const benchmark_output_t *actual,
                                 const benchmark_output_t *reference);
uint32_t bench_hash_output(uint32_t hash, const benchmark_output_t *output);
benchmark_result_t benchmark_run(const benchmark_impl_t *impl, uint32_t run_index);
void benchmark_print_csv_header(void);
void benchmark_print_csv_row(const benchmark_impl_t *impl,
                             uint32_t run_index,
                             const benchmark_result_t *result);

#endif
