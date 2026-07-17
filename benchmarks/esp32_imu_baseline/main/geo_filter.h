#ifndef GEO_ESP32_GEO_FILTER_H
#define GEO_ESP32_GEO_FILTER_H

#include "benchmark_common.h"
#include "geo/fixed.h"

#include <stdint.h>

typedef struct {
    float qw;
    float qx;
    float qy;
    float qz;
    float integral_x;
    float integral_y;
    float integral_z;
} geo_float_filter_t;

typedef struct {
    geo_fixed_t qw;
    geo_fixed_t qx;
    geo_fixed_t qy;
    geo_fixed_t qz;
    geo_fixed_t integral_x;
    geo_fixed_t integral_y;
    geo_fixed_t integral_z;
    uint32_t arithmetic_failures;
} geo_fixed_filter_t;

void geo_float_filter_reset(void *state);
benchmark_output_t geo_float_filter_step(void *state,
                                         const imu_sample_t *sample,
                                         float dt);

void geo_fixed_filter_reset(void *state);
benchmark_output_t geo_fixed_filter_step(void *state,
                                         const imu_sample_t *sample,
                                         float dt);

int geo_filter_self_test(void);

#endif
