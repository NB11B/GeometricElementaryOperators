#ifndef GEO_ESP32_GEO_FILTER_FUSED_H
#define GEO_ESP32_GEO_FILTER_FUSED_H

#include "benchmark_common.h"
#include "geo_filter.h"

typedef geo_float_filter_t geo_float_fused_filter_t;
typedef geo_fixed_filter_t geo_fixed_fused_filter_t;

void geo_float_fused_filter_reset(void *state);
benchmark_output_t geo_float_fused_filter_step(void *state,
                                               const imu_sample_t *sample,
                                               float dt);

void geo_fixed_fused_filter_reset(void *state);
benchmark_output_t geo_fixed_fused_filter_step(void *state,
                                               const imu_sample_t *sample,
                                               float dt);

#endif
