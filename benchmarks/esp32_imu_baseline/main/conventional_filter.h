#ifndef GEO_ESP32_CONVENTIONAL_FILTER_H
#define GEO_ESP32_CONVENTIONAL_FILTER_H

#include "benchmark_common.h"

typedef struct {
    float qw;
    float qx;
    float qy;
    float qz;
    float integral_x;
    float integral_y;
    float integral_z;
} conventional_filter_t;

void conventional_filter_reset(void *state);
benchmark_output_t conventional_filter_step(void *state,
                                            const imu_sample_t *sample,
                                            float dt);

#endif
