#ifndef GEO_ESP32_TINYML_BASELINE_H
#define GEO_ESP32_TINYML_BASELINE_H

#include <stdint.h>
#include "benchmark_common.h"

#define TINYML_WINDOW 32U
#define TINYML_CHANNELS 6U
#define TINYML_INPUTS (TINYML_WINDOW * TINYML_CHANNELS)
#define TINYML_HIDDEN 16U
#define TINYML_OUTPUTS 7U

typedef struct {
    int8_t ring[TINYML_INPUTS];
    int8_t input[TINYML_INPUTS];
    int8_t weights1[TINYML_INPUTS * TINYML_HIDDEN];
    int8_t weights2[TINYML_HIDDEN * TINYML_OUTPUTS];
    int32_t bias1[TINYML_HIDDEN];
    int32_t bias2[TINYML_OUTPUTS];
    uint32_t cursor;
    uint32_t filled;
    float qw;
    float qx;
    float qy;
    float qz;
} tinyml_baseline_t;

void tinyml_baseline_reset(void *state);
benchmark_output_t tinyml_baseline_step(void *state,
                                       const imu_sample_t *sample,
                                       float dt);

#endif
