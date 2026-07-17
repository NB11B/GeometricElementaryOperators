#ifndef GEO_ESP32_IMU_REPLAY_TRACE_H
#define GEO_ESP32_IMU_REPLAY_TRACE_H

#include <stdint.h>
#include "benchmark_common.h"

void replay_trace_sample(uint32_t index,
                         imu_sample_t *sample,
                         benchmark_output_t *reference);

#endif
