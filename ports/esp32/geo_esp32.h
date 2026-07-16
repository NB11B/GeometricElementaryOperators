#ifndef GEO_ESP32_H
#define GEO_ESP32_H

#include "geo/perf.h"

#ifdef __cplusplus
extern "C" {
#endif

geo_cycle_source_t geo_esp32_cycle_source(void);
void geo_esp32_print_memory_report(void);
void geo_esp32_run_smoke_benchmarks(uint64_t iterations);

#ifdef __cplusplus
}
#endif

#endif
