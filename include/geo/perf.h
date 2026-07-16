#ifndef GEO_PERF_H
#define GEO_PERF_H

#include <stddef.h>
#include <stdint.h>

#include "geo/banked.h"
#include "geo/structured_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t (*geo_cycle_now_fn)(void *context);

typedef struct {
    geo_cycle_now_fn now;
    void *context;
    uint64_t frequency_hz;
} geo_cycle_source_t;

typedef struct {
    const char *name;
    uint64_t iterations;
    uint64_t elapsed_ticks;
    double ticks_per_iteration;
    double nanoseconds_per_iteration;
} geo_benchmark_result_t;

typedef struct {
    size_t sizeof_real;
    size_t sizeof_cl20;
    size_t sizeof_opposite;
    size_t sizeof_state;
    size_t sizeof_geometric_register;
    size_t sizeof_instruction;
    size_t sizeof_banked_instruction;
    size_t sizeof_struct_value;
} geo_memory_report_t;

void geo_memory_report(geo_memory_report_t *output);

geo_status_t geo_benchmark_cl20_product(
    const geo_cycle_source_t *clock,
    uint64_t iterations,
    geo_benchmark_result_t *output
);

geo_status_t geo_benchmark_direct_rotor_action(
    const geo_cycle_source_t *clock,
    uint64_t iterations,
    geo_benchmark_result_t *output
);

geo_status_t geo_benchmark_banked_program(
    const geo_cycle_source_t *clock,
    const geo_banked_program_t *program,
    geo_banked_storage_t *storage,
    uint64_t iterations,
    geo_benchmark_result_t *output
);

geo_status_t geo_benchmark_structured_program(
    const geo_cycle_source_t *clock,
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity,
    uint64_t iterations,
    geo_benchmark_result_t *output
);

#ifdef __cplusplus
}
#endif

#endif
