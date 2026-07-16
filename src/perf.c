#include "geo/perf.h"

#include <string.h>

#include "geo/geb36.h"

static volatile geo_real_t GEO_PERF_SINK = (geo_real_t)0;

static geo_status_t geo_benchmark_finish(
    const char *name,
    const geo_cycle_source_t *clock,
    uint64_t iterations,
    uint64_t start,
    uint64_t end,
    geo_benchmark_result_t *output
) {
    uint64_t elapsed;

    if (clock == NULL || output == NULL || name == NULL || iterations == 0u) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    if (end < start) {
        return GEO_STATUS_BAD_OPCODE;
    }

    elapsed = end - start;
    output->name = name;
    output->iterations = iterations;
    output->elapsed_ticks = elapsed;
    output->ticks_per_iteration = (double)elapsed / (double)iterations;
    output->nanoseconds_per_iteration = clock->frequency_hz == 0u
        ? 0.0
        : ((double)elapsed * 1000000000.0) /
          ((double)clock->frequency_hz * (double)iterations);
    return GEO_STATUS_OK;
}

void geo_memory_report(geo_memory_report_t *output) {
    if (output == NULL) {
        return;
    }

    output->sizeof_real = sizeof(geo_real_t);
    output->sizeof_cl20 = sizeof(geo_cl20_t);
    output->sizeof_opposite = sizeof(geo_opposite_t);
    output->sizeof_state = sizeof(geo_state_t);
    output->sizeof_geometric_register = sizeof(geo_geometric_register_t);
    output->sizeof_instruction = sizeof(geo_instruction_t);
    output->sizeof_banked_instruction = sizeof(geo_banked_instruction_t);
    output->sizeof_struct_value = sizeof(geo_struct_value_t);
}

geo_status_t geo_benchmark_cl20_product(
    const geo_cycle_source_t *clock,
    uint64_t iterations,
    geo_benchmark_result_t *output
) {
    geo_cl20_t a;
    geo_cl20_t b;
    geo_cl20_t c;
    uint64_t index;
    uint64_t start;
    uint64_t end;

    if (clock == NULL || clock->now == NULL || output == NULL || iterations == 0u) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    a = geo_cl20_make((geo_real_t)0.5, (geo_real_t)1.25, (geo_real_t)-0.75, (geo_real_t)0.2);
    b = geo_cl20_make((geo_real_t)-0.2, (geo_real_t)0.6, (geo_real_t)1.1, (geo_real_t)-0.4);
    c = geo_cl20_zero();

    start = clock->now(clock->context);
    for (index = 0u; index < iterations; ++index) {
        c = geo_cl20_mul(a, b);
        a.scalar += c.scalar * (geo_real_t)1e-7;
    }
    end = clock->now(clock->context);
    GEO_PERF_SINK = c.scalar + c.e1 + c.e2 + c.e12;

    return geo_benchmark_finish("cl20_product", clock, iterations, start, end, output);
}

geo_status_t geo_benchmark_direct_rotor_action(
    const geo_cycle_source_t *clock,
    uint64_t iterations,
    geo_benchmark_result_t *output
) {
    geo_cl20_t rotor;
    geo_cl20_t value;
    geo_cl20_t result;
    uint64_t index;
    uint64_t start;
    uint64_t end;

    if (clock == NULL || clock->now == NULL || output == NULL || iterations == 0u) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    rotor = geo_cl20_make((geo_real_t)0.9238795325, (geo_real_t)0, (geo_real_t)0, (geo_real_t)-0.3826834324);
    value = geo_cl20_make((geo_real_t)0, (geo_real_t)1, (geo_real_t)0, (geo_real_t)0);
    result = geo_cl20_zero();

    start = clock->now(clock->context);
    for (index = 0u; index < iterations; ++index) {
        result = geo_geb_rotor_action(rotor, value);
        value.e1 += result.e1 * (geo_real_t)1e-7;
    }
    end = clock->now(clock->context);
    GEO_PERF_SINK = result.scalar + result.e1 + result.e2 + result.e12;

    return geo_benchmark_finish("direct_rotor_action", clock, iterations, start, end, output);
}

geo_status_t geo_benchmark_banked_program(
    const geo_cycle_source_t *clock,
    const geo_banked_program_t *program,
    geo_banked_storage_t *storage,
    uint64_t iterations,
    geo_benchmark_result_t *output
) {
    uint64_t index;
    uint64_t start;
    uint64_t end;
    geo_status_t status;

    if (clock == NULL || clock->now == NULL || program == NULL ||
        storage == NULL || output == NULL || iterations == 0u) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    start = clock->now(clock->context);
    for (index = 0u; index < iterations; ++index) {
        status = geo_banked_execute(program, storage);
        if (status != GEO_STATUS_OK) {
            return status;
        }
    }
    end = clock->now(clock->context);

    return geo_benchmark_finish("banked_program", clock, iterations, start, end, output);
}

geo_status_t geo_benchmark_structured_program(
    const geo_cycle_source_t *clock,
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity,
    uint64_t iterations,
    geo_benchmark_result_t *output
) {
    uint64_t index;
    uint64_t start;
    uint64_t end;
    geo_status_t status;

    if (clock == NULL || clock->now == NULL || program == NULL ||
        registers == NULL || output == NULL || iterations == 0u) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    start = clock->now(clock->context);
    for (index = 0u; index < iterations; ++index) {
        status = geo_struct_program_execute(program, registers, register_capacity);
        if (status != GEO_STATUS_OK) {
            return status;
        }
    }
    end = clock->now(clock->context);

    return geo_benchmark_finish("structured_program", clock, iterations, start, end, output);
}
