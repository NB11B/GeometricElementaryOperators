#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#include "geo/perf.h"

static uint64_t host_now(void *context) {
    struct timespec value;
    (void)context;
    if (timespec_get(&value, TIME_UTC) != TIME_UTC) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static void print_result(const geo_benchmark_result_t *result) {
    printf(
        "%-28s iterations=%" PRIu64 " ticks=%" PRIu64
        " ticks/op=%.3f ns/op=%.3f\n",
        result->name,
        result->iterations,
        result->elapsed_ticks,
        result->ticks_per_iteration,
        result->nanoseconds_per_iteration
    );
}

int main(void) {
    const geo_cycle_source_t clock = {host_now, NULL, UINT64_C(1000000000)};
    geo_benchmark_result_t result;
    geo_memory_report_t memory;
    geo_status_t status;
    const uint64_t iterations = UINT64_C(1000000);

    geo_memory_report(&memory);
    puts("Geometric Elementary Operators host benchmark");
    printf("precision bytes             %zu\n", memory.sizeof_real);
    printf("geo_cl20_t bytes            %zu\n", memory.sizeof_cl20);
    printf("geo_opposite_t bytes        %zu\n", memory.sizeof_opposite);
    printf("geo_state_t bytes           %zu\n", memory.sizeof_state);
    printf("geometric bank bytes        %zu\n", memory.sizeof_geometric_register);
    printf("flat instruction bytes      %zu\n", memory.sizeof_instruction);
    printf("banked instruction bytes    %zu\n", memory.sizeof_banked_instruction);
    printf("structured value bytes      %zu\n\n", memory.sizeof_struct_value);

    status = geo_benchmark_cl20_product(&clock, iterations, &result);
    if (status != GEO_STATUS_OK) {
        fprintf(stderr, "cl20 benchmark failed: %d\n", (int)status);
        return 1;
    }
    print_result(&result);

    status = geo_benchmark_direct_rotor_action(&clock, iterations, &result);
    if (status != GEO_STATUS_OK) {
        fprintf(stderr, "rotor benchmark failed: %d\n", (int)status);
        return 1;
    }
    print_result(&result);

    return 0;
}
