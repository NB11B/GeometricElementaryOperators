#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "geo/fused.h"
#include "geo/geb36.h"
#include "geo/structured_program.h"

#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#define ITERATIONS UINT64_C(1000000)

static volatile geo_real_t sink_value = (geo_real_t)0;

static uint64_t now_ticks(void) {
#if defined(_WIN32)
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return (uint64_t)value.QuadPart;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0u;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
#endif
}

static double tick_nanoseconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return 1.0e9 / (double)frequency.QuadPart;
#else
    return 1.0;
#endif
}

static void print_result(const char *name, uint64_t start, uint64_t end) {
    const double ns = ((double)(end - start) * tick_nanoseconds()) / (double)ITERATIONS;
    printf("%-28s %.3f ns/op\n", name, ns);
}

static void bench_addition(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)1, (geo_real_t)2, (geo_real_t)-3, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)-2, (geo_real_t)4, (geo_real_t)1, (geo_real_t)-0.25);
    geo_struct_value_t registers[6];
    const geo_struct_instruction_t structured_instructions[] = {
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 2u, 0u, 0u},
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 3u, 1u, 1u},
        {GEO_STRUCT_OP_UNIPOTENT_COMPOSE, 4u, 2u, 3u},
        {GEO_STRUCT_OP_UNIPOTENT_EXTRACT, 5u, 4u, 4u}
    };
    const geo_struct_program_t structured = {structured_instructions, 4u, 6u, 5u};
    geo_fused_instruction_t fused_instruction;
    geo_fused_program_t fused;
    uint64_t i;
    uint64_t start;
    uint64_t end;

    geo_fused_program_for_target(GEO_GEB_ADDITION, &fused_instruction, &fused);

    start = now_ticks();
    for (i = 0u; i < ITERATIONS; ++i) {
        const geo_cl20_t result = geo_geb_addition(a, b);
        sink_value += result.e1;
    }
    end = now_ticks();
    print_result("addition direct", start, end);

    start = now_ticks();
    for (i = 0u; i < ITERATIONS; ++i) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        geo_struct_program_execute(&structured, registers, 6u);
        sink_value += registers[5].as.cl20.e1;
    }
    end = now_ticks();
    print_result("addition structured", start, end);

    start = now_ticks();
    for (i = 0u; i < ITERATIONS; ++i) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        geo_fused_execute(&fused, registers, 6u);
        sink_value += registers[fused.root_register].as.cl20.e1;
    }
    end = now_ticks();
    print_result("addition fused", start, end);
}

static void bench_dot(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)0, (geo_real_t)2, (geo_real_t)-3, (geo_real_t)0);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)0, (geo_real_t)5, (geo_real_t)7, (geo_real_t)0);
    geo_struct_value_t registers[5];
    const geo_struct_instruction_t structured_instructions[] = {
        {GEO_STRUCT_OP_ORDERED_PRODUCTS, 2u, 0u, 1u},
        {GEO_STRUCT_OP_HADAMARD_EXACT, 3u, 2u, 2u},
        {GEO_STRUCT_OP_SELECT_SYMMETRIC, 4u, 3u, 3u}
    };
    const geo_struct_program_t structured = {structured_instructions, 3u, 5u, 4u};
    geo_fused_instruction_t fused_instruction;
    geo_fused_program_t fused;
    uint64_t i;
    uint64_t start;
    uint64_t end;

    geo_fused_program_for_target(GEO_GEB_VECTOR_DOT, &fused_instruction, &fused);

    start = now_ticks();
    for (i = 0u; i < ITERATIONS; ++i) sink_value += geo_geb_vector_dot(a, b);
    end = now_ticks();
    print_result("vector dot direct", start, end);

    start = now_ticks();
    for (i = 0u; i < ITERATIONS; ++i) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        geo_struct_program_execute(&structured, registers, 5u);
        sink_value += registers[4].as.cl20.scalar;
    }
    end = now_ticks();
    print_result("vector dot structured", start, end);

    start = now_ticks();
    for (i = 0u; i < ITERATIONS; ++i) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        geo_fused_execute(&fused, registers, 5u);
        sink_value += registers[fused.root_register].as.scalar;
    }
    end = now_ticks();
    print_result("vector dot fused", start, end);
}

int main(void) {
    puts("Geometric Elementary Operators comparison benchmark");
    printf("iterations: %" PRIu64 "\n", ITERATIONS);
    bench_addition();
    bench_dot();
    printf("sink: %.6f\n", (double)sink_value);
    return 0;
}
