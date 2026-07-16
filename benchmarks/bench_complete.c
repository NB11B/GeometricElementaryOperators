#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "geo/fixed_geb36.h"
#include "geo/fused.h"
#include "geo/geb36.h"
#include "geo/native_generated.h"
#include "geo/structured_program.h"

#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#define ITERATIONS UINT64_C(1000000)

static volatile geo_real_t sink_real = (geo_real_t)0;
static volatile uint32_t sink_fixed = UINT32_C(0);

static void consume_fixed(geo_fixed_t value) {
    /* Unsigned mixing keeps the anti-optimization sink fully defined. */
    sink_fixed ^= (uint32_t)value;
}

static uint64_t now_ticks(void) {
#if defined(_WIN32)
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return (uint64_t)value.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0u;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

static double tick_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return 1.0e9 / (double)frequency.QuadPart;
#else
    return 1.0;
#endif
}

static void report(const char *name, uint64_t begin, uint64_t end) {
    const double value = (double)(end - begin) * tick_ns() / (double)ITERATIONS;
    printf("%-40s %.3f ns/op\n", name, value);
}

static geo_fixed_cl20_t fixed_mv(double s, double e1, double e2, double e12) {
    geo_fixed_cl20_t value;
    (void)geo_fixed_from_double(s, &value.scalar);
    (void)geo_fixed_from_double(e1, &value.e1);
    (void)geo_fixed_from_double(e2, &value.e2);
    (void)geo_fixed_from_double(e12, &value.e12);
    return value;
}

static void benchmark_addition(void) {
    const geo_cl20_t a = geo_cl20_make(1, 2, -3, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make(-2, 4, 1, (geo_real_t)-0.25);
    const geo_fixed_cl20_t fa = fixed_mv(1, 2, -3, 0.5);
    const geo_fixed_cl20_t fb = fixed_mv(-2, 4, 1, -0.25);
    const geo_struct_instruction_t structured_code[] = {
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 2u, 0u, 0u},
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 3u, 1u, 1u},
        {GEO_STRUCT_OP_UNIPOTENT_COMPOSE, 4u, 2u, 3u},
        {GEO_STRUCT_OP_UNIPOTENT_EXTRACT, 5u, 4u, 4u}
    };
    const geo_struct_program_t structured = {structured_code, 4u, 6u, 5u};
    geo_fused_instruction_t fused_code;
    geo_fused_program_t fused;
    geo_struct_value_t registers[6];
    geo_fixed_geb_result_t fixed_result;
    uint64_t i, begin, end;
    (void)geo_fused_program_for_target(GEO_GEB_ADDITION, &fused_code, &fused);

    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) sink_real += geo_geb_addition(a, b).e1;
    end = now_ticks(); report("addition direct C", begin, end);

    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) sink_real += geo_native_add(a, b).e1;
    end = now_ticks(); report("addition specialized native C", begin, end);

    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) {
        registers[0] = geo_struct_value_from_cl20(a); registers[1] = geo_struct_value_from_cl20(b);
        (void)geo_fused_execute(&fused, registers, 6u); sink_real += registers[fused.root_register].as.cl20.e1;
    }
    end = now_ticks(); report("addition fused IR", begin, end);

    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) {
        registers[0] = geo_struct_value_from_cl20(a); registers[1] = geo_struct_value_from_cl20(b);
        (void)geo_struct_program_execute(&structured, registers, 6u); sink_real += registers[5].as.cl20.e1;
    }
    end = now_ticks(); report("addition structured IR", begin, end);

    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) {
        (void)geo_fixed_geb36_execute(GEO_GEB_ADDITION, fa, fb, fa, &fixed_result);
        consume_fixed(fixed_result.as.cl20.e1);
    }
    end = now_ticks(); report("addition fixed Q16.16", begin, end);
}

static void benchmark_dot(void) {
    const geo_cl20_t a = geo_cl20_make(0, 2, -3, 0);
    const geo_cl20_t b = geo_cl20_make(0, 5, 7, 0);
    const geo_fixed_cl20_t fa = fixed_mv(0, 2, -3, 0);
    const geo_fixed_cl20_t fb = fixed_mv(0, 5, 7, 0);
    const geo_struct_instruction_t structured_code[] = {
        {GEO_STRUCT_OP_ORDERED_PRODUCTS, 2u, 0u, 1u},
        {GEO_STRUCT_OP_HADAMARD_EXACT, 3u, 2u, 2u},
        {GEO_STRUCT_OP_SELECT_SYMMETRIC, 4u, 3u, 3u}
    };
    const geo_struct_program_t structured = {structured_code, 3u, 5u, 4u};
    geo_fused_instruction_t fused_code;
    geo_fused_program_t fused;
    geo_struct_value_t registers[5];
    geo_fixed_geb_result_t fixed_result;
    uint64_t i, begin, end;
    (void)geo_fused_program_for_target(GEO_GEB_VECTOR_DOT, &fused_code, &fused);

    begin = now_ticks(); for (i = 0; i < ITERATIONS; ++i) sink_real += geo_geb_vector_dot(a, b);
    end = now_ticks(); report("vector dot direct C", begin, end);
    begin = now_ticks(); for (i = 0; i < ITERATIONS; ++i) sink_real += geo_native_vector_dot(a, b);
    end = now_ticks(); report("vector dot specialized native C", begin, end);
    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) { registers[0]=geo_struct_value_from_cl20(a); registers[1]=geo_struct_value_from_cl20(b); (void)geo_fused_execute(&fused, registers, 5u); sink_real += registers[fused.root_register].as.scalar; }
    end = now_ticks(); report("vector dot fused IR", begin, end);
    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) { registers[0]=geo_struct_value_from_cl20(a); registers[1]=geo_struct_value_from_cl20(b); (void)geo_struct_program_execute(&structured, registers, 5u); sink_real += registers[4].as.cl20.scalar; }
    end = now_ticks(); report("vector dot structured IR", begin, end);
    begin = now_ticks();
    for (i = 0; i < ITERATIONS; ++i) { (void)geo_fixed_geb36_execute(GEO_GEB_VECTOR_DOT, fa, fb, fa, &fixed_result); consume_fixed(fixed_result.as.scalar); }
    end = now_ticks(); report("vector dot fixed Q16.16", begin, end);
}

static void benchmark_product(void) {
    const geo_cl20_t a = geo_cl20_make(1, 2, -3, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make(-2, 4, 1, (geo_real_t)-0.25);
    const geo_fixed_cl20_t fa = fixed_mv(1, 2, -3, 0.5);
    const geo_fixed_cl20_t fb = fixed_mv(-2, 4, 1, -0.25);
    geo_fixed_geb_result_t fixed_result;
    uint64_t i, begin, end;
    begin = now_ticks(); for (i=0;i<ITERATIONS;++i) sink_real += geo_geb_geometric_product(a,b).e12;
    end=now_ticks(); report("geometric product direct C",begin,end);
    begin = now_ticks(); for (i=0;i<ITERATIONS;++i) sink_real += geo_native_cl20_product(a,b).e12;
    end=now_ticks(); report("geometric product specialized native C",begin,end);
    begin=now_ticks(); for(i=0;i<ITERATIONS;++i){(void)geo_fixed_geb36_execute(GEO_GEB_GEOMETRIC_PRODUCT,fa,fb,fa,&fixed_result);consume_fixed(fixed_result.as.cl20.e12);}
    end=now_ticks(); report("geometric product fixed Q16.16",begin,end);
}

static void benchmark_rotor(void) {
    const geo_cl20_t rotor = geo_cl20_make((geo_real_t)0.9238795,0,0,(geo_real_t)-0.3826834);
    const geo_cl20_t value = geo_cl20_make(0,1,2,0);
    const geo_fixed_cl20_t fr = fixed_mv(0.9238795,0,0,-0.3826834);
    const geo_fixed_cl20_t fv = fixed_mv(0,1,2,0);
    geo_fixed_geb_result_t fixed_result;
    uint64_t i, begin, end;
    begin=now_ticks(); for(i=0;i<ITERATIONS;++i)sink_real+=geo_geb_rotor_action(rotor,value).e1;
    end=now_ticks(); report("rotor action direct C",begin,end);
    begin=now_ticks(); for(i=0;i<ITERATIONS;++i)sink_real+=geo_native_rotor_action(rotor,value).e1;
    end=now_ticks(); report("rotor action specialized native C",begin,end);
    begin=now_ticks(); for(i=0;i<ITERATIONS;++i){(void)geo_fixed_geb36_execute(GEO_GEB_ROTOR_ACTION,fv,fv,fr,&fixed_result);consume_fixed(fixed_result.as.cl20.e1);}
    end=now_ticks(); report("rotor action fixed Q16.16",begin,end);
}

int main(void) {
    puts("Geometric Elementary Operators selected-backend comparison benchmark");
    printf("iterations: %" PRIu64 "\n", ITERATIONS);
    benchmark_addition(); benchmark_dot(); benchmark_product(); benchmark_rotor();
    printf("sinks: %.6f %" PRIu32 "\n", (double)sink_real, sink_fixed);
    return 0;
}
