#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "geo/fixed_geb36.h"
#include "geo/fused.h"
#include "geo/geb36.h"
#include "geo/native_generated.h"
#include "geo/structured_program.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#define DEFAULT_ITERATIONS UINT64_C(1000000)

static uint64_t benchmark_iterations = DEFAULT_ITERATIONS;
static volatile geo_real_t sink_real = (geo_real_t)0;
static volatile uint64_t sink_fixed = UINT64_C(0);

static uint64_t now_ticks(void) {
#if defined(_WIN32)
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return (uint64_t)value.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return UINT64_C(0);
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
    const double value = (double)(end - begin) * tick_ns() /
        (double)benchmark_iterations;
    printf("%-40s %.3f ns/op\n", name, value);
}

static void consume_fixed(geo_fixed_t value) {
    const uint64_t current = sink_fixed;
    const uint64_t word = (uint64_t)(uint32_t)value;
    sink_fixed = current ^
        (word + UINT64_C(0x9e3779b97f4a7c15) +
            (current << 6u) + (current >> 2u));
}

static int parse_iterations(const char *text, uint64_t *output) {
    char *end = NULL;
    unsigned long long parsed;

    if (text == NULL || output == NULL || text[0] == '\0') return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0u) return 0;
    *output = (uint64_t)parsed;
    return 1;
}

static int require_geo_status(geo_status_t status, const char *operation) {
    if (status == GEO_STATUS_OK) return 1;
    fprintf(stderr, "%s failed with geo status %d\n", operation, (int)status);
    return 0;
}

static int require_fixed_status(geo_fixed_status_t status, const char *operation) {
    if (status == GEO_FIXED_OK) return 1;
    fprintf(stderr, "%s failed with fixed status %d\n", operation, (int)status);
    return 0;
}

static int fixed_mv(
    double scalar,
    double e1,
    double e2,
    double e12,
    geo_fixed_cl20_t *output
) {
    if (output == NULL) return 0;
    if (!require_fixed_status(
            geo_fixed_from_double(scalar, &output->scalar),
            "fixed scalar conversion")) return 0;
    if (!require_fixed_status(
            geo_fixed_from_double(e1, &output->e1),
            "fixed e1 conversion")) return 0;
    if (!require_fixed_status(
            geo_fixed_from_double(e2, &output->e2),
            "fixed e2 conversion")) return 0;
    if (!require_fixed_status(
            geo_fixed_from_double(e12, &output->e12),
            "fixed e12 conversion")) return 0;
    return 1;
}

static void print_operation_path_matrix(void) {
    puts("GEO_BENCH_MATRIX|operation=addition|path=direct_c");
    puts("GEO_BENCH_MATRIX|operation=addition|path=specialized_native_c");
    puts("GEO_BENCH_MATRIX|operation=addition|path=fused_ir");
    puts("GEO_BENCH_MATRIX|operation=addition|path=structured_ir");
    puts("GEO_BENCH_MATRIX|operation=addition|path=fixed_qformat");
    puts("GEO_BENCH_MATRIX|operation=vector_dot|path=direct_c");
    puts("GEO_BENCH_MATRIX|operation=vector_dot|path=specialized_native_c");
    puts("GEO_BENCH_MATRIX|operation=vector_dot|path=fused_ir");
    puts("GEO_BENCH_MATRIX|operation=vector_dot|path=structured_ir");
    puts("GEO_BENCH_MATRIX|operation=vector_dot|path=fixed_qformat");
    puts("GEO_BENCH_MATRIX|operation=geometric_product|path=direct_c");
    puts("GEO_BENCH_MATRIX|operation=geometric_product|path=specialized_native_c");
    puts("GEO_BENCH_MATRIX|operation=geometric_product|path=fixed_qformat");
    puts("GEO_BENCH_MATRIX|operation=rotor_action|path=direct_c");
    puts("GEO_BENCH_MATRIX|operation=rotor_action|path=specialized_native_c");
    puts("GEO_BENCH_MATRIX|operation=rotor_action|path=fixed_qformat");
}

static int benchmark_addition(void) {
    const geo_cl20_t a = geo_cl20_make(1, 2, -3, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make(-2, 4, 1, (geo_real_t)-0.25);
    geo_fixed_cl20_t fixed_a;
    geo_fixed_cl20_t fixed_b;
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
    uint64_t index;
    uint64_t begin;
    uint64_t end;

    if (!fixed_mv(1, 2, -3, 0.5, &fixed_a) ||
        !fixed_mv(-2, 4, 1, -0.25, &fixed_b)) return 0;
    if (!require_geo_status(
            geo_fused_program_for_target(GEO_GEB_ADDITION, &fused_code, &fused),
            "addition fused program construction")) return 0;

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_geb_addition(a, b).e1;
    }
    end = now_ticks();
    report("addition direct C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_native_add(a, b).e1;
    }
    end = now_ticks();
    report("addition specialized native C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        if (!require_geo_status(
                geo_fused_execute(&fused, registers, 6u),
                "addition fused execution")) return 0;
        if (registers[fused.root_register].kind != (uint8_t)GEO_STRUCT_VALUE_CL20) {
            fputs("addition fused execution returned the wrong result kind\n", stderr);
            return 0;
        }
        sink_real += registers[fused.root_register].as.cl20.e1;
    }
    end = now_ticks();
    report("addition fused IR", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        if (!require_geo_status(
                geo_struct_program_execute(&structured, registers, 6u),
                "addition structured execution")) return 0;
        if (registers[5].kind != (uint8_t)GEO_STRUCT_VALUE_CL20) {
            fputs("addition structured execution returned the wrong result kind\n", stderr);
            return 0;
        }
        sink_real += registers[5].as.cl20.e1;
    }
    end = now_ticks();
    report("addition structured IR", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        if (!require_fixed_status(
                geo_fixed_geb36_execute(
                    GEO_GEB_ADDITION,
                    fixed_a,
                    fixed_b,
                    fixed_a,
                    &fixed_result
                ),
                "addition fixed execution")) return 0;
        if (fixed_result.kind != (uint8_t)GEO_FIXED_RESULT_CL20) {
            fputs("addition fixed execution returned the wrong result kind\n", stderr);
            return 0;
        }
        consume_fixed(fixed_result.as.cl20.e1);
    }
    end = now_ticks();
    report("addition fixed Q-format", begin, end);
    return 1;
}

static int benchmark_dot(void) {
    const geo_cl20_t a = geo_cl20_make(0, 2, -3, 0);
    const geo_cl20_t b = geo_cl20_make(0, 5, 7, 0);
    geo_fixed_cl20_t fixed_a;
    geo_fixed_cl20_t fixed_b;
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
    uint64_t index;
    uint64_t begin;
    uint64_t end;

    if (!fixed_mv(0, 2, -3, 0, &fixed_a) ||
        !fixed_mv(0, 5, 7, 0, &fixed_b)) return 0;
    if (!require_geo_status(
            geo_fused_program_for_target(GEO_GEB_VECTOR_DOT, &fused_code, &fused),
            "vector-dot fused program construction")) return 0;

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_geb_vector_dot(a, b);
    }
    end = now_ticks();
    report("vector dot direct C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_native_vector_dot(a, b);
    }
    end = now_ticks();
    report("vector dot specialized native C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        if (!require_geo_status(
                geo_fused_execute(&fused, registers, 5u),
                "vector-dot fused execution")) return 0;
        if (registers[fused.root_register].kind != (uint8_t)GEO_STRUCT_VALUE_SCALAR) {
            fputs("vector-dot fused execution returned the wrong result kind\n", stderr);
            return 0;
        }
        sink_real += registers[fused.root_register].as.scalar;
    }
    end = now_ticks();
    report("vector dot fused IR", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        registers[0] = geo_struct_value_from_cl20(a);
        registers[1] = geo_struct_value_from_cl20(b);
        if (!require_geo_status(
                geo_struct_program_execute(&structured, registers, 5u),
                "vector-dot structured execution")) return 0;
        if (registers[4].kind != (uint8_t)GEO_STRUCT_VALUE_CL20) {
            fputs("vector-dot structured execution returned the wrong result kind\n", stderr);
            return 0;
        }
        sink_real += registers[4].as.cl20.scalar;
    }
    end = now_ticks();
    report("vector dot structured IR", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        if (!require_fixed_status(
                geo_fixed_geb36_execute(
                    GEO_GEB_VECTOR_DOT,
                    fixed_a,
                    fixed_b,
                    fixed_a,
                    &fixed_result
                ),
                "vector-dot fixed execution")) return 0;
        if (fixed_result.kind != (uint8_t)GEO_FIXED_RESULT_SCALAR) {
            fputs("vector-dot fixed execution returned the wrong result kind\n", stderr);
            return 0;
        }
        consume_fixed(fixed_result.as.scalar);
    }
    end = now_ticks();
    report("vector dot fixed Q-format", begin, end);
    return 1;
}

static int benchmark_product(void) {
    const geo_cl20_t a = geo_cl20_make(1, 2, -3, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make(-2, 4, 1, (geo_real_t)-0.25);
    geo_fixed_cl20_t fixed_a;
    geo_fixed_cl20_t fixed_b;
    geo_fixed_geb_result_t fixed_result;
    uint64_t index;
    uint64_t begin;
    uint64_t end;

    if (!fixed_mv(1, 2, -3, 0.5, &fixed_a) ||
        !fixed_mv(-2, 4, 1, -0.25, &fixed_b)) return 0;

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_geb_geometric_product(a, b).e12;
    }
    end = now_ticks();
    report("geometric product direct C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_native_cl20_product(a, b).e12;
    }
    end = now_ticks();
    report("geometric product specialized native C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        if (!require_fixed_status(
                geo_fixed_geb36_execute(
                    GEO_GEB_GEOMETRIC_PRODUCT,
                    fixed_a,
                    fixed_b,
                    fixed_a,
                    &fixed_result
                ),
                "geometric-product fixed execution")) return 0;
        if (fixed_result.kind != (uint8_t)GEO_FIXED_RESULT_CL20) {
            fputs("geometric-product fixed execution returned the wrong result kind\n", stderr);
            return 0;
        }
        consume_fixed(fixed_result.as.cl20.e12);
    }
    end = now_ticks();
    report("geometric product fixed Q-format", begin, end);
    return 1;
}

static int benchmark_rotor(void) {
    const geo_cl20_t rotor = geo_cl20_make(
        (geo_real_t)0.9238795,
        0,
        0,
        (geo_real_t)-0.3826834
    );
    const geo_cl20_t value = geo_cl20_make(0, 1, 2, 0);
    geo_fixed_cl20_t fixed_rotor;
    geo_fixed_cl20_t fixed_value;
    geo_fixed_geb_result_t fixed_result;
    uint64_t index;
    uint64_t begin;
    uint64_t end;

    if (!fixed_mv(0.9238795, 0, 0, -0.3826834, &fixed_rotor) ||
        !fixed_mv(0, 1, 2, 0, &fixed_value)) return 0;

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_geb_rotor_action(rotor, value).e1;
    }
    end = now_ticks();
    report("rotor action direct C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        sink_real += geo_native_rotor_action(rotor, value).e1;
    }
    end = now_ticks();
    report("rotor action specialized native C", begin, end);

    begin = now_ticks();
    for (index = 0u; index < benchmark_iterations; ++index) {
        if (!require_fixed_status(
                geo_fixed_geb36_execute(
                    GEO_GEB_ROTOR_ACTION,
                    fixed_value,
                    fixed_value,
                    fixed_rotor,
                    &fixed_result
                ),
                "rotor-action fixed execution")) return 0;
        if (fixed_result.kind != (uint8_t)GEO_FIXED_RESULT_CL20) {
            fputs("rotor-action fixed execution returned the wrong result kind\n", stderr);
            return 0;
        }
        consume_fixed(fixed_result.as.cl20.e1);
    }
    end = now_ticks();
    report("rotor action fixed Q-format", begin, end);
    return 1;
}

static void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s [--iterations N]\n", program);
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--iterations") == 0) {
        if (!parse_iterations(argv[2], &benchmark_iterations)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    } else if (argc != 1) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    puts("Geometric Elementary Operators selected-path comparison benchmark");
    printf("iterations: %" PRIu64 "\n", benchmark_iterations);
    print_operation_path_matrix();

    if (!benchmark_addition() || !benchmark_dot() ||
        !benchmark_product() || !benchmark_rotor()) {
        return EXIT_FAILURE;
    }

    printf("sinks: %.6f %" PRIu64 "\n", (double)sink_real, sink_fixed);
    return EXIT_SUCCESS;
}
