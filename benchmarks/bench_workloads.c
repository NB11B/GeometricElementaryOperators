#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "geo/banked.h"
#include "geo/fixed_program.h"
#include "geo/folding.h"
#include "geo/fused.h"
#include "geo/geb36.h"
#include "geo/geb_witness.h"
#include "geo/native_generated.h"
#include "geo/optimizer.h"
#include "geo/structured_program.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#define GEO_WORKLOAD_CAPACITY 32u

typedef struct {
    uint64_t iterations;
    uint64_t warmup;
    uint32_t seed;
    const char *csv_path;
} workload_options_t;

typedef struct {
    double max_absolute;
    double max_relative;
    size_t mismatches;
} error_stats_t;

typedef int (*workload_step_t)(void *context);

static volatile uint64_t workload_sink = UINT64_C(0);

static void consume_real(geo_real_t value) {
    uint64_t bits = UINT64_C(0);
    memcpy(&bits, &value, sizeof(value));
    workload_sink = (workload_sink << 1u) ^ bits ^ UINT64_C(0x9e3779b97f4a7c15);
}

static void consume_fixed(geo_fixed_t value) {
    workload_sink = (workload_sink << 1u) ^ (uint32_t)value ^ UINT64_C(0x517cc1b727220a95);
}

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

static double tick_nanoseconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return 1.0e9 / (double)frequency.QuadPart;
#else
    return 1.0;
#endif
}

static int parse_u64(const char *text, uint64_t *output) {
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || output == NULL || text[0] == '\0') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return 0;
    *output = (uint64_t)value;
    return 1;
}

static void print_usage(const char *program) {
    printf(
        "Usage: %s [--iterations N] [--warmup N] [--seed N] [--csv PATH]\n",
        program
    );
}

static int parse_options(int argc, char **argv, workload_options_t *options) {
    int index;
    if (options == NULL) return 0;
    options->iterations = UINT64_C(100000);
    options->warmup = UINT64_C(1000);
    options->seed = UINT32_C(0x243f6a88);
    options->csv_path = NULL;

    for (index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        uint64_t parsed;
        if (strcmp(argument, "--help") == 0 || strcmp(argument, "-h") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc) return 0;
        ++index;
        if (strcmp(argument, "--iterations") == 0) {
            if (!parse_u64(argv[index], &parsed) || parsed == 0u) return 0;
            options->iterations = parsed;
        } else if (strcmp(argument, "--warmup") == 0) {
            if (!parse_u64(argv[index], &parsed)) return 0;
            options->warmup = parsed;
        } else if (strcmp(argument, "--seed") == 0) {
            if (!parse_u64(argv[index], &parsed) || parsed > UINT32_MAX) return 0;
            options->seed = (uint32_t)parsed;
        } else if (strcmp(argument, "--csv") == 0) {
            options->csv_path = argv[index];
        } else {
            return 0;
        }
    }
    return 1;
}

static double measure_step(
    workload_step_t step,
    void *context,
    const workload_options_t *options,
    int *status
) {
    uint64_t index;
    uint64_t begin;
    uint64_t end;
    int result = 0;

    for (index = 0u; index < options->warmup; ++index) {
        result = step(context);
        if (result != 0) {
            if (status != NULL) *status = result;
            return -1.0;
        }
    }

    begin = now_ticks();
    for (index = 0u; index < options->iterations; ++index) {
        result = step(context);
        if (result != 0) break;
    }
    end = now_ticks();

    if (status != NULL) *status = result;
    if (result != 0 || end < begin) return -1.0;
    return (double)(end - begin) * tick_nanoseconds() /
        (double)options->iterations;
}

static void update_error(
    error_stats_t *stats,
    double actual,
    double expected,
    double absolute_tolerance,
    double relative_tolerance
) {
    const double absolute = fabs(actual - expected);
    const double denominator = fmax(fabs(expected), 1.0e-30);
    const double relative = absolute / denominator;
    if (absolute > stats->max_absolute) stats->max_absolute = absolute;
    if (relative > stats->max_relative) stats->max_relative = relative;
    if (!isfinite(actual) || !isfinite(expected) ||
        absolute > absolute_tolerance + relative_tolerance * fabs(expected)) {
        ++stats->mismatches;
    }
}

static error_stats_t mv_error(
    geo_cl20_t actual,
    geo_cl20_t expected,
    double absolute_tolerance,
    double relative_tolerance
) {
    error_stats_t stats = {0.0, 0.0, 0u};
    update_error(&stats, (double)actual.scalar, (double)expected.scalar,
        absolute_tolerance, relative_tolerance);
    update_error(&stats, (double)actual.e1, (double)expected.e1,
        absolute_tolerance, relative_tolerance);
    update_error(&stats, (double)actual.e2, (double)expected.e2,
        absolute_tolerance, relative_tolerance);
    update_error(&stats, (double)actual.e12, (double)expected.e12,
        absolute_tolerance, relative_tolerance);
    return stats;
}

static error_stats_t scalar_error(
    geo_real_t actual,
    geo_real_t expected,
    double absolute_tolerance,
    double relative_tolerance
) {
    error_stats_t stats = {0.0, 0.0, 0u};
    update_error(&stats, (double)actual, (double)expected,
        absolute_tolerance, relative_tolerance);
    return stats;
}

static void emit_row(
    FILE *csv,
    const char *operation,
    const char *backend,
    double ns_per_operation,
    error_stats_t errors
) {
    printf(
        "RESULT operation=%s backend=%s ns_per_op=%.6f max_absolute=%.17g "
        "max_relative=%.17g mismatches=%zu\n",
        operation,
        backend,
        ns_per_operation,
        errors.max_absolute,
        errors.max_relative,
        errors.mismatches
    );
    if (csv != NULL) {
        fprintf(
            csv,
            "%s,%s,%.9f,%.17g,%.17g,%zu\n",
            operation,
            backend,
            ns_per_operation,
            errors.max_absolute,
            errors.max_relative,
            errors.mismatches
        );
    }
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static geo_real_t random_real(uint32_t *state) {
    const uint32_t value = next_random(state) >> 8;
    const double unit = (double)value / (double)UINT32_C(0x00ffffff);
    return (geo_real_t)(unit - 0.5);
}

static geo_cl20_t random_mv(uint32_t *state) {
    return geo_cl20_make(
        random_real(state),
        random_real(state),
        random_real(state),
        random_real(state)
    );
}

static geo_cl20_t random_vector(uint32_t *state) {
    return geo_cl20_make(
        (geo_real_t)0,
        random_real(state),
        random_real(state),
        (geo_real_t)0
    );
}

static geo_cl20_t rotor_from_seed(uint32_t *state) {
    const double angle = 0.25 + 0.25 * (double)(next_random(state) & UINT32_C(0xffff)) /
        65535.0;
    return geo_cl20_make(
        (geo_real_t)cos(angle),
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)-sin(angle)
    );
}

static geo_cl20_t reference_mv(
    uint8_t target,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform
) {
    switch ((geo_geb_target_id_t)target) {
        case GEO_GEB_ADDITION: return geo_geb_addition(a, b);
        case GEO_GEB_GEOMETRIC_PRODUCT: return geo_geb_geometric_product(a, b);
        case GEO_GEB_REVERSE_PRODUCT: return geo_geb_reverse_product(a, b);
        case GEO_GEB_ROTOR_ACTION: return geo_geb_rotor_action(transform, a);
        case GEO_GEB_ROTOR_COMPOSITION: return geo_geb_rotor_composition(a, b);
        case GEO_GEB_DILATION: return geo_geb_dilation(transform, a);
        default: return geo_cl20_zero();
    }
}

typedef struct {
    uint8_t target;
    geo_cl20_t a;
    geo_cl20_t b;
    geo_cl20_t transform;
    geo_cl20_t output;
} mv_context_t;

static int direct_mv_step(void *opaque) {
    mv_context_t *context = (mv_context_t *)opaque;
    context->output = reference_mv(
        context->target,
        context->a,
        context->b,
        context->transform
    );
    consume_real(context->output.e1);
    return 0;
}

static int native_mv_step(void *opaque) {
    mv_context_t *context = (mv_context_t *)opaque;
    switch ((geo_geb_target_id_t)context->target) {
        case GEO_GEB_ADDITION:
            context->output = geo_native_add(context->a, context->b);
            break;
        case GEO_GEB_GEOMETRIC_PRODUCT:
            context->output = geo_native_cl20_product(context->a, context->b);
            break;
        case GEO_GEB_ROTOR_ACTION:
            context->output = geo_native_rotor_action(context->transform, context->a);
            break;
        default:
            return 1;
    }
    consume_real(context->output.e1);
    return 0;
}

typedef struct {
    geo_cl20_t a;
    geo_cl20_t b;
    geo_real_t output;
} scalar_context_t;

static int direct_dot_step(void *opaque) {
    scalar_context_t *context = (scalar_context_t *)opaque;
    context->output = geo_geb_vector_dot(context->a, context->b);
    consume_real(context->output);
    return 0;
}

static int native_dot_step(void *opaque) {
    scalar_context_t *context = (scalar_context_t *)opaque;
    context->output = geo_native_vector_dot(context->a, context->b);
    consume_real(context->output);
    return 0;
}

typedef struct {
    geo_struct_instruction_t instructions[4];
    geo_struct_program_t program;
    geo_struct_value_t registers[6];
    geo_cl20_t a;
    geo_cl20_t b;
    geo_cl20_t mv_output;
    geo_real_t scalar_output;
    int scalar;
} structured_context_t;

static void setup_structured_add(
    structured_context_t *context,
    geo_cl20_t a,
    geo_cl20_t b
) {
    memset(context, 0, sizeof(*context));
    context->instructions[0] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_UNIPOTENT_ENCODE, 2u, 0u, 0u
    };
    context->instructions[1] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_UNIPOTENT_ENCODE, 3u, 1u, 1u
    };
    context->instructions[2] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_UNIPOTENT_COMPOSE, 4u, 2u, 3u
    };
    context->instructions[3] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_UNIPOTENT_EXTRACT, 5u, 4u, 4u
    };
    context->program = (geo_struct_program_t){
        context->instructions, 4u, 6u, 5u
    };
    context->a = a;
    context->b = b;
    context->scalar = 0;
}

static void setup_structured_dot(
    structured_context_t *context,
    geo_cl20_t a,
    geo_cl20_t b
) {
    memset(context, 0, sizeof(*context));
    context->instructions[0] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_ORDERED_PRODUCTS, 2u, 0u, 1u
    };
    context->instructions[1] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_HADAMARD_EXACT, 3u, 2u, 2u
    };
    context->instructions[2] = (geo_struct_instruction_t){
        GEO_STRUCT_OP_SELECT_SYMMETRIC, 4u, 3u, 3u
    };
    context->program = (geo_struct_program_t){
        context->instructions, 3u, 5u, 4u
    };
    context->a = a;
    context->b = b;
    context->scalar = 1;
}

static int structured_step(void *opaque) {
    structured_context_t *context = (structured_context_t *)opaque;
    geo_status_t status;
    context->registers[0] = geo_struct_value_from_cl20(context->a);
    context->registers[1] = geo_struct_value_from_cl20(context->b);
    status = geo_struct_program_execute(
        &context->program,
        context->registers,
        6u
    );
    if (status != GEO_STATUS_OK) return (int)status;
    if (context->scalar) {
        context->scalar_output =
            context->registers[context->program.root_register].as.cl20.scalar;
        consume_real(context->scalar_output);
    } else {
        context->mv_output =
            context->registers[context->program.root_register].as.cl20;
        consume_real(context->mv_output.e1);
    }
    return 0;
}

typedef struct {
    geo_fused_instruction_t instruction;
    geo_fused_program_t program;
    geo_struct_value_t registers[6];
    geo_cl20_t a;
    geo_cl20_t b;
    geo_cl20_t mv_output;
    geo_real_t scalar_output;
    int scalar;
} fused_context_t;

static int setup_fused(
    fused_context_t *context,
    uint8_t target,
    geo_cl20_t a,
    geo_cl20_t b,
    int scalar
) {
    memset(context, 0, sizeof(*context));
    context->a = a;
    context->b = b;
    context->scalar = scalar;
    return geo_fused_program_for_target(
        target,
        &context->instruction,
        &context->program
    ) == GEO_STATUS_OK;
}

static int fused_step(void *opaque) {
    fused_context_t *context = (fused_context_t *)opaque;
    geo_status_t status;
    context->registers[0] = geo_struct_value_from_cl20(context->a);
    context->registers[1] = geo_struct_value_from_cl20(context->b);
    status = geo_fused_execute(&context->program, context->registers, 6u);
    if (status != GEO_STATUS_OK) return (int)status;
    if (context->scalar) {
        context->scalar_output =
            context->registers[context->program.root_register].as.scalar;
        consume_real(context->scalar_output);
    } else {
        context->mv_output =
            context->registers[context->program.root_register].as.cl20;
        consume_real(context->mv_output.e1);
    }
    return 0;
}

static int fixed_from_mv(geo_cl20_t input, geo_fixed_cl20_t *output) {
    if (geo_fixed_from_double((double)input.scalar, &output->scalar) != GEO_FIXED_OK) return 0;
    if (geo_fixed_from_double((double)input.e1, &output->e1) != GEO_FIXED_OK) return 0;
    if (geo_fixed_from_double((double)input.e2, &output->e2) != GEO_FIXED_OK) return 0;
    if (geo_fixed_from_double((double)input.e12, &output->e12) != GEO_FIXED_OK) return 0;
    return 1;
}

typedef struct {
    geo_fixed_program_instruction_t instruction;
    geo_fixed_program_t program;
    geo_fixed_geb_result_t registers[4];
    geo_fixed_program_status_t status;
} fixed_context_t;

static int setup_fixed(
    fixed_context_t *context,
    uint8_t target,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform
) {
    geo_fixed_cl20_t fixed_a;
    geo_fixed_cl20_t fixed_b;
    geo_fixed_cl20_t fixed_transform;
    memset(context, 0, sizeof(*context));
    if (!fixed_from_mv(a, &fixed_a) ||
        !fixed_from_mv(b, &fixed_b) ||
        !fixed_from_mv(transform, &fixed_transform)) {
        return 0;
    }
    context->instruction = (geo_fixed_program_instruction_t){
        target, 3u, 0u, 1u, 2u
    };
    context->program = (geo_fixed_program_t){
        &context->instruction, 1u, 4u, 3u
    };
    context->registers[0] = geo_fixed_program_value_from_cl20(fixed_a);
    context->registers[1] = geo_fixed_program_value_from_cl20(fixed_b);
    context->registers[2] = geo_fixed_program_value_from_cl20(fixed_transform);
    return 1;
}

static int fixed_step(void *opaque) {
    fixed_context_t *context = (fixed_context_t *)opaque;
    context->status = geo_fixed_program_execute(
        &context->program,
        context->registers,
        4u
    );
    if (context->status != GEO_FIXED_PROGRAM_OK) return (int)context->status;
    switch ((geo_fixed_result_kind_t)context->registers[3].kind) {
        case GEO_FIXED_RESULT_SCALAR:
            consume_fixed(context->registers[3].as.scalar);
            break;
        case GEO_FIXED_RESULT_PROJECTIVE:
            consume_fixed(context->registers[3].as.projective.represented.e1);
            break;
        case GEO_FIXED_RESULT_UNIPOTENT:
            consume_fixed(context->registers[3].as.unipotent_payload.e1);
            break;
        case GEO_FIXED_RESULT_CL20:
        default:
            consume_fixed(context->registers[3].as.cl20.e1);
            break;
    }
    return 0;
}

static int fixed_output_mv(fixed_context_t *context, geo_cl20_t *output) {
    geo_fixed_cl20_t value;
    if (geo_fixed_program_read_cl20(&context->registers[3], &value) !=
        GEO_FIXED_PROGRAM_OK) {
        return 0;
    }
    output->scalar = (geo_real_t)geo_fixed_to_double(value.scalar);
    output->e1 = (geo_real_t)geo_fixed_to_double(value.e1);
    output->e2 = (geo_real_t)geo_fixed_to_double(value.e2);
    output->e12 = (geo_real_t)geo_fixed_to_double(value.e12);
    return 1;
}

static int fixed_output_scalar(fixed_context_t *context, geo_real_t *output) {
    if (context->registers[3].kind != (uint8_t)GEO_FIXED_RESULT_SCALAR) return 0;
    *output = (geo_real_t)geo_fixed_to_double(context->registers[3].as.scalar);
    return 1;
}

typedef struct {
    const geo_geb_witness_t *witness;
    geo_instruction_t optimized_instructions[GEO_WORKLOAD_CAPACITY];
    uint8_t node_registers[GEO_WORKLOAD_CAPACITY];
    uint8_t live_lanes[GEO_WORKLOAD_CAPACITY];
    uint16_t representatives[GEO_WORKLOAD_CAPACITY];
    geo_optimizer_workspace_t optimizer_workspace;
    geo_optimized_witness_t optimized;
    geo_instruction_t folded_instructions[GEO_WORKLOAD_CAPACITY];
    geo_state_t initial_registers[GEO_WORKLOAD_CAPACITY];
    uint8_t old_to_new[GEO_WORKLOAD_CAPACITY];
    uint8_t constant_flags[GEO_WORKLOAD_CAPACITY];
    uint8_t register_kinds[GEO_WORKLOAD_CAPACITY];
    geo_folding_workspace_t folding_workspace;
    geo_folded_program_t folded;
    geo_banked_instruction_t banked_instructions[GEO_WORKLOAD_CAPACITY];
    geo_banked_ref_t logical_refs[GEO_WORKLOAD_CAPACITY];
    geo_banked_plan_workspace_t plan_workspace;
    geo_banked_program_t banked;
    geo_real_t scalars[GEO_WORKLOAD_CAPACITY];
    geo_geometric_register_t geometrics[GEO_WORKLOAD_CAPACITY];
    geo_state_t unified[GEO_WORKLOAD_CAPACITY];
    geo_banked_storage_t storage;
    geo_cl20_t output;
} banked_context_t;

static int setup_banked(
    banked_context_t *context,
    uint8_t target,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform
) {
    geo_state_t terminals[3];
    uint8_t terminal_constant_flags[3] = {0u, 0u, 0u};
    geo_status_t status;

    memset(context, 0, sizeof(*context));
    context->witness = geo_geb_witness_for_target(target);
    if (context->witness == NULL) return 0;

    context->optimizer_workspace = (geo_optimizer_workspace_t){
        context->optimized_instructions, GEO_WORKLOAD_CAPACITY,
        context->node_registers, GEO_WORKLOAD_CAPACITY,
        context->live_lanes, GEO_WORKLOAD_CAPACITY,
        context->representatives, GEO_WORKLOAD_CAPACITY
    };
    context->folding_workspace = (geo_folding_workspace_t){
        context->folded_instructions, GEO_WORKLOAD_CAPACITY,
        context->initial_registers, GEO_WORKLOAD_CAPACITY,
        context->old_to_new, GEO_WORKLOAD_CAPACITY,
        context->constant_flags, GEO_WORKLOAD_CAPACITY,
        context->register_kinds, GEO_WORKLOAD_CAPACITY
    };
    context->plan_workspace = (geo_banked_plan_workspace_t){
        context->banked_instructions, GEO_WORKLOAD_CAPACITY,
        context->logical_refs, GEO_WORKLOAD_CAPACITY
    };
    context->storage = (geo_banked_storage_t){
        context->scalars, GEO_WORKLOAD_CAPACITY,
        context->geometrics, GEO_WORKLOAD_CAPACITY,
        context->unified, GEO_WORKLOAD_CAPACITY
    };

    if (target == (uint8_t)GEO_GEB_ROTOR_ACTION ||
        target == (uint8_t)GEO_GEB_DILATION) {
        terminals[0] = geo_state_from_cl20(transform);
        terminals[1] = geo_state_from_cl20(a);
        terminals[2] = geo_state_from_cl20(geo_cl20_reverse(transform));
    } else {
        terminals[0] = geo_state_from_cl20(a);
        terminals[1] = geo_state_from_cl20(b);
        terminals[2] = geo_state_zero();
    }

    status = geo_witness_compile_optimized(
        &context->witness->tree,
        &context->optimizer_workspace,
        &context->optimized
    );
    if (status != GEO_STATUS_OK) return 0;
    status = geo_program_fold_constants(
        &context->optimized,
        terminals,
        terminal_constant_flags,
        context->witness->tree.terminal_count,
        &context->folding_workspace,
        &context->folded
    );
    if (status != GEO_STATUS_OK) return 0;
    status = geo_banked_plan(
        &context->folded,
        &context->plan_workspace,
        &context->banked
    );
    return status == GEO_STATUS_OK;
}

static int banked_step(void *opaque) {
    banked_context_t *context = (banked_context_t *)opaque;
    geo_state_t root;
    geo_status_t status;

    status = geo_banked_initialize(
        &context->folded,
        &context->banked,
        context->logical_refs,
        &context->storage
    );
    if (status != GEO_STATUS_OK) return (int)status;
    status = geo_banked_execute(&context->banked, &context->storage);
    if (status != GEO_STATUS_OK) return (int)status;
    status = geo_banked_read_state(&context->storage, context->banked.root, &root);
    if (status != GEO_STATUS_OK) return (int)status;
    status = geo_geb_witness_extract(context->witness, &root, &context->output);
    if (status != GEO_STATUS_OK) return (int)status;
    consume_real(context->output.e1);
    return 0;
}

static double native_tolerance(void) {
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    return 1.0e-12;
#else
    return 2.0e-5;
#endif
}

static double fixed_tolerance(void) {
    return 128.0 / (double)(INT64_C(1) << GEO_FIXED_FRACTION_BITS);
}

static int record_mv_backend(
    FILE *csv,
    const workload_options_t *options,
    const char *operation,
    const char *backend,
    workload_step_t step,
    void *context,
    geo_cl20_t *actual,
    geo_cl20_t expected,
    double tolerance
) {
    int status = 0;
    const double ns = measure_step(step, context, options, &status);
    error_stats_t errors;
    if (ns < 0.0 || status != 0) {
        fprintf(stderr, "backend failure operation=%s backend=%s status=%d\n",
            operation, backend, status);
        return 0;
    }
    errors = mv_error(*actual, expected, tolerance, tolerance);
    emit_row(csv, operation, backend, ns, errors);
    return errors.mismatches == 0u;
}

static int record_scalar_backend(
    FILE *csv,
    const workload_options_t *options,
    const char *operation,
    const char *backend,
    workload_step_t step,
    void *context,
    geo_real_t *actual,
    geo_real_t expected,
    double tolerance
) {
    int status = 0;
    const double ns = measure_step(step, context, options, &status);
    error_stats_t errors;
    if (ns < 0.0 || status != 0) {
        fprintf(stderr, "backend failure operation=%s backend=%s status=%d\n",
            operation, backend, status);
        return 0;
    }
    errors = scalar_error(*actual, expected, tolerance, tolerance);
    emit_row(csv, operation, backend, ns, errors);
    return errors.mismatches == 0u;
}

static int record_fixed_mv(
    FILE *csv,
    const workload_options_t *options,
    const char *operation,
    fixed_context_t *context,
    geo_cl20_t expected
) {
    int status = 0;
    const double ns = measure_step(fixed_step, context, options, &status);
    geo_cl20_t actual;
    error_stats_t errors;
    if (ns < 0.0 || status != 0 || !fixed_output_mv(context, &actual)) {
        fprintf(stderr, "fixed backend failure operation=%s status=%d\n",
            operation, status);
        return 0;
    }
    errors = mv_error(actual, expected, fixed_tolerance(), fixed_tolerance());
    emit_row(csv, operation, "fixed_program", ns, errors);
    return errors.mismatches == 0u;
}

static int record_fixed_scalar(
    FILE *csv,
    const workload_options_t *options,
    const char *operation,
    fixed_context_t *context,
    geo_real_t expected
) {
    int status = 0;
    const double ns = measure_step(fixed_step, context, options, &status);
    geo_real_t actual;
    error_stats_t errors;
    if (ns < 0.0 || status != 0 || !fixed_output_scalar(context, &actual)) {
        fprintf(stderr, "fixed backend failure operation=%s status=%d\n",
            operation, status);
        return 0;
    }
    errors = scalar_error(actual, expected, fixed_tolerance(), fixed_tolerance());
    emit_row(csv, operation, "fixed_program", ns, errors);
    return errors.mismatches == 0u;
}

static int run_addition(
    FILE *csv,
    const workload_options_t *options,
    geo_cl20_t a,
    geo_cl20_t b
) {
    const char *operation = "addition";
    const geo_cl20_t expected = geo_geb_addition(a, b);
    mv_context_t direct = {(uint8_t)GEO_GEB_ADDITION, a, b, geo_cl20_zero(), geo_cl20_zero()};
    mv_context_t native = direct;
    structured_context_t structured;
    fused_context_t fused;
    fixed_context_t fixed_context;
    int ok = 1;

    setup_structured_add(&structured, a, b);
    if (!setup_fused(&fused, GEO_GEB_ADDITION, a, b, 0) ||
        !setup_fixed(&fixed_context, GEO_GEB_ADDITION, a, b, geo_cl20_zero())) {
        return 0;
    }

    ok &= record_mv_backend(csv, options, operation, "direct_c", direct_mv_step,
        &direct, &direct.output, expected, native_tolerance());
    ok &= record_mv_backend(csv, options, operation, "native_specialized", native_mv_step,
        &native, &native.output, expected, native_tolerance());
    ok &= record_mv_backend(csv, options, operation, "structured_ir", structured_step,
        &structured, &structured.mv_output, expected, native_tolerance());
    ok &= record_mv_backend(csv, options, operation, "fused_ir", fused_step,
        &fused, &fused.mv_output, expected, native_tolerance());
    ok &= record_fixed_mv(csv, options, operation, &fixed_context, expected);
    return ok;
}

static int run_dot(
    FILE *csv,
    const workload_options_t *options,
    geo_cl20_t a,
    geo_cl20_t b
) {
    const char *operation = "vector_dot";
    const geo_real_t expected = geo_geb_vector_dot(a, b);
    scalar_context_t direct = {a, b, (geo_real_t)0};
    scalar_context_t native = direct;
    structured_context_t structured;
    fused_context_t fused;
    fixed_context_t fixed_context;
    int ok = 1;

    setup_structured_dot(&structured, a, b);
    if (!setup_fused(&fused, GEO_GEB_VECTOR_DOT, a, b, 1) ||
        !setup_fixed(&fixed_context, GEO_GEB_VECTOR_DOT, a, b, geo_cl20_zero())) {
        return 0;
    }

    ok &= record_scalar_backend(csv, options, operation, "direct_c", direct_dot_step,
        &direct, &direct.output, expected, native_tolerance());
    ok &= record_scalar_backend(csv, options, operation, "native_specialized", native_dot_step,
        &native, &native.output, expected, native_tolerance());
    ok &= record_scalar_backend(csv, options, operation, "structured_ir", structured_step,
        &structured, &structured.scalar_output, expected, native_tolerance());
    ok &= record_scalar_backend(csv, options, operation, "fused_ir", fused_step,
        &fused, &fused.scalar_output, expected, native_tolerance());
    ok &= record_fixed_scalar(csv, options, operation, &fixed_context, expected);
    return ok;
}

static int run_mv_family(
    FILE *csv,
    const workload_options_t *options,
    const char *operation,
    uint8_t target,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform,
    int native_supported,
    int banked_supported
) {
    const geo_cl20_t expected = reference_mv(target, a, b, transform);
    mv_context_t direct = {target, a, b, transform, geo_cl20_zero()};
    mv_context_t native = direct;
    fixed_context_t fixed_context;
    banked_context_t banked;
    int ok = 1;

    if (!setup_fixed(&fixed_context, target, a, b, transform)) return 0;
    ok &= record_mv_backend(csv, options, operation, "direct_c", direct_mv_step,
        &direct, &direct.output, expected, native_tolerance());
    if (native_supported) {
        ok &= record_mv_backend(csv, options, operation, "native_specialized", native_mv_step,
            &native, &native.output, expected, native_tolerance());
    }
    ok &= record_fixed_mv(csv, options, operation, &fixed_context, expected);
    if (banked_supported) {
        if (!setup_banked(&banked, target, a, b, transform)) {
            fprintf(stderr, "unable to prepare banked witness for %s\n", operation);
            return 0;
        }
        ok &= record_mv_backend(csv, options, operation, "banked_witness", banked_step,
            &banked, &banked.output, expected, native_tolerance());
    }
    return ok;
}

int main(int argc, char **argv) {
    workload_options_t options;
    uint32_t random_state;
    geo_cl20_t a;
    geo_cl20_t b;
    geo_cl20_t vector_a;
    geo_cl20_t vector_b;
    geo_cl20_t rotor_a;
    geo_cl20_t rotor_b;
    FILE *csv = NULL;
    int ok = 1;

    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (options.csv_path != NULL) {
        csv = fopen(options.csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "unable to open CSV output: %s\n", options.csv_path);
            return EXIT_FAILURE;
        }
        fputs("operation,backend,ns_per_op,max_absolute,max_relative,mismatches\n", csv);
    }

    random_state = options.seed;
    a = random_mv(&random_state);
    b = random_mv(&random_state);
    vector_a = random_vector(&random_state);
    vector_b = random_vector(&random_state);
    rotor_a = rotor_from_seed(&random_state);
    rotor_b = rotor_from_seed(&random_state);

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    const char *precision = "double";
#else
    const char *precision = "float";
#endif
    printf(
        "Geometric Elementary Operators shared workload harness\n"
        "configuration: precision=%s iterations=%" PRIu64
        " warmup=%" PRIu64 " seed=%" PRIu32 "\n",
        precision,
        options.iterations,
        options.warmup,
        options.seed
    );

    ok &= run_addition(csv, &options, a, b);
    ok &= run_dot(csv, &options, vector_a, vector_b);
    ok &= run_mv_family(csv, &options, "geometric_product",
        GEO_GEB_GEOMETRIC_PRODUCT, a, b, geo_cl20_zero(), 1, 1);
    ok &= run_mv_family(csv, &options, "reverse_product",
        GEO_GEB_REVERSE_PRODUCT, a, b, geo_cl20_zero(), 0, 1);
    ok &= run_mv_family(csv, &options, "rotor_action",
        GEO_GEB_ROTOR_ACTION, vector_a, geo_cl20_zero(), rotor_a, 1, 1);
    ok &= run_mv_family(csv, &options, "rotor_composition",
        GEO_GEB_ROTOR_COMPOSITION, rotor_a, rotor_b, geo_cl20_zero(), 0, 1);
    ok &= run_mv_family(csv, &options, "dilation",
        GEO_GEB_DILATION, vector_b, geo_cl20_zero(), rotor_b, 0, 1);

    if (csv != NULL) fclose(csv);
    printf("sink=%" PRIu64 "\n", workload_sink);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
