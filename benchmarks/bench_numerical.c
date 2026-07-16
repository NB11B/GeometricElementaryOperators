#include "geo/fixed_geb36.h"
#include "geo/geb36.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    NUMERICAL_MULTIVECTOR = 0,
    NUMERICAL_SCALAR = 1
} numerical_kind_t;

typedef struct {
    const char *name;
    uint8_t target;
    numerical_kind_t kind;
} numerical_operation_t;

typedef struct {
    uint64_t samples;
    uint32_t seed;
    const char *csv_path;
} numerical_options_t;

typedef struct {
    uint64_t completed;
    uint64_t overflows;
    uint64_t mismatches;
    double max_absolute;
    double max_relative;
    double max_angular;
    double max_projective_scale_error;
} numerical_stats_t;

static const numerical_operation_t OPERATIONS[] = {
    {"addition", GEO_GEB_ADDITION, NUMERICAL_MULTIVECTOR},
    {"geometric_product", GEO_GEB_GEOMETRIC_PRODUCT, NUMERICAL_MULTIVECTOR},
    {"reverse_product", GEO_GEB_REVERSE_PRODUCT, NUMERICAL_MULTIVECTOR},
    {"vector_dot", GEO_GEB_VECTOR_DOT, NUMERICAL_SCALAR},
    {"vector_wedge", GEO_GEB_VECTOR_WEDGE, NUMERICAL_MULTIVECTOR},
    {"rotor_action", GEO_GEB_ROTOR_ACTION, NUMERICAL_MULTIVECTOR},
    {"rotor_composition", GEO_GEB_ROTOR_COMPOSITION, NUMERICAL_MULTIVECTOR},
    {"dilation", GEO_GEB_DILATION, NUMERICAL_MULTIVECTOR},
    {"vector_inverse_projective", GEO_GEB_VECTOR_INVERSE_PROJECTIVE,
        NUMERICAL_MULTIVECTOR}
};

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

static void usage(const char *program) {
    printf("Usage: %s [--samples N] [--seed N] [--csv PATH]\n", program);
}

static int parse_options(int argc, char **argv, numerical_options_t *options) {
    int index;
    if (options == NULL) return 0;
    options->samples = UINT64_C(10000);
    options->seed = UINT32_C(0x6a09e667);
    options->csv_path = NULL;

    for (index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        uint64_t parsed;
        if (strcmp(argument, "--help") == 0 || strcmp(argument, "-h") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc) return 0;
        ++index;
        if (strcmp(argument, "--samples") == 0) {
            if (!parse_u64(argv[index], &parsed) || parsed == 0u) return 0;
            options->samples = parsed;
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

static uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static geo_real_t random_real(uint32_t *state) {
    const uint32_t bits = next_random(state) >> 8;
    const double unit = (double)bits / (double)UINT32_C(0x00ffffff);
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
    geo_cl20_t value = geo_cl20_make(
        (geo_real_t)0,
        random_real(state),
        random_real(state),
        (geo_real_t)0
    );
    if (fabs((double)value.e1) + fabs((double)value.e2) < 0.125) {
        value.e1 = (geo_real_t)0.25;
        value.e2 = (geo_real_t)-0.125;
    }
    return value;
}

static geo_cl20_t random_rotor(uint32_t *state) {
    const double unit =
        (double)(next_random(state) & UINT32_C(0xffff)) / 65535.0;
    const double angle = 0.05 + 0.45 * unit;
    return geo_cl20_make(
        (geo_real_t)cos(angle),
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)-sin(angle)
    );
}

static void fixtures(
    uint8_t target,
    uint32_t *state,
    geo_cl20_t *a,
    geo_cl20_t *b,
    geo_cl20_t *transform
) {
    *a = random_mv(state);
    *b = random_mv(state);
    *transform = random_rotor(state);

    switch ((geo_geb_target_id_t)target) {
        case GEO_GEB_VECTOR_DOT:
        case GEO_GEB_VECTOR_WEDGE:
            *a = random_vector(state);
            *b = random_vector(state);
            break;
        case GEO_GEB_ROTOR_ACTION:
        case GEO_GEB_DILATION:
            *a = random_vector(state);
            *b = geo_cl20_zero();
            break;
        case GEO_GEB_ROTOR_COMPOSITION:
            *a = random_rotor(state);
            *b = random_rotor(state);
            *transform = geo_cl20_zero();
            break;
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE:
            *a = random_vector(state);
            *b = geo_cl20_zero();
            *transform = geo_cl20_zero();
            break;
        default:
            break;
    }
}

static int to_fixed(geo_cl20_t input, geo_fixed_cl20_t *output) {
    if (output == NULL) return 0;
    return geo_fixed_from_double((double)input.scalar, &output->scalar) == GEO_FIXED_OK &&
        geo_fixed_from_double((double)input.e1, &output->e1) == GEO_FIXED_OK &&
        geo_fixed_from_double((double)input.e2, &output->e2) == GEO_FIXED_OK &&
        geo_fixed_from_double((double)input.e12, &output->e12) == GEO_FIXED_OK;
}

static geo_cl20_t from_fixed(geo_fixed_cl20_t input) {
    return geo_cl20_make(
        (geo_real_t)geo_fixed_to_double(input.scalar),
        (geo_real_t)geo_fixed_to_double(input.e1),
        (geo_real_t)geo_fixed_to_double(input.e2),
        (geo_real_t)geo_fixed_to_double(input.e12)
    );
}

static int reference_result(
    const numerical_operation_t *operation,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform,
    geo_cl20_t *mv,
    geo_real_t *scalar
) {
    if (operation == NULL || mv == NULL || scalar == NULL) return 0;
    switch ((geo_geb_target_id_t)operation->target) {
        case GEO_GEB_ADDITION:
            *mv = geo_geb_addition(a, b);
            return 1;
        case GEO_GEB_GEOMETRIC_PRODUCT:
            *mv = geo_geb_geometric_product(a, b);
            return 1;
        case GEO_GEB_REVERSE_PRODUCT:
            *mv = geo_geb_reverse_product(a, b);
            return 1;
        case GEO_GEB_VECTOR_DOT:
            *scalar = geo_geb_vector_dot(a, b);
            return 1;
        case GEO_GEB_VECTOR_WEDGE:
            *mv = geo_geb_vector_wedge(a, b);
            return 1;
        case GEO_GEB_ROTOR_ACTION:
            *mv = geo_geb_rotor_action(transform, a);
            return 1;
        case GEO_GEB_ROTOR_COMPOSITION:
            *mv = geo_geb_rotor_composition(a, b);
            return 1;
        case GEO_GEB_DILATION:
            *mv = geo_geb_dilation(transform, a);
            return 1;
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE: {
            const geo_real_t norm = geo_cl20_vector_norm_squared(a);
            if (norm == (geo_real_t)0) return 0;
            *mv = geo_cl20_scale(a, (geo_real_t)1 / norm);
            return 1;
        }
        default:
            return 0;
    }
}

static double component(geo_cl20_t value, size_t index) {
    switch (index) {
        case 0u: return (double)value.scalar;
        case 1u: return (double)value.e1;
        case 2u: return (double)value.e2;
        default: return (double)value.e12;
    }
}

static double fitted_scale(geo_cl20_t represented, geo_cl20_t canonical) {
    double numerator = 0.0;
    double denominator = 0.0;
    size_t index;
    for (index = 0u; index < 4u; ++index) {
        const double r = component(represented, index);
        const double c = component(canonical, index);
        numerator += r * c;
        denominator += c * c;
    }
    if (denominator == 0.0) return numerator == 0.0 ? 1.0 : INFINITY;
    return numerator / denominator;
}

static int normalize_result(
    const geo_fixed_geb_result_t *result,
    numerical_kind_t expected_kind,
    geo_cl20_t expected_mv,
    geo_cl20_t *actual_mv,
    geo_real_t *actual_scalar,
    double *scale_error
) {
    geo_fixed_cl20_t normalized;
    geo_fixed_status_t status;

    if (result == NULL || actual_mv == NULL ||
        actual_scalar == NULL || scale_error == NULL) {
        return 0;
    }
    *scale_error = 0.0;

    if (expected_kind == NUMERICAL_SCALAR) {
        if (result->kind != (uint8_t)GEO_FIXED_RESULT_SCALAR) return 0;
        *actual_scalar = (geo_real_t)geo_fixed_to_double(result->as.scalar);
        return 1;
    }

    switch ((geo_fixed_result_kind_t)result->kind) {
        case GEO_FIXED_RESULT_CL20:
            *actual_mv = from_fixed(result->as.cl20);
            return 1;
        case GEO_FIXED_RESULT_UNIPOTENT:
            *actual_mv = from_fixed(result->as.unipotent_payload);
            return 1;
        case GEO_FIXED_RESULT_PROJECTIVE: {
            const geo_fixed_t denominator = result->as.projective.denominator;
            const double denominator_real = geo_fixed_to_double(denominator);
            geo_cl20_t represented;
            if (denominator == 0) return 0;
            status = geo_fixed_div(
                result->as.projective.represented.scalar,
                denominator,
                &normalized.scalar
            );
            if (status != GEO_FIXED_OK) return 0;
            status = geo_fixed_div(
                result->as.projective.represented.e1,
                denominator,
                &normalized.e1
            );
            if (status != GEO_FIXED_OK) return 0;
            status = geo_fixed_div(
                result->as.projective.represented.e2,
                denominator,
                &normalized.e2
            );
            if (status != GEO_FIXED_OK) return 0;
            status = geo_fixed_div(
                result->as.projective.represented.e12,
                denominator,
                &normalized.e12
            );
            if (status != GEO_FIXED_OK) return 0;
            *actual_mv = from_fixed(normalized);
            represented = from_fixed(result->as.projective.represented);
            *scale_error = fabs(
                fitted_scale(represented, expected_mv) - denominator_real
            );
            return 1;
        }
        default:
            return 0;
    }
}

static int update_error(
    numerical_stats_t *stats,
    double actual,
    double expected,
    double tolerance
) {
    const double absolute = fabs(actual - expected);
    const double relative = absolute / fmax(fabs(expected), 1.0e-30);
    if (absolute > stats->max_absolute) stats->max_absolute = absolute;
    if (relative > stats->max_relative) stats->max_relative = relative;
    return !isfinite(actual) || !isfinite(expected) ||
        absolute > tolerance + tolerance * fabs(expected);
}

static double angular_error(geo_cl20_t actual, geo_cl20_t expected) {
    const double actual_norm = hypot((double)actual.e1, (double)actual.e2);
    const double expected_norm = hypot((double)expected.e1, (double)expected.e2);
    double cosine;
    if (actual_norm == 0.0 && expected_norm == 0.0) return 0.0;
    if (actual_norm == 0.0 || expected_norm == 0.0) {
        return 3.14159265358979323846;
    }
    cosine = ((double)actual.e1 * (double)expected.e1 +
        (double)actual.e2 * (double)expected.e2) /
        (actual_norm * expected_norm);
    if (cosine > 1.0) cosine = 1.0;
    if (cosine < -1.0) cosine = -1.0;
    return acos(cosine);
}

static int run_operation(
    const numerical_operation_t *operation,
    uint64_t sample_count,
    uint32_t *random_state,
    numerical_stats_t *stats
) {
    const double tolerance =
        512.0 / (double)(INT64_C(1) << GEO_FIXED_FRACTION_BITS);
    uint64_t sample;

    memset(stats, 0, sizeof(*stats));
    for (sample = 0u; sample < sample_count; ++sample) {
        geo_cl20_t a;
        geo_cl20_t b;
        geo_cl20_t transform;
        geo_cl20_t expected_mv = geo_cl20_zero();
        geo_cl20_t actual_mv = geo_cl20_zero();
        geo_real_t expected_scalar = (geo_real_t)0;
        geo_real_t actual_scalar = (geo_real_t)0;
        geo_fixed_cl20_t fixed_a;
        geo_fixed_cl20_t fixed_b;
        geo_fixed_cl20_t fixed_transform;
        geo_fixed_geb_result_t result;
        geo_fixed_status_t status;
        double scale_error = 0.0;
        int mismatch = 0;
        size_t index;

        fixtures(operation->target, random_state, &a, &b, &transform);
        if (!reference_result(
                operation, a, b, transform, &expected_mv, &expected_scalar)) {
            return 0;
        }
        if (!to_fixed(a, &fixed_a) ||
            !to_fixed(b, &fixed_b) ||
            !to_fixed(transform, &fixed_transform)) {
            return 0;
        }

        status = geo_fixed_geb36_execute(
            operation->target,
            fixed_a,
            fixed_b,
            fixed_transform,
            &result
        );
        if (status == GEO_FIXED_OVERFLOW) {
            ++stats->overflows;
            continue;
        }
        if (status != GEO_FIXED_OK) return 0;
        if (!normalize_result(
                &result,
                operation->kind,
                expected_mv,
                &actual_mv,
                &actual_scalar,
                &scale_error)) {
            return 0;
        }

        ++stats->completed;
        if (operation->kind == NUMERICAL_SCALAR) {
            mismatch = update_error(
                stats,
                (double)actual_scalar,
                (double)expected_scalar,
                tolerance
            );
        } else {
            const double angle = angular_error(actual_mv, expected_mv);
            for (index = 0u; index < 4u; ++index) {
                mismatch |= update_error(
                    stats,
                    component(actual_mv, index),
                    component(expected_mv, index),
                    tolerance
                );
            }
            if (angle > stats->max_angular) stats->max_angular = angle;
            if (scale_error > stats->max_projective_scale_error) {
                stats->max_projective_scale_error = scale_error;
            }
        }
        if (mismatch) ++stats->mismatches;
    }
    return 1;
}

static void report(
    FILE *csv,
    const numerical_operation_t *operation,
    uint64_t requested,
    const numerical_stats_t *stats
) {
    printf(
        "ERROR operation=%s backend=fixed_geb36 requested=%" PRIu64
        " completed=%" PRIu64 " overflows=%" PRIu64
        " max_absolute=%.17g max_relative=%.17g max_angular=%.17g"
        " max_projective_scale=%.17g mismatches=%" PRIu64 "\n",
        operation->name,
        requested,
        stats->completed,
        stats->overflows,
        stats->max_absolute,
        stats->max_relative,
        stats->max_angular,
        stats->max_projective_scale_error,
        stats->mismatches
    );
    if (csv != NULL) {
        fprintf(
            csv,
            "%s,fixed_geb36,%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%.17g,%.17g,%.17g,%.17g,%" PRIu64 "\n",
            operation->name,
            requested,
            stats->completed,
            stats->overflows,
            stats->max_absolute,
            stats->max_relative,
            stats->max_angular,
            stats->max_projective_scale_error,
            stats->mismatches
        );
    }
}

int main(int argc, char **argv) {
    numerical_options_t options;
    uint32_t random_state;
    FILE *csv = NULL;
    size_t operation_index;
    uint64_t total_mismatches = 0u;

    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (options.csv_path != NULL) {
        csv = fopen(options.csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "unable to open CSV output: %s\n", options.csv_path);
            return EXIT_FAILURE;
        }
        fputs(
            "operation,backend,requested,completed,overflows,max_absolute,"
            "max_relative,max_angular,max_projective_scale,mismatches\n",
            csv
        );
    }

    random_state = options.seed;
    printf(
        "Geometric Elementary Operators numerical envelope\n"
        "samples=%" PRIu64 " seed=%" PRIu32 " q_fraction_bits=%d\n",
        options.samples,
        options.seed,
        GEO_FIXED_FRACTION_BITS
    );

    for (operation_index = 0u;
         operation_index < sizeof(OPERATIONS) / sizeof(OPERATIONS[0]);
         ++operation_index) {
        numerical_stats_t stats;
        if (!run_operation(
                &OPERATIONS[operation_index],
                options.samples,
                &random_state,
                &stats)) {
            fprintf(
                stderr,
                "unable to evaluate %s\n",
                OPERATIONS[operation_index].name
            );
            if (csv != NULL) fclose(csv);
            return EXIT_FAILURE;
        }
        report(csv, &OPERATIONS[operation_index], options.samples, &stats);
        total_mismatches += stats.mismatches;
    }

    if (csv != NULL) fclose(csv);
    return total_mismatches == 0u ? EXIT_SUCCESS : EXIT_FAILURE;
}
