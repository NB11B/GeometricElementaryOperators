#include "geo/fixed_geb36.h"
#include "geo/geb36.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    uint8_t target;
    geo_fixed_result_kind_t kind;
    double tolerance_lsb;
} numerical_operation_t;

typedef struct {
    uint64_t samples;
    uint32_t seed;
    const char *csv_path;
} numerical_options_t;

typedef struct {
    uint64_t completed;
    uint64_t overflows;
    uint64_t status_failures;
    uint64_t kind_failures;
    uint64_t mismatches;
    double max_absolute;
    double max_relative;
    double max_angular;
    double max_projective_scale_error;
} numerical_stats_t;

#define OP(name_, id_, kind_, tolerance_) \
    {name_, id_, kind_, tolerance_}

/* Frozen GEB-36 order, expected fixed result kind, and operation error budget. */
static const numerical_operation_t OPERATIONS[] = {
    OP("zero", 1u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("one", 2u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("minus_one", 3u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("e1", 4u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("e2", 5u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("pseudoscalar", 6u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("negation", 7u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("reversion", 8u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("grade_involution", 9u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("clifford_conjugation", 10u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("scalar_projection", 11u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("vector_projection", 12u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("bivector_projection", 13u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("addition", 14u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("subtraction", 15u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("geometric_product", 16u, GEO_FIXED_RESULT_CL20, 2.0),
    OP("reverse_product", 17u, GEO_FIXED_RESULT_CL20, 2.0),
    OP("vector_dot", 18u, GEO_FIXED_RESULT_SCALAR, 1.0),
    OP("vector_wedge", 19u, GEO_FIXED_RESULT_CL20, 1.0),
    OP("commutator", 20u, GEO_FIXED_RESULT_CL20, 3.0),
    OP("anticommutator", 21u, GEO_FIXED_RESULT_CL20, 3.0),
    OP("vector_norm_squared", 22u, GEO_FIXED_RESULT_SCALAR, 1.0),
    OP("distance_squared", 23u, GEO_FIXED_RESULT_SCALAR, 1.0),
    OP("projection_numerator", 24u, GEO_FIXED_RESULT_PROJECTIVE, 1.0),
    OP("rejection_numerator", 25u, GEO_FIXED_RESULT_PROJECTIVE, 2.0),
    OP("reflection_numerator", 26u, GEO_FIXED_RESULT_PROJECTIVE, 3.0),
    OP("dual", 27u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("even_projection", 28u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("odd_projection", 29u, GEO_FIXED_RESULT_CL20, 0.0),
    OP("rotor_action", 30u, GEO_FIXED_RESULT_CL20, 4.0),
    OP("rotor_composition", 31u, GEO_FIXED_RESULT_CL20, 2.0),
    OP("rotor_norm_squared", 32u, GEO_FIXED_RESULT_SCALAR, 2.0),
    OP("dilation", 33u, GEO_FIXED_RESULT_CL20, 4.0),
    OP("translation_unipotent", 34u, GEO_FIXED_RESULT_UNIPOTENT, 0.0),
    /* Denominator rounding is amplified by the reciprocal; strong vectors
     * bound the observed sensitivity below this explicit 48-LSB envelope. */
    OP("vector_inverse_projective", 35u, GEO_FIXED_RESULT_PROJECTIVE, 48.0),
    OP("angle_cosine_numerator", 36u, GEO_FIXED_RESULT_SCALAR, 1.0)
};

#undef OP

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

static geo_fixed_t signed_magnitude(uint32_t *state, int32_t low, int32_t high) {
    const uint32_t draw = next_random(state);
    const uint32_t span = (uint32_t)(high - low + 1);
    const int32_t magnitude = low +
        (int32_t)((draw & UINT32_C(0x7fffffff)) % span);
    return (draw & UINT32_C(0x80000000)) != 0u ? magnitude : -magnitude;
}

static geo_fixed_t general_component(uint32_t *state) {
#if GEO_FIXED_FRACTION_BITS == 1
    return signed_magnitude(state, 1, 1);
#else
    const int32_t scale = (int32_t)(INT32_C(1) << GEO_FIXED_FRACTION_BITS);
    return signed_magnitude(state, 1, scale / 4);
#endif
}

static geo_fixed_cl20_t general_mv(uint32_t *state) {
    geo_fixed_cl20_t value;
    value.scalar = general_component(state);
    value.e1 = general_component(state);
    value.e2 = general_component(state);
    value.e12 = general_component(state);
    return value;
}

static geo_fixed_cl20_t vector_mv(uint32_t *state, int strong) {
    geo_fixed_cl20_t value = {0, 0, 0, 0};
#if GEO_FIXED_FRACTION_BITS == 1
    (void)strong;
    value.e1 = signed_magnitude(state, 1, 1);
    value.e2 = signed_magnitude(state, 1, 1);
#else
    const int32_t scale = (int32_t)(INT32_C(1) << GEO_FIXED_FRACTION_BITS);
    const int32_t low = strong ? 3 * (scale / 16) : scale / 16;
    value.e1 = signed_magnitude(state, low, scale / 4);
    value.e2 = signed_magnitude(state, low, scale / 4);
#endif
    return value;
}

static geo_fixed_cl20_t rotor_mv(uint32_t *state) {
    geo_fixed_cl20_t value = {0, 0, 0, 0};
#if GEO_FIXED_FRACTION_BITS == 1
    value.scalar = signed_magnitude(state, 1, 1);
    value.e12 = signed_magnitude(state, 1, 1);
#else
    const int32_t scale = (int32_t)(INT32_C(1) << GEO_FIXED_FRACTION_BITS);
    value.scalar = signed_magnitude(state, 3 * (scale / 16), scale / 4);
    value.e12 = signed_magnitude(state, scale / 16, scale / 8);
#endif
    return value;
}

static int cl20_raw_equal(geo_fixed_cl20_t left, geo_fixed_cl20_t right) {
    return left.scalar == right.scalar && left.e1 == right.e1 &&
        left.e2 == right.e2 && left.e12 == right.e12;
}

static void observe_component(geo_fixed_t value, int *positive, int *negative) {
    if (value > 0) *positive = 1;
    if (value < 0) *negative = 1;
}

static int has_mixed_signs(geo_fixed_cl20_t value) {
    int positive = 0;
    int negative = 0;
    observe_component(value.scalar, &positive, &negative);
    observe_component(value.e1, &positive, &negative);
    observe_component(value.e2, &positive, &negative);
    observe_component(value.e12, &positive, &negative);
    return positive && negative;
}

static int vector_cross_is_nonzero(
    geo_fixed_cl20_t left,
    geo_fixed_cl20_t right,
    int *nonzero
) {
    const int64_t left_e1 = (int64_t)left.e1;
    const int64_t left_e2 = (int64_t)left.e2;
    const int64_t right_e1 = (int64_t)right.e1;
    const int64_t right_e2 = (int64_t)right.e2;
    const int64_t bound = INT64_C(1) << 28;
    const int64_t absolute_left_e1 = left_e1 < 0 ? -left_e1 : left_e1;
    const int64_t absolute_left_e2 = left_e2 < 0 ? -left_e2 : left_e2;
    const int64_t absolute_right_e1 = right_e1 < 0 ? -right_e1 : right_e1;
    const int64_t absolute_right_e2 = right_e2 < 0 ? -right_e2 : right_e2;
    int64_t cross;

    if (nonzero == NULL || absolute_left_e1 > bound || absolute_left_e2 > bound ||
        absolute_right_e1 > bound || absolute_right_e2 > bound) return 0;
    /* Fixtures are bounded to 2^28 at Q30, so each product is at most 2^56
     * and their difference is at most 2^57: both are exactly representable. */
    cross = left_e1 * right_e2 - left_e2 * right_e1;
    *nonzero = cross != 0;
    return 1;
}

static int fixture_diversity_is_valid(uint32_t seed) {
    uint32_t state = seed;
    int positive = 0;
    int negative = 0;
    int general_operands_differ = 0;
    int vector_operands_differ = 0;
    int nonparallel_vectors = 0;
    int mixed_general = 0;
    unsigned int sample;

    for (sample = 0u; sample < 64u; ++sample) {
        const geo_fixed_cl20_t general_a = general_mv(&state);
        const geo_fixed_cl20_t general_b = general_mv(&state);
        const geo_fixed_cl20_t vector_a = vector_mv(&state, 0);
        const geo_fixed_cl20_t vector_b = vector_mv(&state, 0);
        int cross_is_nonzero = 0;

        observe_component(general_a.scalar, &positive, &negative);
        observe_component(general_a.e1, &positive, &negative);
        observe_component(general_a.e2, &positive, &negative);
        observe_component(general_a.e12, &positive, &negative);
        observe_component(general_b.scalar, &positive, &negative);
        observe_component(general_b.e1, &positive, &negative);
        observe_component(general_b.e2, &positive, &negative);
        observe_component(general_b.e12, &positive, &negative);
        observe_component(vector_a.e1, &positive, &negative);
        observe_component(vector_a.e2, &positive, &negative);
        observe_component(vector_b.e1, &positive, &negative);
        observe_component(vector_b.e2, &positive, &negative);

        if (!cl20_raw_equal(general_a, general_b)) general_operands_differ = 1;
        if (!cl20_raw_equal(vector_a, vector_b)) vector_operands_differ = 1;
        if (has_mixed_signs(general_a) || has_mixed_signs(general_b)) {
            mixed_general = 1;
        }
        if (!vector_cross_is_nonzero(vector_a, vector_b, &cross_is_nonzero)) {
            return 0;
        }
        if (cross_is_nonzero) nonparallel_vectors = 1;
    }
    return positive && negative && general_operands_differ &&
        vector_operands_differ && nonparallel_vectors && mixed_general;
}

static void fixtures(
    uint8_t target,
    uint32_t *state,
    geo_fixed_cl20_t *a,
    geo_fixed_cl20_t *b,
    geo_fixed_cl20_t *transform
) {
    *a = general_mv(state);
    *b = general_mv(state);
    *transform = rotor_mv(state);
    switch ((geo_geb_target_id_t)target) {
        case GEO_GEB_VECTOR_DOT:
        case GEO_GEB_VECTOR_WEDGE:
        case GEO_GEB_VECTOR_NORM_SQUARED:
        case GEO_GEB_DISTANCE_SQUARED:
        case GEO_GEB_PROJECTION_NUMERATOR:
        case GEO_GEB_REJECTION_NUMERATOR:
        case GEO_GEB_REFLECTION_NUMERATOR:
        case GEO_GEB_ANGLE_COSINE_NUMERATOR:
            *a = vector_mv(state, 0);
            *b = vector_mv(state, 0);
            break;
        case GEO_GEB_ROTOR_ACTION:
        case GEO_GEB_DILATION:
            *a = vector_mv(state, 0);
            break;
        case GEO_GEB_ROTOR_COMPOSITION:
        case GEO_GEB_ROTOR_NORM_SQUARED:
            *a = rotor_mv(state);
            *b = rotor_mv(state);
            break;
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE:
            *a = vector_mv(state, 1);
            break;
        default:
            break;
    }
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
    uint8_t target,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform,
    geo_cl20_t *mv,
    geo_real_t *scalar
) {
    if (mv == NULL || scalar == NULL) return 0;
    switch ((geo_geb_target_id_t)target) {
        case GEO_GEB_ZERO: *mv = geo_geb_zero(); break;
        case GEO_GEB_ONE: *mv = geo_geb_one(); break;
        case GEO_GEB_MINUS_ONE: *mv = geo_geb_minus_one(); break;
        case GEO_GEB_E1: *mv = geo_geb_e1(); break;
        case GEO_GEB_E2: *mv = geo_geb_e2(); break;
        case GEO_GEB_PSEUDOSCALAR: *mv = geo_geb_pseudoscalar(); break;
        case GEO_GEB_NEGATION: *mv = geo_geb_negation(a); break;
        case GEO_GEB_REVERSION: *mv = geo_geb_reversion(a); break;
        case GEO_GEB_GRADE_INVOLUTION: *mv = geo_geb_grade_involution(a); break;
        case GEO_GEB_CLIFFORD_CONJUGATION: *mv = geo_geb_clifford_conjugation(a); break;
        case GEO_GEB_SCALAR_PROJECTION: *mv = geo_geb_scalar_projection(a); break;
        case GEO_GEB_VECTOR_PROJECTION: *mv = geo_geb_vector_projection(a); break;
        case GEO_GEB_BIVECTOR_PROJECTION: *mv = geo_geb_bivector_projection(a); break;
        case GEO_GEB_ADDITION: *mv = geo_geb_addition(a, b); break;
        case GEO_GEB_SUBTRACTION: *mv = geo_geb_subtraction(a, b); break;
        case GEO_GEB_GEOMETRIC_PRODUCT: *mv = geo_geb_geometric_product(a, b); break;
        case GEO_GEB_REVERSE_PRODUCT: *mv = geo_geb_reverse_product(a, b); break;
        case GEO_GEB_VECTOR_DOT: *scalar = geo_geb_vector_dot(a, b); break;
        case GEO_GEB_VECTOR_WEDGE: *mv = geo_geb_vector_wedge(a, b); break;
        case GEO_GEB_COMMUTATOR: *mv = geo_geb_commutator(a, b); break;
        case GEO_GEB_ANTICOMMUTATOR: *mv = geo_geb_anticommutator(a, b); break;
        case GEO_GEB_VECTOR_NORM_SQUARED: *scalar = geo_geb_vector_norm_squared(a); break;
        case GEO_GEB_DISTANCE_SQUARED: *scalar = geo_geb_distance_squared(a, b); break;
        case GEO_GEB_PROJECTION_NUMERATOR: *mv = geo_geb_projection_numerator(a, b); break;
        case GEO_GEB_REJECTION_NUMERATOR: *mv = geo_geb_rejection_numerator(a, b); break;
        case GEO_GEB_REFLECTION_NUMERATOR: *mv = geo_geb_reflection_numerator(a, b); break;
        case GEO_GEB_DUAL: *mv = geo_geb_dual(a); break;
        case GEO_GEB_EVEN_PROJECTION: *mv = geo_geb_even_projection(a); break;
        case GEO_GEB_ODD_PROJECTION: *mv = geo_geb_odd_projection(a); break;
        case GEO_GEB_ROTOR_ACTION: *mv = geo_geb_rotor_action(transform, a); break;
        case GEO_GEB_ROTOR_COMPOSITION: *mv = geo_geb_rotor_composition(a, b); break;
        case GEO_GEB_ROTOR_NORM_SQUARED: *scalar = geo_geb_rotor_norm_squared(a); break;
        case GEO_GEB_DILATION: *mv = geo_geb_dilation(transform, a); break;
        case GEO_GEB_TRANSLATION_UNIPOTENT:
            *mv = geo_geb_translation_unipotent(a).payload;
            break;
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE: {
            const geo_cl20_t represented = geo_geb_vector_inverse_projective(a);
            const geo_real_t denominator = geo_geb_vector_norm_squared(a);
            if (denominator == (geo_real_t)0) return 0;
            *mv = geo_cl20_scale(represented, (geo_real_t)1 / denominator);
            break;
        }
        case GEO_GEB_ANGLE_COSINE_NUMERATOR:
            *scalar = geo_geb_angle_cosine_numerator(a, b);
            break;
        default: return 0;
    }
    return 1;
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
    geo_fixed_result_kind_t expected_kind,
    geo_cl20_t expected_mv,
    geo_cl20_t *actual_mv,
    geo_real_t *actual_scalar,
    double *scale_error
) {
    if (result == NULL || actual_mv == NULL || actual_scalar == NULL ||
        scale_error == NULL || result->kind != (uint8_t)expected_kind) return 0;
    *scale_error = 0.0;
    switch (expected_kind) {
        case GEO_FIXED_RESULT_SCALAR:
            *actual_scalar = (geo_real_t)geo_fixed_to_double(result->as.scalar);
            return 1;
        case GEO_FIXED_RESULT_CL20:
            *actual_mv = from_fixed(result->as.cl20);
            return 1;
        case GEO_FIXED_RESULT_UNIPOTENT:
            *actual_mv = from_fixed(result->as.unipotent_payload);
            return 1;
        case GEO_FIXED_RESULT_PROJECTIVE: {
            const double denominator =
                geo_fixed_to_double(result->as.projective.denominator);
            const geo_cl20_t represented =
                from_fixed(result->as.projective.represented);
            if (denominator == 0.0 || !isfinite(denominator)) return 0;
            *actual_mv = geo_cl20_make(
                (geo_real_t)(component(represented, 0u) / denominator),
                (geo_real_t)(component(represented, 1u) / denominator),
                (geo_real_t)(component(represented, 2u) / denominator),
                (geo_real_t)(component(represented, 3u) / denominator)
            );
            *scale_error = fabs(fitted_scale(represented, expected_mv) - denominator);
            return 1;
        }
        default: return 0;
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
    return !isfinite(actual) || !isfinite(expected) || absolute > tolerance;
}

static double angular_error(geo_cl20_t actual, geo_cl20_t expected) {
    const double actual_norm = hypot((double)actual.e1, (double)actual.e2);
    const double expected_norm = hypot((double)expected.e1, (double)expected.e2);
    double cosine;
    if (actual_norm == 0.0 && expected_norm == 0.0) return 0.0;
    if (actual_norm == 0.0 || expected_norm == 0.0) return 3.14159265358979323846;
    cosine = ((double)actual.e1 * (double)expected.e1 +
        (double)actual.e2 * (double)expected.e2) / (actual_norm * expected_norm);
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
    const double lsb = 1.0 / (double)(INT64_C(1) << GEO_FIXED_FRACTION_BITS);
    const double tolerance = operation->tolerance_lsb * lsb;
    uint64_t sample;
    memset(stats, 0, sizeof(*stats));
    for (sample = 0u; sample < sample_count; ++sample) {
        geo_fixed_cl20_t fixed_a;
        geo_fixed_cl20_t fixed_b;
        geo_fixed_cl20_t fixed_transform;
        geo_cl20_t a;
        geo_cl20_t b;
        geo_cl20_t transform;
        geo_cl20_t expected_mv = geo_cl20_zero();
        geo_cl20_t actual_mv = geo_cl20_zero();
        geo_real_t expected_scalar = (geo_real_t)0;
        geo_real_t actual_scalar = (geo_real_t)0;
        geo_fixed_geb_result_t result;
        geo_fixed_status_t status;
        double scale_error = 0.0;
        int mismatch = 0;
        size_t index;

        fixtures(operation->target, random_state, &fixed_a, &fixed_b, &fixed_transform);
        a = from_fixed(fixed_a);
        b = from_fixed(fixed_b);
        transform = from_fixed(fixed_transform);
        if (!reference_result(operation->target, a, b, transform,
                &expected_mv, &expected_scalar)) return 0;
        memset(&result, 0xa5, sizeof(result));
        status = geo_fixed_geb36_execute(operation->target, fixed_a, fixed_b,
            fixed_transform, &result);
        if (status == GEO_FIXED_OVERFLOW) {
            ++stats->overflows;
            continue;
        }
        if (status != GEO_FIXED_OK) {
            ++stats->status_failures;
            continue;
        }
        if (result.kind != (uint8_t)operation->kind) {
            ++stats->kind_failures;
            continue;
        }
        if (!normalize_result(&result, operation->kind, expected_mv,
                &actual_mv, &actual_scalar, &scale_error)) {
            ++stats->kind_failures;
            continue;
        }

        ++stats->completed;
        if (operation->kind == GEO_FIXED_RESULT_SCALAR) {
            mismatch = update_error(stats, (double)actual_scalar,
                (double)expected_scalar, tolerance);
        } else {
            const double angle = angular_error(actual_mv, expected_mv);
            for (index = 0u; index < 4u; ++index) {
                mismatch |= update_error(stats, component(actual_mv, index),
                    component(expected_mv, index), tolerance);
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

static const char *kind_name(geo_fixed_result_kind_t kind) {
    switch (kind) {
        case GEO_FIXED_RESULT_CL20: return "cl20";
        case GEO_FIXED_RESULT_SCALAR: return "scalar";
        case GEO_FIXED_RESULT_PROJECTIVE: return "projective";
        case GEO_FIXED_RESULT_UNIPOTENT: return "unipotent";
        default: return "invalid";
    }
}

static void report(FILE *csv, const numerical_operation_t *operation,
    uint64_t requested, const numerical_stats_t *stats) {
    printf(
        "ERROR operation=%s backend=fixed_geb36 expected_kind=%s requested=%" PRIu64
        " completed=%" PRIu64 " overflows=%" PRIu64
        " status_failures=%" PRIu64 " kind_failures=%" PRIu64
        " max_absolute=%.17g max_relative=%.17g max_angular=%.17g"
        " max_projective_scale=%.17g mismatches=%" PRIu64 "\n",
        operation->name, kind_name(operation->kind), requested, stats->completed,
        stats->overflows, stats->status_failures, stats->kind_failures,
        stats->max_absolute, stats->max_relative, stats->max_angular,
        stats->max_projective_scale_error, stats->mismatches);
    if (csv != NULL) {
        fprintf(csv,
            "%s,fixed_geb36,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%" PRIu64 "\n",
            operation->name, kind_name(operation->kind), requested,
            stats->completed, stats->overflows, stats->status_failures,
            stats->kind_failures, stats->max_absolute, stats->max_relative,
            stats->max_angular, stats->max_projective_scale_error,
            stats->mismatches);
    }
}

int main(int argc, char **argv) {
    numerical_options_t options;
    uint32_t random_state;
    FILE *csv = NULL;
    size_t operation_index;
    uint64_t total_failures = 0u;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!fixture_diversity_is_valid(options.seed)) {
        fputs("deterministic fixture-diversity validation failed\n", stderr);
        return EXIT_FAILURE;
    }
    if (options.csv_path != NULL) {
        csv = fopen(options.csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "unable to open CSV output: %s\n", options.csv_path);
            return EXIT_FAILURE;
        }
        fputs("operation,backend,expected_kind,requested,completed,overflows,"
            "status_failures,kind_failures,max_absolute,max_relative,max_angular,"
            "max_projective_scale,mismatches\n", csv);
    }
    random_state = options.seed;
    printf("Geometric Elementary Operators fixed GEB-36 typed numerical envelope\n"
        "samples=%" PRIu64 " seed=%" PRIu32 " precision=%s q_fraction_bits=%d\n",
        options.samples, options.seed,
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
        "double",
#else
        "float",
#endif
        GEO_FIXED_FRACTION_BITS);
    for (operation_index = 0u;
         operation_index < sizeof(OPERATIONS) / sizeof(OPERATIONS[0]);
         ++operation_index) {
        numerical_stats_t stats;
        if (!run_operation(&OPERATIONS[operation_index], options.samples,
                &random_state, &stats)) {
            fprintf(stderr, "unable to evaluate %s\n", OPERATIONS[operation_index].name);
            if (csv != NULL) fclose(csv);
            return EXIT_FAILURE;
        }
        report(csv, &OPERATIONS[operation_index], options.samples, &stats);
        total_failures += stats.overflows + stats.status_failures +
            stats.kind_failures + stats.mismatches;
    }
    if (csv != NULL) fclose(csv);
    return total_failures == 0u ? EXIT_SUCCESS : EXIT_FAILURE;
}
