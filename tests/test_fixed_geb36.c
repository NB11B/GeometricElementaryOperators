#include "geo/fixed_geb36.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static geo_fixed_cl20_t encode(geo_cl20_t value) {
    geo_fixed_cl20_t result;
    if (geo_fixed_from_double((double)value.scalar, &result.scalar) != GEO_FIXED_OK ||
        geo_fixed_from_double((double)value.e1, &result.e1) != GEO_FIXED_OK ||
        geo_fixed_from_double((double)value.e2, &result.e2) != GEO_FIXED_OK ||
        geo_fixed_from_double((double)value.e12, &result.e12) != GEO_FIXED_OK) {
        fprintf(stderr, "encode failure\n");
        exit(EXIT_FAILURE);
    }
    return result;
}

static geo_cl20_t decode(geo_fixed_cl20_t value) {
    return geo_cl20_make(
        (geo_real_t)geo_fixed_to_double(value.scalar),
        (geo_real_t)geo_fixed_to_double(value.e1),
        (geo_real_t)geo_fixed_to_double(value.e2),
        (geo_real_t)geo_fixed_to_double(value.e12)
    );
}

static void fail_id(uint8_t id, const char *message) {
    ++failures;
    fprintf(stderr, "FAIL target %u: %s\n", (unsigned)id, message);
}

static int near_scalar(geo_fixed_t actual, geo_real_t expected) {
    return fabs(geo_fixed_to_double(actual) - (double)expected) <= 5e-3;
}

static int near_mv(geo_fixed_cl20_t actual, geo_cl20_t expected) {
    return geo_cl20_near(decode(actual), expected, (geo_real_t)5e-3);
}

static geo_cl20_t expected_cl20(
    uint8_t id,
    geo_cl20_t a,
    geo_cl20_t b,
    geo_cl20_t transform
) {
    switch ((geo_geb_target_id_t)id) {
        case GEO_GEB_ZERO: return geo_geb_zero();
        case GEO_GEB_ONE: return geo_geb_one();
        case GEO_GEB_MINUS_ONE: return geo_geb_minus_one();
        case GEO_GEB_E1: return geo_geb_e1();
        case GEO_GEB_E2: return geo_geb_e2();
        case GEO_GEB_PSEUDOSCALAR: return geo_geb_pseudoscalar();
        case GEO_GEB_NEGATION: return geo_geb_negation(a);
        case GEO_GEB_REVERSION: return geo_geb_reversion(a);
        case GEO_GEB_GRADE_INVOLUTION: return geo_geb_grade_involution(a);
        case GEO_GEB_CLIFFORD_CONJUGATION: return geo_geb_clifford_conjugation(a);
        case GEO_GEB_SCALAR_PROJECTION: return geo_geb_scalar_projection(a);
        case GEO_GEB_VECTOR_PROJECTION: return geo_geb_vector_projection(a);
        case GEO_GEB_BIVECTOR_PROJECTION: return geo_geb_bivector_projection(a);
        case GEO_GEB_ADDITION: return geo_geb_addition(a, b);
        case GEO_GEB_SUBTRACTION: return geo_geb_subtraction(a, b);
        case GEO_GEB_GEOMETRIC_PRODUCT: return geo_geb_geometric_product(a, b);
        case GEO_GEB_REVERSE_PRODUCT: return geo_geb_reverse_product(a, b);
        case GEO_GEB_VECTOR_WEDGE: return geo_geb_vector_wedge(a, b);
        case GEO_GEB_COMMUTATOR: return geo_geb_commutator(a, b);
        case GEO_GEB_ANTICOMMUTATOR: return geo_geb_anticommutator(a, b);
        case GEO_GEB_PROJECTION_NUMERATOR: return geo_geb_projection_numerator(a, b);
        case GEO_GEB_REJECTION_NUMERATOR: return geo_geb_rejection_numerator(a, b);
        case GEO_GEB_REFLECTION_NUMERATOR: return geo_geb_reflection_numerator(a, b);
        case GEO_GEB_DUAL: return geo_geb_dual(a);
        case GEO_GEB_EVEN_PROJECTION: return geo_geb_even_projection(a);
        case GEO_GEB_ODD_PROJECTION: return geo_geb_odd_projection(a);
        case GEO_GEB_ROTOR_ACTION: return geo_geb_rotor_action(transform, a);
        case GEO_GEB_ROTOR_COMPOSITION: return geo_geb_rotor_composition(a, b);
        case GEO_GEB_DILATION: return geo_geb_dilation(transform, a);
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE: return geo_geb_vector_inverse_projective(a);
        default: return geo_cl20_zero();
    }
}

static geo_real_t expected_scalar(uint8_t id, geo_cl20_t a, geo_cl20_t b) {
    switch ((geo_geb_target_id_t)id) {
        case GEO_GEB_VECTOR_DOT: return geo_geb_vector_dot(a, b);
        case GEO_GEB_VECTOR_NORM_SQUARED: return geo_geb_vector_norm_squared(a);
        case GEO_GEB_DISTANCE_SQUARED: return geo_geb_distance_squared(a, b);
        case GEO_GEB_ROTOR_NORM_SQUARED: return geo_geb_rotor_norm_squared(a);
        case GEO_GEB_ANGLE_COSINE_NUMERATOR: return geo_geb_angle_cosine_numerator(a, b);
        default: return (geo_real_t)0;
    }
}

int main(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)0.25, (geo_real_t)0.5, (geo_real_t)-0.375, (geo_real_t)0.125);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)-0.125, (geo_real_t)0.25, (geo_real_t)0.625, (geo_real_t)-0.25);
    const geo_cl20_t transform = geo_cl20_make((geo_real_t)0.9238795, (geo_real_t)0, (geo_real_t)0, (geo_real_t)-0.3826834);
    const geo_fixed_cl20_t fa = encode(a);
    const geo_fixed_cl20_t fb = encode(b);
    const geo_fixed_cl20_t ft = encode(transform);
    uint8_t id;

    for (id = 1u; id <= 36u; ++id) {
        geo_fixed_geb_result_t result;
        const geo_fixed_status_t status = geo_fixed_geb36_execute(id, fa, fb, ft, &result);
        if (status != GEO_FIXED_OK) {
            fail_id(id, "execution status");
            continue;
        }

        switch ((geo_geb_target_id_t)id) {
            case GEO_GEB_VECTOR_DOT:
            case GEO_GEB_VECTOR_NORM_SQUARED:
            case GEO_GEB_DISTANCE_SQUARED:
            case GEO_GEB_ROTOR_NORM_SQUARED:
            case GEO_GEB_ANGLE_COSINE_NUMERATOR:
                if (result.kind != GEO_FIXED_RESULT_SCALAR ||
                    !near_scalar(result.as.scalar, expected_scalar(id, a, b))) {
                    fail_id(id, "scalar mismatch");
                }
                break;
            case GEO_GEB_PROJECTION_NUMERATOR:
            case GEO_GEB_REJECTION_NUMERATOR:
            case GEO_GEB_REFLECTION_NUMERATOR:
            case GEO_GEB_VECTOR_INVERSE_PROJECTIVE:
                if (result.kind != GEO_FIXED_RESULT_PROJECTIVE ||
                    !near_mv(result.as.projective.represented, expected_cl20(id, a, b, transform))) {
                    fail_id(id, "projective mismatch");
                }
                if (id == GEO_GEB_VECTOR_INVERSE_PROJECTIVE &&
                    !near_scalar(result.as.projective.denominator, geo_geb_vector_norm_squared(a))) {
                    fail_id(id, "inverse denominator mismatch");
                }
                break;
            case GEO_GEB_TRANSLATION_UNIPOTENT:
                if (result.kind != GEO_FIXED_RESULT_UNIPOTENT || !near_mv(result.as.unipotent_payload, a)) {
                    fail_id(id, "unipotent payload mismatch");
                }
                break;
            default:
                if (result.kind != GEO_FIXED_RESULT_CL20 ||
                    !near_mv(result.as.cl20, expected_cl20(id, a, b, transform))) {
                    fail_id(id, "multivector mismatch");
                }
                break;
        }
    }

    if (failures != 0) {
        fprintf(stderr, "%d fixed GEB-36 assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All 36 fixed-point GEB targets passed.");
    return EXIT_SUCCESS;
}
