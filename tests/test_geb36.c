#include "geo/geb36.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-5f)
#endif

static int failures = 0;

static void expect_true(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void expect_near_real(geo_real_t actual, geo_real_t expected, const char *message) {
    const double error = fabs((double)(actual - expected));
    if (error > (double)GEO_TEST_TOLERANCE) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%.17g expected=%.17g\n", message, (double)actual, (double)expected);
    }
}

static void expect_near_cl20(geo_cl20_t actual, geo_cl20_t expected, const char *message) {
    if (!geo_cl20_near(actual, expected, GEO_TEST_TOLERANCE)) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static geo_cl20_t vector(geo_real_t x, geo_real_t y) {
    return geo_cl20_make((geo_real_t)0, x, y, (geo_real_t)0);
}

static void test_manifest(void) {
    size_t count = 0u;
    size_t i;
    size_t exact = 0u;
    size_t projective = 0u;
    size_t supplied = 0u;
    const geo_geb_target_info_t *manifest = geo_geb36_manifest(&count);

    expect_true(manifest != NULL, "manifest exists");
    expect_true(count == 36u, "manifest has 36 targets");

    for (i = 0u; i < count; ++i) {
        expect_true(manifest[i].id == (uint8_t)(i + 1u), "manifest ids are contiguous");
        expect_true(manifest[i].name != NULL && strlen(manifest[i].name) > 0u, "manifest names are present");
        if (manifest[i].closure == GEO_GEB_EXACT) {
            ++exact;
        } else if (manifest[i].closure == GEO_GEB_PROJECTIVE_SCALED) {
            ++projective;
        } else if (manifest[i].closure == GEO_GEB_EXACT_WITH_SUPPLIED_TRANSFORM) {
            ++supplied;
        }
    }

    expect_true(exact == 29u, "29 exact targets");
    expect_true(projective == 5u, "5 projective targets");
    expect_true(supplied == 2u, "2 supplied-transform targets");
    expect_true(geo_geb36_target_info(0u) == NULL, "target zero rejected");
    expect_true(geo_geb36_target_info(37u) == NULL, "target 37 rejected");
}

static void test_constants_and_unary(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)2, (geo_real_t)3, (geo_real_t)-4, (geo_real_t)5);

    expect_near_cl20(geo_geb_zero(), geo_cl20_zero(), "zero");
    expect_near_cl20(geo_geb_one(), geo_cl20_one(), "one");
    expect_near_cl20(geo_geb_minus_one(), geo_cl20_neg(geo_cl20_one()), "minus one");
    expect_near_cl20(geo_geb_e1(), geo_cl20_basis_e1(), "e1");
    expect_near_cl20(geo_geb_e2(), geo_cl20_basis_e2(), "e2");
    expect_near_cl20(geo_geb_pseudoscalar(), geo_cl20_basis_e12(), "pseudoscalar");
    expect_near_cl20(geo_geb_negation(a), geo_cl20_neg(a), "negation");
    expect_near_cl20(geo_geb_reversion(a), geo_cl20_reverse(a), "reversion");
    expect_near_cl20(geo_geb_grade_involution(a), geo_cl20_grade_involution(a), "grade involution");
    expect_near_cl20(geo_geb_clifford_conjugation(a), geo_cl20_clifford_conjugate(a), "Clifford conjugation");
    expect_near_cl20(geo_geb_scalar_projection(a), geo_cl20_make((geo_real_t)2, 0, 0, 0), "scalar projection");
    expect_near_cl20(geo_geb_vector_projection(a), geo_cl20_make(0, (geo_real_t)3, (geo_real_t)-4, 0), "vector projection");
    expect_near_cl20(geo_geb_bivector_projection(a), geo_cl20_make(0, 0, 0, (geo_real_t)5), "bivector projection");
    expect_near_cl20(geo_geb_even_projection(a), geo_cl20_make((geo_real_t)2, 0, 0, (geo_real_t)5), "even projection");
    expect_near_cl20(geo_geb_odd_projection(a), geo_cl20_make(0, (geo_real_t)3, (geo_real_t)-4, 0), "odd projection");
}

static void test_binary_algebra(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)1, (geo_real_t)2, (geo_real_t)-1, (geo_real_t)3);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)-2, (geo_real_t)4, (geo_real_t)5, (geo_real_t)-1);
    const geo_cl20_t ab = geo_cl20_mul(a, b);
    const geo_cl20_t ba = geo_cl20_mul(b, a);

    expect_near_cl20(geo_geb_addition(a, b), geo_cl20_add(a, b), "addition");
    expect_near_cl20(geo_geb_subtraction(a, b), geo_cl20_sub(a, b), "subtraction");
    expect_near_cl20(geo_geb_geometric_product(a, b), ab, "geometric product");
    expect_near_cl20(geo_geb_reverse_product(a, b), geo_cl20_mul(geo_cl20_reverse(b), geo_cl20_reverse(a)), "reverse product");
    expect_near_cl20(geo_geb_commutator(a, b), geo_cl20_scale(geo_cl20_sub(ab, ba), (geo_real_t)0.5), "commutator");
    expect_near_cl20(geo_geb_anticommutator(a, b), geo_cl20_scale(geo_cl20_add(ab, ba), (geo_real_t)0.5), "anticommutator");
}

static void test_vector_metric_and_projective(void) {
    const geo_cl20_t a = vector((geo_real_t)2, (geo_real_t)-3);
    const geo_cl20_t b = vector((geo_real_t)5, (geo_real_t)7);
    const geo_cl20_t projection_numerator = geo_geb_projection_numerator(a, b);
    const geo_real_t b_norm = geo_cl20_vector_norm_squared(b);

    expect_near_real(geo_geb_vector_dot(a, b), (geo_real_t)-11, "vector dot");
    expect_near_cl20(geo_geb_vector_wedge(a, b), geo_cl20_make(0, 0, 0, (geo_real_t)29), "vector wedge");
    expect_near_real(geo_geb_vector_norm_squared(a), (geo_real_t)13, "vector norm squared");
    expect_near_real(geo_geb_distance_squared(a, b), (geo_real_t)109, "distance squared");
    expect_near_cl20(projection_numerator, geo_cl20_scale(b, (geo_real_t)-11), "projection numerator");
    expect_near_cl20(
        geo_geb_rejection_numerator(a, b),
        geo_cl20_sub(geo_cl20_scale(a, b_norm), projection_numerator),
        "rejection numerator"
    );
    expect_near_cl20(
        geo_geb_reflection_numerator(a, b),
        geo_cl20_sub(geo_cl20_scale(a, b_norm), geo_cl20_scale(projection_numerator, (geo_real_t)2)),
        "reflection numerator"
    );
    expect_near_cl20(geo_geb_vector_inverse_projective(a), a, "vector inverse projective numerator");
    expect_near_real(geo_geb_angle_cosine_numerator(a, b), (geo_real_t)-11, "angle cosine numerator");
}

static void test_dual_and_transforms(void) {
    const geo_cl20_t e1 = geo_cl20_basis_e1();
    const geo_cl20_t e2 = geo_cl20_basis_e2();
    const geo_cl20_t rotor = geo_cl20_make((geo_real_t)0.7071067811865475244, 0, 0, (geo_real_t)-0.7071067811865475244);
    const geo_cl20_t composed = geo_geb_rotor_composition(rotor, rotor);
    const geo_cl20_t action = geo_geb_rotor_action(rotor, e1);
    const geo_cl20_t scalar_transform = geo_cl20_make((geo_real_t)2, 0, 0, 0);
    const geo_cl20_t dilated = geo_geb_dilation(scalar_transform, e1);
    const geo_unipotent_t translation = geo_geb_translation_unipotent(e2);

    expect_near_cl20(geo_geb_dual(e1), geo_cl20_neg(e2), "right dual e1 = -e2");
    expect_near_cl20(action, e2, "90 degree rotor action");
    expect_near_real(geo_geb_rotor_norm_squared(rotor), (geo_real_t)1, "rotor norm squared");
    expect_near_cl20(composed, geo_cl20_make(0, 0, 0, (geo_real_t)-1), "rotor composition");
    expect_near_cl20(dilated, geo_cl20_scale(e1, (geo_real_t)4), "supplied sandwich dilation");
    expect_near_cl20(geo_unipotent_extract(translation), e2, "translation unipotent");
}

int main(void) {
    test_manifest();
    test_constants_and_unary();
    test_binary_algebra();
    test_vector_metric_and_projective();
    test_dual_and_transforms();

    if (failures != 0) {
        fprintf(stderr, "%d GEB-36 test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All GEB-36 reference tests passed.");
    return EXIT_SUCCESS;
}
