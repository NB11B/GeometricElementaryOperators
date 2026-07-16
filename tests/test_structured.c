#include "geo/structured.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static void expect_near_real(
    geo_real_t actual,
    geo_real_t expected,
    const char *message
) {
    const double error = fabs((double)(actual - expected));
    if (error > (double)GEO_TEST_TOLERANCE) {
        ++failures;
        fprintf(
            stderr,
            "FAIL: %s: actual=%.17g expected=%.17g error=%.17g\n",
            message,
            (double)actual,
            (double)expected,
            error
        );
    }
}

static void expect_near_cl20(
    geo_cl20_t actual,
    geo_cl20_t expected,
    const char *message
) {
    if (!geo_cl20_near(actual, expected, GEO_TEST_TOLERANCE)) {
        ++failures;
        fprintf(
            stderr,
            "FAIL: %s\n"
            "  actual   = (%.17g, %.17g, %.17g, %.17g)\n"
            "  expected = (%.17g, %.17g, %.17g, %.17g)\n",
            message,
            (double)actual.scalar,
            (double)actual.e1,
            (double)actual.e2,
            (double)actual.e12,
            (double)expected.scalar,
            (double)expected.e1,
            (double)expected.e2,
            (double)expected.e12
        );
    }
}

static void test_unipotent_addition(void) {
    const geo_cl20_t a = geo_cl20_make(
        (geo_real_t)1,
        (geo_real_t)2,
        (geo_real_t)-3,
        (geo_real_t)4
    );
    const geo_cl20_t b = geo_cl20_make(
        (geo_real_t)-5,
        (geo_real_t)6,
        (geo_real_t)7,
        (geo_real_t)-8
    );
    const geo_unipotent_t ua = geo_unipotent_from_cl20(a);
    const geo_unipotent_t ub = geo_unipotent_from_cl20(b);
    const geo_unipotent_t product = geo_unipotent_mul(ua, ub);

    expect_near_cl20(
        geo_unipotent_extract(product),
        geo_cl20_add(a, b),
        "U(A)U(B) carries A+B"
    );

    expect_near_cl20(
        geo_unipotent_extract(geo_unipotent_mul(geo_unipotent_identity(), ua)),
        a,
        "unipotent identity"
    );
}

static void test_ordered_products_and_hadamard(void) {
    const geo_cl20_t a = geo_cl20_make(
        (geo_real_t)0,
        (geo_real_t)2,
        (geo_real_t)-3,
        (geo_real_t)0
    );
    const geo_cl20_t b = geo_cl20_make(
        (geo_real_t)0,
        (geo_real_t)5,
        (geo_real_t)7,
        (geo_real_t)0
    );
    const geo_ordered_pair_t ordered = geo_ordered_products(a, b);
    const geo_hadamard_pair_t projective = geo_hadamard_mix_projective(ordered);
    const geo_hadamard_pair_t exact = geo_hadamard_mix_exact(ordered);
    geo_scaled_cl20_t normalized_symmetric;
    geo_scaled_cl20_t normalized_antisymmetric;

    expect_near_cl20(ordered.ab, geo_cl20_mul(a, b), "ordered ab");
    expect_near_cl20(ordered.ba, geo_cl20_mul(b, a), "ordered ba");

    expect_true(
        projective.symmetric.scale.numerator == 2 &&
        projective.symmetric.scale.denominator == 1,
        "projective symmetric scale is two"
    );
    expect_true(
        projective.antisymmetric.scale.numerator == 2 &&
        projective.antisymmetric.scale.denominator == 1,
        "projective antisymmetric scale is two"
    );

    expect_near_real(
        projective.symmetric.represented.scalar,
        (geo_real_t)-22,
        "projective symmetric is twice dot"
    );
    expect_near_real(
        projective.antisymmetric.represented.e12,
        (geo_real_t)58,
        "projective antisymmetric is twice wedge"
    );

    expect_near_real(exact.symmetric.represented.scalar, (geo_real_t)-11, "exact dot");
    expect_near_real(exact.antisymmetric.represented.e12, (geo_real_t)29, "exact wedge");

    expect_true(
        geo_scaled_cl20_normalize(&projective.symmetric, &normalized_symmetric) == GEO_STATUS_OK,
        "normalize symmetric projective value"
    );
    expect_true(
        geo_scaled_cl20_normalize(&projective.antisymmetric, &normalized_antisymmetric) == GEO_STATUS_OK,
        "normalize antisymmetric projective value"
    );

    expect_near_cl20(
        normalized_symmetric.represented,
        exact.symmetric.represented,
        "deferred symmetric normalization matches exact mixer"
    );
    expect_near_cl20(
        normalized_antisymmetric.represented,
        exact.antisymmetric.represented,
        "deferred antisymmetric normalization matches exact mixer"
    );
}

static void test_metric_helpers(void) {
    const geo_cl20_t v = geo_cl20_make(
        (geo_real_t)0,
        (geo_real_t)3,
        (geo_real_t)4,
        (geo_real_t)0
    );
    const geo_cl20_t onto_x = geo_cl20_make(
        (geo_real_t)0,
        (geo_real_t)2,
        (geo_real_t)0,
        (geo_real_t)0
    );
    const geo_cl20_t zero = geo_cl20_zero();
    geo_cl20_t result;

    expect_true(
        geo_vector_normalize(v, &result) == GEO_STATUS_OK,
        "vector normalization succeeds"
    );
    expect_near_cl20(
        result,
        geo_cl20_make((geo_real_t)0, (geo_real_t)0.6, (geo_real_t)0.8, (geo_real_t)0),
        "normalized 3-4 vector"
    );

    expect_true(
        geo_vector_inverse(onto_x, &result) == GEO_STATUS_OK,
        "vector inverse succeeds"
    );
    expect_near_cl20(
        result,
        geo_cl20_make((geo_real_t)0, (geo_real_t)0.5, (geo_real_t)0, (geo_real_t)0),
        "vector inverse a/(a dot a)"
    );

    expect_true(
        geo_vector_projection(v, onto_x, &result) == GEO_STATUS_OK,
        "projection succeeds"
    );
    expect_near_cl20(
        result,
        geo_cl20_make((geo_real_t)0, (geo_real_t)3, (geo_real_t)0, (geo_real_t)0),
        "projection onto x axis"
    );

    expect_true(
        geo_vector_rejection(v, onto_x, &result) == GEO_STATUS_OK,
        "rejection succeeds"
    );
    expect_near_cl20(
        result,
        geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)4, (geo_real_t)0),
        "rejection from x axis"
    );

    expect_true(
        geo_vector_reflection(v, onto_x, &result) == GEO_STATUS_OK,
        "reflection succeeds"
    );
    expect_near_cl20(
        result,
        geo_cl20_make((geo_real_t)0, (geo_real_t)-3, (geo_real_t)4, (geo_real_t)0),
        "reflection across plane with x normal"
    );

    expect_true(
        geo_vector_normalize(zero, &result) == GEO_STATUS_ZERO_NORM,
        "zero vector normalization is rejected"
    );
    expect_true(
        geo_vector_projection(v, zero, &result) == GEO_STATUS_ZERO_NORM,
        "projection onto zero vector is rejected"
    );
}

int main(void) {
    test_unipotent_addition();
    test_ordered_products_and_hadamard();
    test_metric_helpers();

    if (failures != 0) {
        fprintf(stderr, "%d structured test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All structured geometry kernel tests passed.");
    return EXIT_SUCCESS;
}
