#include "geo/cl20.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-5f)
#endif

static int failures = 0;

static void expect_true(const int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void expect_near_real(
    const geo_real_t actual,
    const geo_real_t expected,
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
    const geo_cl20_t actual,
    const geo_cl20_t expected,
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

static void test_basis_products(void) {
    const geo_cl20_t one = geo_cl20_one();
    const geo_cl20_t e1 = geo_cl20_basis_e1();
    const geo_cl20_t e2 = geo_cl20_basis_e2();
    const geo_cl20_t e12 = geo_cl20_basis_e12();

    expect_near_cl20(geo_cl20_mul(e1, e1), one, "e1^2 = 1");
    expect_near_cl20(geo_cl20_mul(e2, e2), one, "e2^2 = 1");
    expect_near_cl20(geo_cl20_mul(e12, e12), geo_cl20_neg(one), "e12^2 = -1");
    expect_near_cl20(geo_cl20_mul(e1, e2), e12, "e1 e2 = e12");
    expect_near_cl20(geo_cl20_mul(e2, e1), geo_cl20_neg(e12), "e2 e1 = -e12");
    expect_near_cl20(geo_cl20_mul(one, e1), e1, "1 e1 = e1");
    expect_near_cl20(geo_cl20_mul(e2, one), e2, "e2 1 = e2");
}

static void test_involutions(void) {
    const geo_cl20_t value = geo_cl20_make(
        (geo_real_t)2,
        (geo_real_t)3,
        (geo_real_t)-5,
        (geo_real_t)7
    );

    expect_near_cl20(
        geo_cl20_reverse(value),
        geo_cl20_make((geo_real_t)2, (geo_real_t)3, (geo_real_t)-5, (geo_real_t)-7),
        "reversion"
    );

    expect_near_cl20(
        geo_cl20_grade_involution(value),
        geo_cl20_make((geo_real_t)2, (geo_real_t)-3, (geo_real_t)5, (geo_real_t)7),
        "grade involution"
    );

    expect_near_cl20(
        geo_cl20_clifford_conjugate(value),
        geo_cl20_make((geo_real_t)2, (geo_real_t)-3, (geo_real_t)5, (geo_real_t)-7),
        "Clifford conjugation"
    );

    expect_near_cl20(
        geo_cl20_reverse(geo_cl20_reverse(value)),
        value,
        "reversion is involutive"
    );
}

static void test_reverse_antihomomorphism(void) {
    const geo_cl20_t a = geo_cl20_make(
        (geo_real_t)1.25,
        (geo_real_t)-2.0,
        (geo_real_t)0.5,
        (geo_real_t)3.0
    );
    const geo_cl20_t b = geo_cl20_make(
        (geo_real_t)-0.75,
        (geo_real_t)4.0,
        (geo_real_t)-1.5,
        (geo_real_t)0.25
    );

    const geo_cl20_t lhs = geo_cl20_reverse(geo_cl20_mul(a, b));
    const geo_cl20_t rhs = geo_cl20_mul(geo_cl20_reverse(b), geo_cl20_reverse(a));

    expect_near_cl20(lhs, rhs, "reverse(AB) = reverse(B) reverse(A)");
}

static void test_vector_decomposition(void) {
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

    const geo_cl20_t ab = geo_cl20_mul(a, b);
    const geo_real_t dot = geo_cl20_vector_dot(a, b);
    const geo_real_t wedge = geo_cl20_vector_wedge(a, b);

    expect_near_real(dot, (geo_real_t)-11, "vector dot");
    expect_near_real(wedge, (geo_real_t)29, "vector wedge");
    expect_near_real(ab.scalar, dot, "geometric product scalar equals dot");
    expect_near_real(ab.e12, wedge, "geometric product bivector equals wedge");
    expect_near_real(ab.e1, (geo_real_t)0, "vector product has no e1 component");
    expect_near_real(ab.e2, (geo_real_t)0, "vector product has no e2 component");
    expect_near_real(geo_cl20_vector_norm_squared(a), (geo_real_t)13, "vector norm squared");
}

static void test_grade_projection(void) {
    const geo_cl20_t value = geo_cl20_make(
        (geo_real_t)2,
        (geo_real_t)3,
        (geo_real_t)-5,
        (geo_real_t)7
    );

    expect_near_cl20(
        geo_cl20_project(value, GEO_GRADE_SCALAR),
        geo_cl20_make((geo_real_t)2, (geo_real_t)0, (geo_real_t)0, (geo_real_t)0),
        "scalar projection"
    );

    expect_near_cl20(
        geo_cl20_project(value, GEO_GRADE_VECTOR),
        geo_cl20_make((geo_real_t)0, (geo_real_t)3, (geo_real_t)-5, (geo_real_t)0),
        "vector projection"
    );

    expect_near_cl20(
        geo_cl20_project(value, GEO_GRADE_BIVECTOR),
        geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, (geo_real_t)7),
        "bivector projection"
    );
}

int main(void) {
    test_basis_products();
    test_involutions();
    test_reverse_antihomomorphism();
    test_vector_decomposition();
    test_grade_projection();

    if (failures != 0) {
        fprintf(stderr, "%d test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All Cl(2,0) kernel tests passed.");
    return EXIT_SUCCESS;
}
