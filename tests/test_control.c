#include "geo/control.h"

#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-5f)
#endif

static int failures = 0;

static void expect_near_mat2(
    geo_mat2_t actual,
    geo_mat2_t expected,
    const char *message
) {
    if (!geo_mat2_near(actual, expected, GEO_TEST_TOLERANCE)) {
        ++failures;
        fprintf(
            stderr,
            "FAIL: %s\n"
            "  actual   = [[%.17g, %.17g], [%.17g, %.17g]]\n"
            "  expected = [[%.17g, %.17g], [%.17g, %.17g]]\n",
            message,
            (double)actual.m00,
            (double)actual.m01,
            (double)actual.m10,
            (double)actual.m11,
            (double)expected.m00,
            (double)expected.m01,
            (double)expected.m10,
            (double)expected.m11
        );
    }
}

static void test_matrix_units(void) {
    const geo_mat2_t e11 = geo_mat2_e11();
    const geo_mat2_t e12 = geo_mat2_e12();
    const geo_mat2_t e21 = geo_mat2_e21();
    const geo_mat2_t e22 = geo_mat2_e22();
    const geo_mat2_t zero = geo_mat2_zero();

    expect_near_mat2(geo_mat2_mul(e11, e11), e11, "E11 E11 = E11");
    expect_near_mat2(geo_mat2_mul(e22, e22), e22, "E22 E22 = E22");
    expect_near_mat2(geo_mat2_mul(e12, e21), e11, "E12 E21 = E11");
    expect_near_mat2(geo_mat2_mul(e21, e12), e22, "E21 E12 = E22");
    expect_near_mat2(geo_mat2_mul(e11, e22), zero, "E11 E22 = 0");
    expect_near_mat2(geo_mat2_mul(e22, e11), zero, "E22 E11 = 0");
}

static void test_gc_witnesses(void) {
    const geo_mat2_t identity = geo_mat2_identity();
    const geo_mat2_t p = geo_mat2_e11();
    const geo_mat2_t exchange = geo_mat2_exchange();

    const geo_mat2_t zero = geo_control_gc(identity, identity);
    const geo_mat2_t minus_identity = geo_control_gc(identity, zero);
    const geo_mat2_t i_minus_p = geo_control_gc(identity, p);
    const geo_mat2_t e22 = geo_control_gc(i_minus_p, p);
    const geo_mat2_t s_minus_p = geo_control_gc(exchange, p);
    const geo_mat2_t e12 = geo_control_gc(s_minus_p, p);
    const geo_mat2_t s_squared_minus_s = geo_control_gc(exchange, exchange);
    const geo_mat2_t e21 = geo_control_gc(i_minus_p, s_squared_minus_s);

    expect_near_mat2(zero, geo_mat2_zero(), "G(I,I) = 0");
    expect_near_mat2(
        minus_identity,
        geo_mat2_neg_identity(),
        "G(I,G(I,I)) = -I"
    );
    expect_near_mat2(
        e22,
        geo_mat2_e22(),
        "G(G(I,P),P) = E22"
    );
    expect_near_mat2(
        e12,
        geo_mat2_e12(),
        "G(G(S,P),P) = E12"
    );
    expect_near_mat2(
        e21,
        geo_mat2_e21(),
        "G(G(I,P),G(S,S)) = E21"
    );
}

int main(void) {
    test_matrix_units();
    test_gc_witnesses();

    if (failures != 0) {
        fprintf(stderr, "%d control test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All control algebra kernel tests passed.");
    return EXIT_SUCCESS;
}
