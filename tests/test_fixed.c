#include "geo/fixed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void expect_true(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void expect_near(double actual, double expected, double tolerance, const char *message) {
    if (fabs(actual - expected) > tolerance) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%.12f expected=%.12f\n", message, actual, expected);
    }
}

int main(void) {
    geo_fixed_t a;
    geo_fixed_t b;
    geo_fixed_t product;
    geo_fixed_cl20_t e1 = {0, 0, 0, 0};
    geo_fixed_cl20_t e2 = {0, 0, 0, 0};
    geo_fixed_cl20_t result;
    const double tolerance = 2.0 / (double)(UINT64_C(1) << GEO_FIXED_FRACTION_BITS);

    expect_true(geo_fixed_from_double(1.5, &a) == GEO_FIXED_OK, "convert 1.5");
    expect_true(geo_fixed_from_double(-0.25, &b) == GEO_FIXED_OK, "convert -0.25");
    expect_true(geo_fixed_mul(a, b, &product) == GEO_FIXED_OK, "fixed multiply");
    expect_near(geo_fixed_to_double(product), -0.375, tolerance, "1.5 * -0.25");

    expect_true(geo_fixed_from_double(1.0, &e1.e1) == GEO_FIXED_OK, "encode e1");
    expect_true(geo_fixed_from_double(1.0, &e2.e2) == GEO_FIXED_OK, "encode e2");
    expect_true(geo_fixed_cl20_mul(e1, e2, &result) == GEO_FIXED_OK, "fixed Cl20 e1e2");
    expect_near(geo_fixed_to_double(result.scalar), 0.0, tolerance, "fixed e1e2 scalar");
    expect_near(geo_fixed_to_double(result.e1), 0.0, tolerance, "fixed e1e2 e1");
    expect_near(geo_fixed_to_double(result.e2), 0.0, tolerance, "fixed e1e2 e2");
    expect_near(geo_fixed_to_double(result.e12), 1.0, tolerance, "fixed e1e2 e12");

    expect_true(geo_fixed_cl20_mul(e2, e1, &result) == GEO_FIXED_OK, "fixed Cl20 e2e1");
    expect_near(geo_fixed_to_double(result.e12), -1.0, tolerance, "fixed e2e1 e12");

    if (failures != 0) {
        fprintf(stderr, "%d fixed-point test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All fixed-point kernel tests passed.");
    return EXIT_SUCCESS;
}
