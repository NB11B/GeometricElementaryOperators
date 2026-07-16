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

static geo_fixed_t q(double value) {
    geo_fixed_t result = 0;
    if (geo_fixed_from_double(value, &result) != GEO_FIXED_OK) {
        ++failures;
    }
    return result;
}

int main(void) {
    geo_fixed_t result;
    geo_fixed_cl20_t a = {0, q(1.5), q(-0.5), 0};
    geo_fixed_cl20_t b = {0, q(0.25), q(2.0), 0};
    geo_fixed_cl20_t rotor = {q(cos(0.25)), 0, 0, q(-sin(0.25))};
    geo_fixed_cl20_t vector = {0, q(1.0), 0, 0};
    geo_fixed_cl20_t rotated;

    expect_true(geo_fixed_mul_rounded(q(0.1), q(0.2), &result) == GEO_FIXED_OK, "rounded multiply status");
    expect_true(fabs(geo_fixed_to_double(result) - 0.02) < 5e-5, "rounded multiply accuracy");

    expect_true(geo_fixed_vector_dot(a, b, &result) == GEO_FIXED_OK, "fixed dot status");
    expect_true(fabs(geo_fixed_to_double(result) - (-0.625)) < 8e-5, "fixed dot value");

    expect_true(geo_fixed_vector_wedge(a, b, &result) == GEO_FIXED_OK, "fixed wedge status");
    expect_true(fabs(geo_fixed_to_double(result) - 3.125) < 8e-5, "fixed wedge value");

    expect_true(geo_fixed_rotor_action(rotor, vector, &rotated) == GEO_FIXED_OK, "fixed rotor status");
    expect_true(fabs(geo_fixed_to_double(rotated.scalar)) < 2e-4, "fixed rotor scalar grade");
    expect_true(fabs(geo_fixed_to_double(rotated.e12)) < 2e-4, "fixed rotor bivector grade");

    expect_true(geo_fixed_saturating_add(INT32_MAX, 1) == INT32_MAX, "saturating add");
    expect_true(geo_fixed_saturating_sub(INT32_MIN, 1) == INT32_MIN, "saturating subtract");

    if (failures != 0) {
        fprintf(stderr, "%d extended fixed-point assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All extended fixed-point tests passed.");
    return EXIT_SUCCESS;
}
