#include "geo/fixed_control.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static geo_fixed_t fixed(double value) {
    geo_fixed_t output = 0;
    if (geo_fixed_from_double(value, &output) != GEO_FIXED_OK) {
        fprintf(stderr, "fixture conversion failed: %.17g\n", value);
        exit(EXIT_FAILURE);
    }
    return output;
}

static int same_matrix(geo_fixed_m2_t left, geo_fixed_m2_t right) {
    return left.m00 == right.m00 && left.m01 == right.m01 &&
        left.m10 == right.m10 && left.m11 == right.m11;
}

static void test_identity_control(void) {
    const geo_fixed_m2_t identity = {
        fixed(1.0), fixed(0.0), fixed(0.0), fixed(1.0)
    };
    const geo_fixed_m2_t right = {
        fixed(0.75), fixed(-0.25), fixed(0.5), fixed(1.25)
    };
    const geo_fixed_m2_t expected = {
        fixed(-0.25), fixed(-0.25), fixed(0.5), fixed(0.25)
    };
    geo_fixed_m2_t output;

    expect(
        geo_fixed_control_apply(identity, right, &output) == GEO_FIXED_OK,
        "fixed control applies identity routing state"
    );
    expect(same_matrix(output, expected), "Gc(I,Y)=Y-I");
}

static void test_zero_control(void) {
    const geo_fixed_m2_t zero = {0, 0, 0, 0};
    const geo_fixed_m2_t right = {
        fixed(0.25), fixed(0.5), fixed(-0.25), fixed(1.0)
    };
    geo_fixed_m2_t output = {
        fixed(0.25), fixed(0.25), fixed(0.25), fixed(0.25)
    };

    expect(
        geo_fixed_control_apply(zero, right, &output) == GEO_FIXED_OK,
        "zero routing state executes"
    );
    expect(same_matrix(output, zero), "Gc(0,Y)=0");
}

static void test_overflow_is_transactional(void) {
    const geo_fixed_m2_t left = {INT32_MAX, 0, 0, 0};
    const geo_fixed_m2_t right = {fixed(2.0), 0, 0, 0};
    geo_fixed_m2_t sentinel;
    geo_fixed_m2_t output;

    memset(&sentinel, 0x5a, sizeof(sentinel));
    output = sentinel;
    expect(
        geo_fixed_control_apply(left, right, &output) == GEO_FIXED_OVERFLOW,
        "fixed routing control reports multiplication overflow"
    );
    expect(memcmp(&output, &sentinel, sizeof(output)) == 0,
        "failed routing control leaves output unchanged");
    expect(
        geo_fixed_control_apply(left, right, NULL) == GEO_FIXED_OVERFLOW,
        "fixed routing control rejects null output"
    );
}

int main(void) {
    test_identity_control();
    test_zero_control();
    test_overflow_is_transactional();

    if (failures != 0) {
        fprintf(stderr, "%d fixed-control assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All fixed-control tests passed.");
    return EXIT_SUCCESS;
}
