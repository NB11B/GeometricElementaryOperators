#include "geo/omega.h"

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
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_opposite_lane(void) {
    const geo_cl20_t a = geo_cl20_make(
        (geo_real_t)1,
        (geo_real_t)2,
        (geo_real_t)-1,
        (geo_real_t)3
    );
    const geo_cl20_t b = geo_cl20_make(
        (geo_real_t)-2,
        (geo_real_t)1,
        (geo_real_t)4,
        (geo_real_t)-1
    );
    const geo_opposite_t product = geo_opposite_mul(
        geo_opposite_from_cl20(a),
        geo_opposite_from_cl20(b)
    );

    expect_near_cl20(product.forward, geo_cl20_mul(a, b), "opposite forward product");
    expect_near_cl20(
        product.reverse,
        geo_cl20_reverse(geo_cl20_mul(a, b)),
        "opposite reverse product"
    );
    expect_true(
        geo_opposite_is_consistent(product, GEO_TEST_TOLERANCE),
        "opposite pair remains consistent"
    );
}

static void test_scalar_lane(void) {
    const geo_state_t left = geo_state_from_scalar((geo_real_t)0);
    const geo_state_t right = geo_state_from_scalar((geo_real_t)1);
    geo_state_t output;
    const geo_status_t status = geo_omega_apply(
        &left,
        &right,
        GEO_LANE_SCALAR,
        &output
    );

    expect_true(status == GEO_STATUS_OK, "scalar Omega status");
    expect_near_real(output.scalar, (geo_real_t)1, "exp(0) - log(1) = 1");
    expect_true(
        output.active_lanes == GEO_LANE_SCALAR,
        "scalar Omega activates only scalar lane"
    );
}

static void test_scalar_domain_guard(void) {
    const geo_state_t left = geo_state_from_scalar((geo_real_t)0);
    const geo_state_t right = geo_state_from_scalar((geo_real_t)0);
    geo_state_t output;

    expect_true(
        geo_omega_apply(&left, &right, GEO_LANE_SCALAR, &output) ==
            GEO_STATUS_LOG_DOMAIN,
        "Omega rejects nonpositive logarithm input"
    );
}

static void test_geometric_lane(void) {
    const geo_cl20_t e1 = geo_cl20_basis_e1();
    const geo_cl20_t e2 = geo_cl20_basis_e2();
    const geo_state_t left = geo_state_from_cl20(e1);
    const geo_state_t right = geo_state_from_cl20(e2);
    geo_state_t output;
    const geo_status_t status = geo_omega_apply(
        &left,
        &right,
        GEO_LANE_GEOMETRIC,
        &output
    );

    expect_true(status == GEO_STATUS_OK, "geometric Omega status");
    expect_near_cl20(
        output.geometric.forward,
        geo_cl20_basis_e12(),
        "Omega geometric lane computes e1 e2"
    );
    expect_true(
        output.active_lanes == GEO_LANE_GEOMETRIC,
        "geometric Omega activates only geometric lane"
    );
    expect_true(
        output.scale.numerator == 1 && output.scale.denominator == 1,
        "geometric scale remains exact"
    );
}

static void test_flat_program(void) {
    geo_state_t registers[4];
    const geo_instruction_t instructions[] = {
        {
            GEO_OPCODE_OMEGA,
            2u,
            0u,
            1u,
            GEO_LANE_GEOMETRIC
        },
        {
            GEO_OPCODE_COPY,
            3u,
            2u,
            0u,
            GEO_LANE_NONE
        }
    };
    const geo_program_t program = {
        instructions,
        sizeof(instructions) / sizeof(instructions[0]),
        sizeof(registers) / sizeof(registers[0])
    };

    registers[0] = geo_state_from_cl20(geo_cl20_basis_e1());
    registers[1] = geo_state_from_cl20(geo_cl20_basis_e2());
    registers[2] = geo_state_zero();
    registers[3] = geo_state_zero();

    expect_true(
        geo_program_execute(&program, registers, 4u) == GEO_STATUS_OK,
        "flat program execution status"
    );
    expect_near_cl20(
        registers[3].geometric.forward,
        geo_cl20_basis_e12(),
        "flat program computes and copies e12"
    );
}

int main(void) {
    test_opposite_lane();
    test_scalar_lane();
    test_scalar_domain_guard();
    test_geometric_lane();
    test_flat_program();

    if (failures != 0) {
        fprintf(stderr, "%d test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All Omega kernel tests passed.");
    return EXIT_SUCCESS;
}
