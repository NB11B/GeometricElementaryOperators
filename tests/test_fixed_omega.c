#include "geo/fixed_omega.h"

#include <limits.h>
#include <math.h>
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

#if GEO_FIXED_FRACTION_BITS >= 8
static int near_double(geo_fixed_t actual, double expected, double tolerance) {
    return fabs(geo_fixed_to_double(actual) - expected) <= tolerance;
}
#endif

static void test_fixed_eml(void) {
    const geo_fixed_t zero = fixed(0.0);
    const geo_fixed_t one = fixed(1.0);
    geo_fixed_t output = fixed(0.25);

    expect(
        geo_fixed_eml_exp(zero, &output) == GEO_FIXED_OMEGA_OK && output == one,
        "fixed exp(0) is exactly one"
    );
    output = fixed(0.25);
    expect(
        geo_fixed_eml_log(one, &output) == GEO_FIXED_OMEGA_OK && output == zero,
        "fixed log(1) is exactly zero"
    );
    output = zero;
    expect(
        geo_fixed_eml_apply(zero, one, &output) == GEO_FIXED_OMEGA_OK && output == one,
        "fixed EML apply computes exp(0)-log(1)"
    );

#if GEO_FIXED_FRACTION_BITS >= 8
    output = zero;
    expect(
        geo_fixed_eml_exp(fixed(0.25), &output) == GEO_FIXED_OMEGA_OK &&
            near_double(output, exp(0.25), 32.0 / (double)(INT64_C(1) << GEO_FIXED_FRACTION_BITS)),
        "fixed exp approximation is bounded"
    );
    output = zero;
    expect(
        geo_fixed_eml_log(fixed(1.5), &output) == GEO_FIXED_OMEGA_OK &&
            near_double(output, log(1.5), 48.0 / (double)(INT64_C(1) << GEO_FIXED_FRACTION_BITS)),
        "fixed log approximation is bounded"
    );
#endif

    output = fixed(0.25);
    expect(
        geo_fixed_eml_log(zero, &output) == GEO_FIXED_OMEGA_LOG_DOMAIN,
        "fixed log rejects zero"
    );
    expect(output == fixed(0.25), "failed fixed log leaves output unchanged");
    expect(
        geo_fixed_eml_exp(zero, NULL) == GEO_FIXED_OMEGA_NULL_ARGUMENT,
        "fixed exp rejects a null output"
    );
}

static void test_opposite_and_scale(void) {
    const geo_fixed_cl20_t e1 = {0, 0, 0, 0};
    geo_fixed_cl20_t left_value = e1;
    geo_fixed_cl20_t right_value = e1;
    geo_fixed_opposite_t left;
    geo_fixed_opposite_t right;
    geo_fixed_opposite_t product;
    geo_fixed_opposite_t sentinel;
    geo_fixed_state_t left_state;
    geo_fixed_state_t right_state;
    geo_fixed_state_t result;

    left_value.e1 = fixed(1.0);
    right_value.e2 = fixed(1.0);
    expect(
        geo_fixed_opposite_from_cl20(left_value, &left) == GEO_FIXED_OMEGA_OK,
        "construct fixed left opposite lane"
    );
    expect(
        geo_fixed_opposite_from_cl20(right_value, &right) == GEO_FIXED_OMEGA_OK,
        "construct fixed right opposite lane"
    );
    expect(
        geo_fixed_opposite_mul(left, right, &product) == GEO_FIXED_OMEGA_OK,
        "multiply fixed opposite lanes"
    );
    expect(product.forward.e12 == fixed(1.0), "forward lane computes e1*e2");
    expect(product.reverse.e12 == fixed(-1.0), "reverse lane propagates opposite order");

    memset(&sentinel, 0x5a, sizeof(sentinel));
    left_value = (geo_fixed_cl20_t){0, 0, 0, INT32_MIN};
    product = sentinel;
    expect(
        geo_fixed_opposite_from_cl20(left_value, &product) == GEO_FIXED_OMEGA_OVERFLOW,
        "opposite construction reports unrepresentable reverse"
    );
    expect(memcmp(&product, &sentinel, sizeof(product)) == 0,
        "failed opposite construction leaves output unchanged");

    left_value = (geo_fixed_cl20_t){0, fixed(0.5), fixed(0.25), 0};
    right_value = (geo_fixed_cl20_t){0, fixed(-0.25), fixed(0.5), 0};
    expect(
        geo_fixed_state_from_cl20(left_value, &left_state) == GEO_FIXED_OMEGA_OK,
        "construct fixed geometric state"
    );
    expect(
        geo_fixed_state_from_cl20(right_value, &right_state) == GEO_FIXED_OMEGA_OK,
        "construct second fixed geometric state"
    );
    left_state.scale = (geo_fixed_scale_t){6, 8};
    right_state.scale = (geo_fixed_scale_t){-10, 15};
    expect(
        geo_fixed_omega_apply(
            &left_state,
            &right_state,
            GEO_FIXED_LANE_GEOMETRIC,
            &result
        ) == GEO_FIXED_OMEGA_OK,
        "fixed geometric Omega application succeeds"
    );
    expect(result.scale.numerator == -1 && result.scale.denominator == 2,
        "fixed projective scales reduce exactly");
}

static void test_unified_omega(void) {
    const geo_fixed_cl20_t e1 = {0, fixed(1.0), 0, 0};
    const geo_fixed_cl20_t e2 = {0, 0, fixed(1.0), 0};
    geo_fixed_state_t left;
    geo_fixed_state_t right;
    geo_fixed_state_t output;

    expect(
        geo_fixed_state_from_cl20(e1, &left) == GEO_FIXED_OMEGA_OK &&
        geo_fixed_state_from_cl20(e2, &right) == GEO_FIXED_OMEGA_OK,
        "construct unified geometric fixtures"
    );
    left.scalar = fixed(0.0);
    right.scalar = fixed(1.0);
    left.active_lanes = GEO_FIXED_LANE_ALL;
    right.active_lanes = GEO_FIXED_LANE_ALL;

    expect(
        geo_fixed_omega_apply(&left, &right, GEO_FIXED_LANE_ALL, &output) ==
            GEO_FIXED_OMEGA_OK,
        "fixed unified Omega application succeeds"
    );
    expect(output.scalar == fixed(1.0), "fixed unified scalar lane matches EML");
    expect(output.geometric.forward.e12 == fixed(1.0),
        "fixed unified forward lane matches product");
    expect(output.geometric.reverse.e12 == fixed(-1.0),
        "fixed unified reverse lane matches reversion");

    right.scalar = fixed(0.0);
    expect(
        geo_fixed_omega_apply(&left, &right, GEO_FIXED_LANE_ALL, &output) ==
            GEO_FIXED_OMEGA_LOG_DOMAIN,
        "fixed unified Omega propagates scalar-domain errors"
    );
}

static void test_fixed_omega_program(void) {
    const geo_fixed_omega_instruction_t instructions[] = {
        {
            GEO_FIXED_OMEGA_OPCODE_APPLY,
            2u,
            0u,
            1u,
            GEO_FIXED_LANE_ALL
        },
        {
            GEO_FIXED_OMEGA_OPCODE_COPY,
            3u,
            2u,
            0u,
            GEO_FIXED_LANE_GEOMETRIC
        }
    };
    const geo_fixed_omega_program_t program = {instructions, 2u, 4u};
    const geo_fixed_cl20_t e1 = {0, fixed(1.0), 0, 0};
    const geo_fixed_cl20_t e2 = {0, 0, fixed(1.0), 0};
    geo_fixed_state_t registers[4];
    geo_fixed_state_t sentinel = geo_fixed_state_from_scalar(fixed(0.25));
    geo_fixed_omega_program_t malformed = {NULL, 1u, 1u};
    geo_fixed_omega_instruction_t bad_instruction = {
        99u, 2u, 0u, 1u, GEO_FIXED_LANE_ALL
    };
    geo_fixed_omega_program_t bad_program = {&bad_instruction, 1u, 3u};

    memset(registers, 0, sizeof(registers));
    expect(
        geo_fixed_state_from_cl20(e1, &registers[0]) == GEO_FIXED_OMEGA_OK &&
        geo_fixed_state_from_cl20(e2, &registers[1]) == GEO_FIXED_OMEGA_OK,
        "construct program fixtures"
    );
    registers[0].scalar = fixed(0.0);
    registers[1].scalar = fixed(1.0);
    registers[0].active_lanes = GEO_FIXED_LANE_ALL;
    registers[1].active_lanes = GEO_FIXED_LANE_ALL;

    expect(
        geo_fixed_omega_program_execute(&program, registers, 4u) ==
            GEO_FIXED_OMEGA_OK,
        "fixed Omega program executes"
    );
    expect(registers[2].scalar == fixed(1.0), "program computes scalar lane");
    expect(registers[2].geometric.forward.e12 == fixed(1.0),
        "program computes geometric lane");
    expect(registers[3].active_lanes == GEO_FIXED_LANE_GEOMETRIC,
        "copy instruction prunes unrequested scalar lane");
    expect(registers[3].geometric.forward.e12 == fixed(1.0),
        "copy instruction preserves requested geometric lane");

    expect(
        geo_fixed_omega_program_execute(&malformed, registers, 4u) ==
            GEO_FIXED_OMEGA_NULL_ARGUMENT,
        "fixed Omega program rejects null instruction stream"
    );
    registers[2] = sentinel;
    expect(
        geo_fixed_omega_program_execute(&bad_program, registers, 4u) ==
            GEO_FIXED_OMEGA_BAD_OPCODE,
        "fixed Omega program rejects invalid opcode"
    );
    expect(registers[2].scalar == sentinel.scalar &&
        registers[2].active_lanes == sentinel.active_lanes,
        "failed fixed Omega instruction leaves destination unchanged"
    );
}

int main(void) {
    test_fixed_eml();
    test_opposite_and_scale();
    test_unified_omega();
    test_fixed_omega_program();

    if (failures != 0) {
        fprintf(stderr, "%d fixed-Omega assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All fixed-Omega tests passed.");
    return EXIT_SUCCESS;
}
