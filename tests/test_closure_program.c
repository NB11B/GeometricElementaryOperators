#include "geo/geb36.h"
#include "geo/structured_program.h"

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

static void expect_near_real(geo_real_t actual, geo_real_t expected, const char *message) {
    const double error = fabs((double)(actual - expected));
    if (error > (double)GEO_TEST_TOLERANCE) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%.17g expected=%.17g error=%.17g\n",
            message, (double)actual, (double)expected, error);
    }
}

static void expect_near_cl20(geo_cl20_t actual, geo_cl20_t expected, const char *message) {
    if (!geo_cl20_near(actual, expected, GEO_TEST_TOLERANCE)) {
        ++failures;
        fprintf(stderr,
            "FAIL: %s\n  actual=(%.17g, %.17g, %.17g, %.17g)\n"
            "  expected=(%.17g, %.17g, %.17g, %.17g)\n",
            message,
            (double)actual.scalar, (double)actual.e1,
            (double)actual.e2, (double)actual.e12,
            (double)expected.scalar, (double)expected.e1,
            (double)expected.e2, (double)expected.e12);
    }
}

static geo_struct_value_t execute_unary(
    geo_struct_opcode_t opcode,
    geo_struct_value_t input
) {
    const geo_struct_instruction_t instructions[] = {
        {(uint8_t)opcode, 1u, 0u, 0u}
    };
    const geo_struct_program_t program = {instructions, 1u, 2u, 1u};
    geo_struct_value_t registers[2];
    geo_status_t status;

    registers[0] = input;
    registers[1] = geo_struct_value_from_cl20(geo_cl20_zero());
    status = geo_struct_program_execute(&program, registers, 2u);
    expect_true(status == GEO_STATUS_OK, "unary structured program executes");
    return registers[1];
}

static void test_constant_terminal_programs(void) {
    const geo_cl20_t constants[] = {
        geo_geb_zero(),
        geo_geb_one(),
        geo_geb_minus_one(),
        geo_geb_e1(),
        geo_geb_e2()
    };
    size_t index;

    for (index = 0u; index < sizeof(constants) / sizeof(constants[0]); ++index) {
        const geo_struct_program_t program = {NULL, 0u, 1u, 0u};
        geo_struct_value_t registers[1];
        geo_cl20_t output;
        geo_status_t status;

        registers[0] = geo_struct_value_from_cl20(constants[index]);
        status = geo_struct_program_execute(&program, registers, 1u);
        expect_true(status == GEO_STATUS_OK, "terminal-only program executes");
        status = geo_struct_read_cl20(&registers[0], &output);
        expect_true(status == GEO_STATUS_OK, "terminal-only output is Cl20");
        expect_near_cl20(output, constants[index], "terminal constant preserved");
    }
}

static void test_involutions_and_projections(void) {
    const geo_cl20_t value = geo_cl20_make(
        (geo_real_t)2,
        (geo_real_t)3,
        (geo_real_t)-5,
        (geo_real_t)7
    );
    geo_struct_value_t result;
    geo_cl20_t output;
    geo_status_t status;

    result = execute_unary(GEO_STRUCT_OP_CL20_NEGATE, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "negation output readable");
    expect_near_cl20(output, geo_geb_negation(value), "compiled negation");

    result = execute_unary(GEO_STRUCT_OP_CL20_REVERSE, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "reversion output readable");
    expect_near_cl20(output, geo_geb_reversion(value), "compiled reversion");

    result = execute_unary(GEO_STRUCT_OP_CL20_GRADE_INVOLUTION, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "grade involution output readable");
    expect_near_cl20(output, geo_geb_grade_involution(value), "compiled grade involution");

    result = execute_unary(GEO_STRUCT_OP_CL20_CLIFFORD_CONJUGATE, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "conjugation output readable");
    expect_near_cl20(output, geo_geb_clifford_conjugation(value), "compiled Clifford conjugation");

    result = execute_unary(GEO_STRUCT_OP_PROJECT_SCALAR, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "scalar projection readable");
    expect_near_cl20(output, geo_geb_scalar_projection(value), "compiled scalar projection");

    result = execute_unary(GEO_STRUCT_OP_PROJECT_VECTOR, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "vector projection readable");
    expect_near_cl20(output, geo_geb_vector_projection(value), "compiled vector projection");

    result = execute_unary(GEO_STRUCT_OP_PROJECT_BIVECTOR, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "bivector projection readable");
    expect_near_cl20(output, geo_geb_bivector_projection(value), "compiled bivector projection");

    result = execute_unary(GEO_STRUCT_OP_PROJECT_EVEN, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "even projection readable");
    expect_near_cl20(output, geo_geb_even_projection(value), "compiled even projection");

    result = execute_unary(GEO_STRUCT_OP_PROJECT_ODD, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &output);
    expect_true(status == GEO_STATUS_OK, "odd projection readable");
    expect_near_cl20(output, geo_geb_odd_projection(value), "compiled odd projection");
}

static void test_dual_and_rotor_norm(void) {
    const geo_cl20_t value = geo_cl20_make(
        (geo_real_t)1.5,
        (geo_real_t)-2,
        (geo_real_t)4,
        (geo_real_t)0.75
    );
    const geo_real_t angle = (geo_real_t)0.37;
    const geo_cl20_t rotor = geo_cl20_make(
        (geo_real_t)cos((double)(angle * (geo_real_t)0.5)),
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)-sin((double)(angle * (geo_real_t)0.5))
    );
    geo_struct_value_t result;
    geo_cl20_t cl20_output;
    geo_real_t scalar_output;
    geo_status_t status;

    result = execute_unary(GEO_STRUCT_OP_DUAL, geo_struct_value_from_cl20(value));
    status = geo_struct_read_cl20(&result, &cl20_output);
    expect_true(status == GEO_STATUS_OK, "dual output readable");
    expect_near_cl20(cl20_output, geo_geb_dual(value), "compiled dual");

    result = execute_unary(GEO_STRUCT_OP_ROTOR_NORM_SQUARED, geo_struct_value_from_cl20(rotor));
    status = geo_struct_read_scalar(&result, &scalar_output);
    expect_true(status == GEO_STATUS_OK, "rotor norm output readable");
    expect_near_real(scalar_output, geo_geb_rotor_norm_squared(rotor), "compiled rotor norm squared");
}

static void test_translation_unipotent(void) {
    const geo_cl20_t translation = geo_cl20_make(
        (geo_real_t)0,
        (geo_real_t)1.25,
        (geo_real_t)-0.5,
        (geo_real_t)0
    );
    const geo_struct_instruction_t instructions[] = {
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 1u, 0u, 0u}
    };
    const geo_struct_program_t program = {instructions, 1u, 2u, 1u};
    geo_struct_value_t registers[2];
    geo_unipotent_t output;
    geo_unipotent_t expected;
    geo_status_t status;

    registers[0] = geo_struct_value_from_cl20(translation);
    registers[1] = geo_struct_value_from_cl20(geo_cl20_zero());
    status = geo_struct_program_execute(&program, registers, 2u);
    expect_true(status == GEO_STATUS_OK, "translation program executes");
    status = geo_struct_read_unipotent(&registers[1], &output);
    expect_true(status == GEO_STATUS_OK, "translation unipotent output readable");
    expected = geo_geb_translation_unipotent(translation);
    expect_near_cl20(
        geo_unipotent_extract(output),
        geo_unipotent_extract(expected),
        "compiled translation unipotent"
    );
}

int main(void) {
    test_constant_terminal_programs();
    test_involutions_and_projections();
    test_dual_and_rotor_norm();
    test_translation_unipotent();

    if (failures != 0) {
        fprintf(stderr, "%d closure assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All remaining GEB closure-program tests passed.");
    return EXIT_SUCCESS;
}
