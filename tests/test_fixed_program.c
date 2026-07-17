#include "geo/fixed_program.h"

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

static geo_fixed_cl20_t mv(double scalar, double e1, double e2, double e12) {
    geo_fixed_cl20_t value;
    value.scalar = fixed(scalar);
    value.e1 = fixed(e1);
    value.e2 = fixed(e2);
    value.e12 = fixed(e12);
    return value;
}

static int same_mv(geo_fixed_cl20_t a, geo_fixed_cl20_t b) {
    return a.scalar == b.scalar && a.e1 == b.e1 &&
        a.e2 == b.e2 && a.e12 == b.e12;
}

static void test_all_targets(void) {
    uint8_t target;
    const geo_fixed_cl20_t a = mv(0.125, 0.25, -0.125, 0.0625);
    const geo_fixed_cl20_t b = mv(-0.0625, 0.125, 0.25, -0.03125);
    const geo_fixed_cl20_t transform = mv(0.96875, 0.0, 0.0, -0.25);

    for (target = (uint8_t)GEO_GEB_ZERO;
         target <= (uint8_t)GEO_GEB_ANGLE_COSINE_NUMERATOR;
         ++target) {
        geo_fixed_program_instruction_t instruction = {target, 3u, 0u, 1u, 2u};
        geo_fixed_program_t program = {&instruction, 1u, 4u, 3u};
        geo_fixed_geb_result_t registers[4];
        geo_fixed_program_status_t status;

        registers[0] = geo_fixed_program_value_from_cl20(a);
        registers[1] = geo_fixed_program_value_from_cl20(b);
        registers[2] = geo_fixed_program_value_from_cl20(transform);
        memset(&registers[3], 0x5a, sizeof(registers[3]));

        status = geo_fixed_program_execute(&program, registers, 4u);
        if (status != GEO_FIXED_PROGRAM_OK) {
            ++failures;
            fprintf(stderr, "FAIL: target %u returned program status %u\n",
                (unsigned int)target, (unsigned int)status);
        }
    }
}

static void test_chained_execution(void) {
    const geo_fixed_program_instruction_t instructions[] = {
        {(uint8_t)GEO_GEB_ADDITION, 3u, 0u, 1u, 2u},
        {(uint8_t)GEO_GEB_VECTOR_DOT, 4u, 3u, 1u, 2u},
        {(uint8_t)GEO_GEB_NEGATION, 5u, 4u, 0u, 0u},
        {(uint8_t)GEO_GEB_VECTOR_INVERSE_PROJECTIVE, 6u, 0u, 0u, 0u},
        {(uint8_t)GEO_GEB_NEGATION, 7u, 6u, 0u, 0u},
        {(uint8_t)GEO_GEB_ROTOR_ACTION, 8u, 3u, 0u, 2u}
    };
    const geo_fixed_program_t program = {instructions, 6u, 9u, 8u};
    const geo_fixed_cl20_t a = mv(0.0, 0.5, 0.25, 0.0);
    const geo_fixed_cl20_t b = mv(0.0, -0.25, 0.5, 0.0);
    const geo_fixed_cl20_t rotor = mv(0.96875, 0.0, 0.0, -0.25);
    geo_fixed_geb_result_t registers[9];
    geo_fixed_geb_result_t direct;
    geo_fixed_cl20_t normalized;
    geo_fixed_cl20_t expected;
    geo_fixed_program_status_t status;

    memset(registers, 0, sizeof(registers));
    registers[0] = geo_fixed_program_value_from_cl20(a);
    registers[1] = geo_fixed_program_value_from_cl20(b);
    registers[2] = geo_fixed_program_value_from_cl20(rotor);

    status = geo_fixed_program_execute(&program, registers, 9u);
    expect(status == GEO_FIXED_PROGRAM_OK, "chained program succeeds");

    expect(
        geo_fixed_geb36_execute(GEO_GEB_ADDITION, a, b, rotor, &direct) == GEO_FIXED_OK,
        "direct addition fixture succeeds"
    );
    expect(registers[3].kind == (uint8_t)GEO_FIXED_RESULT_CL20,
        "addition result remains multivector typed");
    expect(same_mv(registers[3].as.cl20, direct.as.cl20),
        "program addition matches direct fixed backend");

    expect(registers[4].kind == (uint8_t)GEO_FIXED_RESULT_SCALAR,
        "dot result is scalar typed");
    expect(registers[5].kind == (uint8_t)GEO_FIXED_RESULT_CL20,
        "scalar input promotes to scalar multivector when chained");
    expect(registers[5].as.cl20.scalar == -registers[4].as.scalar,
        "promoted scalar participates in following GEB operation");

    expect(registers[6].kind == (uint8_t)GEO_FIXED_RESULT_PROJECTIVE,
        "inverse remains projective until consumed");
    expect(
        geo_fixed_program_read_cl20(&registers[6], &normalized) == GEO_FIXED_PROGRAM_OK,
        "projective result normalizes on typed read"
    );
    expected = normalized;
    expected.scalar = -expected.scalar;
    expected.e1 = -expected.e1;
    expected.e2 = -expected.e2;
    expected.e12 = -expected.e12;
    expect(same_mv(registers[7].as.cl20, expected),
        "projective result normalizes before chained negation");

    expect(
        geo_fixed_geb36_execute(GEO_GEB_ROTOR_ACTION, registers[3].as.cl20,
            registers[3].as.cl20, rotor, &direct) == GEO_FIXED_OK,
        "direct rotor fixture succeeds"
    );
    expect(same_mv(registers[8].as.cl20, direct.as.cl20),
        "program transform operand matches direct rotor action");
}

static void test_error_propagation(void) {
    const geo_fixed_program_instruction_t add = {
        (uint8_t)GEO_GEB_ADDITION, 2u, 0u, 1u, 0u
    };
    const geo_fixed_program_t add_program = {&add, 1u, 3u, 2u};
    geo_fixed_geb_result_t registers[3];
    geo_fixed_geb_result_t sentinel;
    geo_fixed_program_instruction_t inverse = {
        (uint8_t)GEO_GEB_VECTOR_INVERSE_PROJECTIVE, 1u, 0u, 0u, 0u
    };
    geo_fixed_program_t inverse_program = {&inverse, 1u, 2u, 1u};
    geo_fixed_program_instruction_t invalid = {0u, 1u, 0u, 0u, 0u};
    geo_fixed_program_t invalid_program = {&invalid, 1u, 2u, 1u};
    geo_fixed_program_t malformed = {NULL, 1u, 1u, 0u};

    registers[0] = geo_fixed_program_value_from_cl20(
        (geo_fixed_cl20_t){INT32_MAX, 0, 0, 0}
    );
    registers[1] = geo_fixed_program_value_from_cl20(
        (geo_fixed_cl20_t){1, 0, 0, 0}
    );
    sentinel = geo_fixed_program_value_from_scalar(12345);
    registers[2] = sentinel;
    expect(
        geo_fixed_program_execute(&add_program, registers, 3u) ==
            GEO_FIXED_PROGRAM_OVERFLOW,
        "arithmetic overflow propagates through program status"
    );
    expect(registers[2].kind == sentinel.kind &&
        registers[2].as.scalar == sentinel.as.scalar,
        "failed instruction leaves destination unchanged"
    );

    registers[0] = geo_fixed_program_value_from_cl20(
        (geo_fixed_cl20_t){0, 0, 0, 0}
    );
    registers[1] = sentinel;
    expect(
        geo_fixed_program_execute(&inverse_program, registers, 2u) ==
            GEO_FIXED_PROGRAM_INVALID_SCALE,
        "zero projective denominator is rejected"
    );

    expect(
        geo_fixed_program_execute(&invalid_program, registers, 2u) ==
            GEO_FIXED_PROGRAM_BAD_TARGET,
        "invalid target is reported"
    );
    expect(
        geo_fixed_program_execute(&malformed, registers, 2u) ==
            GEO_FIXED_PROGRAM_NULL_ARGUMENT,
        "null instruction stream is rejected"
    );
}

int main(void) {
    test_all_targets();
    test_chained_execution();
    test_error_propagation();

    if (failures != 0) {
        fprintf(stderr, "%d fixed-program assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All fixed-program tests passed.");
    return EXIT_SUCCESS;
}
