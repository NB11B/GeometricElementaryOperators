#include "geo/structured_program.h"
#include "geo/geb36.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOLERANCE ((geo_real_t)1e-5f)
#endif

static int failures = 0;

static void expect_cl20(
    geo_cl20_t actual,
    geo_cl20_t expected,
    const char *message
) {
    if (!geo_cl20_near(actual, expected, GEO_TEST_TOLERANCE)) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_addition_and_subtraction(void) {
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
    const geo_struct_instruction_t add_instructions[] = {
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 2u, 0u, 0u},
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 3u, 1u, 1u},
        {GEO_STRUCT_OP_UNIPOTENT_COMPOSE, 4u, 2u, 3u},
        {GEO_STRUCT_OP_UNIPOTENT_EXTRACT, 5u, 4u, 4u}
    };
    const geo_struct_instruction_t sub_instructions[] = {
        {GEO_STRUCT_OP_CL20_NEGATE, 2u, 1u, 1u},
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 3u, 0u, 0u},
        {GEO_STRUCT_OP_UNIPOTENT_ENCODE, 4u, 2u, 2u},
        {GEO_STRUCT_OP_UNIPOTENT_COMPOSE, 5u, 3u, 4u},
        {GEO_STRUCT_OP_UNIPOTENT_EXTRACT, 6u, 5u, 5u}
    };
    const geo_struct_program_t add_program = {
        add_instructions,
        sizeof(add_instructions) / sizeof(add_instructions[0]),
        6u,
        5u
    };
    const geo_struct_program_t sub_program = {
        sub_instructions,
        sizeof(sub_instructions) / sizeof(sub_instructions[0]),
        7u,
        6u
    };
    geo_struct_value_t add_registers[6];
    geo_struct_value_t sub_registers[7];
    geo_cl20_t result;

    add_registers[0] = geo_struct_value_from_cl20(a);
    add_registers[1] = geo_struct_value_from_cl20(b);
    if (geo_struct_program_execute(&add_program, add_registers, 6u) != GEO_STATUS_OK ||
        geo_struct_read_cl20(&add_registers[add_program.root_register], &result) != GEO_STATUS_OK) {
        ++failures;
        fprintf(stderr, "FAIL: addition structured execution\n");
    } else {
        expect_cl20(result, geo_geb_addition(a, b), "unipotent addition matches GEB reference");
    }

    sub_registers[0] = geo_struct_value_from_cl20(a);
    sub_registers[1] = geo_struct_value_from_cl20(b);
    if (geo_struct_program_execute(&sub_program, sub_registers, 7u) != GEO_STATUS_OK ||
        geo_struct_read_cl20(&sub_registers[sub_program.root_register], &result) != GEO_STATUS_OK) {
        ++failures;
        fprintf(stderr, "FAIL: subtraction structured execution\n");
    } else {
        expect_cl20(result, geo_geb_subtraction(a, b), "unipotent subtraction matches GEB reference");
    }
}

static void test_dot_wedge_commutators(void) {
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
    const geo_struct_instruction_t symmetric_instructions[] = {
        {GEO_STRUCT_OP_ORDERED_PRODUCTS, 2u, 0u, 1u},
        {GEO_STRUCT_OP_HADAMARD_EXACT, 3u, 2u, 2u},
        {GEO_STRUCT_OP_SELECT_SYMMETRIC, 4u, 3u, 3u}
    };
    const geo_struct_instruction_t antisymmetric_instructions[] = {
        {GEO_STRUCT_OP_ORDERED_PRODUCTS, 2u, 0u, 1u},
        {GEO_STRUCT_OP_HADAMARD_EXACT, 3u, 2u, 2u},
        {GEO_STRUCT_OP_SELECT_ANTISYMMETRIC, 4u, 3u, 3u}
    };
    const geo_struct_program_t symmetric_program = {
        symmetric_instructions,
        3u,
        5u,
        4u
    };
    const geo_struct_program_t antisymmetric_program = {
        antisymmetric_instructions,
        3u,
        5u,
        4u
    };
    geo_struct_value_t registers[5];
    geo_cl20_t result;
    geo_cl20_t expected;

    registers[0] = geo_struct_value_from_cl20(a);
    registers[1] = geo_struct_value_from_cl20(b);
    if (geo_struct_program_execute(&symmetric_program, registers, 5u) != GEO_STATUS_OK ||
        geo_struct_read_cl20(&registers[4], &result) != GEO_STATUS_OK) {
        ++failures;
        fprintf(stderr, "FAIL: symmetric structured execution\n");
    } else {
        expected = geo_geb_anticommutator(a, b);
        expect_cl20(result, expected, "Hadamard symmetric matches anticommutator");
        expect_cl20(
            result,
            geo_cl20_make(geo_geb_vector_dot(a, b), (geo_real_t)0, (geo_real_t)0, (geo_real_t)0),
            "Hadamard symmetric matches vector dot"
        );
    }

    registers[0] = geo_struct_value_from_cl20(a);
    registers[1] = geo_struct_value_from_cl20(b);
    if (geo_struct_program_execute(&antisymmetric_program, registers, 5u) != GEO_STATUS_OK ||
        geo_struct_read_cl20(&registers[4], &result) != GEO_STATUS_OK) {
        ++failures;
        fprintf(stderr, "FAIL: antisymmetric structured execution\n");
    } else {
        expected = geo_geb_commutator(a, b);
        expect_cl20(result, expected, "Hadamard antisymmetric matches commutator");
        expect_cl20(result, geo_geb_vector_wedge(a, b), "Hadamard antisymmetric matches vector wedge");
    }
}

static void test_projective_hadamard_scale(void) {
    const geo_cl20_t a = geo_cl20_basis_e1();
    const geo_cl20_t b = geo_cl20_basis_e2();
    const geo_ordered_pair_t ordered = geo_ordered_products(a, b);
    const geo_hadamard_pair_t mixed = geo_hadamard_mix_projective(ordered);

    if (mixed.antisymmetric.scale.numerator != 2 ||
        mixed.antisymmetric.scale.denominator != 1) {
        ++failures;
        fprintf(stderr, "FAIL: projective Hadamard carries scale 2\n");
    }
    expect_cl20(
        mixed.antisymmetric.represented,
        geo_cl20_scale(geo_geb_vector_wedge(a, b), (geo_real_t)2),
        "projective wedge represents twice canonical wedge"
    );
}

int main(void) {
    test_addition_and_subtraction();
    test_dot_wedge_commutators();
    test_projective_hadamard_scale();

    if (failures != 0) {
        fprintf(stderr, "%d test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All structured representation program tests passed.");
    return EXIT_SUCCESS;
}
