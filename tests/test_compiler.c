#include "geo/compiler.h"

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

static void test_scalar_tree(void) {
    const geo_witness_node_t nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 0},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 1},
        {GEO_WITNESS_OMEGA, GEO_LANE_SCALAR, 0, 1, 0}
    };
    const geo_witness_tree_t tree = {
        nodes,
        sizeof(nodes) / sizeof(nodes[0]),
        2,
        2
    };
    geo_instruction_t instructions[1];
    uint8_t node_registers[3];
    geo_compile_workspace_t workspace = {
        instructions,
        sizeof(instructions) / sizeof(instructions[0]),
        node_registers,
        sizeof(node_registers) / sizeof(node_registers[0])
    };
    geo_compiled_witness_t compiled;
    geo_state_t registers[3];

    expect_true(
        geo_witness_compile(&tree, &workspace, &compiled) == GEO_STATUS_OK,
        "compile scalar witness"
    );
    expect_true(compiled.program.instruction_count == 1, "one scalar instruction");
    expect_true(compiled.program.register_count == 3, "three scalar registers");
    expect_true(compiled.root_register == 2, "scalar root register");

    registers[0] = geo_state_from_scalar((geo_real_t)0);
    registers[1] = geo_state_from_scalar((geo_real_t)1);
    registers[2] = geo_state_zero();

    expect_true(
        geo_program_execute(&compiled.program, registers, 3) == GEO_STATUS_OK,
        "execute scalar witness"
    );
    expect_true(
        registers[compiled.root_register].scalar > (geo_real_t)1 - GEO_TEST_TOLERANCE &&
        registers[compiled.root_register].scalar < (geo_real_t)1 + GEO_TEST_TOLERANCE,
        "scalar witness result"
    );
}

static void test_geometric_tree(void) {
    const geo_witness_node_t nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 0},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 1},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 2},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0, 1, 0},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 3, 2, 0}
    };
    const geo_witness_tree_t tree = {
        nodes,
        sizeof(nodes) / sizeof(nodes[0]),
        3,
        4
    };
    geo_instruction_t instructions[2];
    uint8_t node_registers[5];
    geo_compile_workspace_t workspace = {
        instructions,
        sizeof(instructions) / sizeof(instructions[0]),
        node_registers,
        sizeof(node_registers) / sizeof(node_registers[0])
    };
    geo_compiled_witness_t compiled;
    geo_state_t registers[5];
    const geo_cl20_t expected = geo_cl20_mul(
        geo_cl20_mul(geo_cl20_basis_e1(), geo_cl20_basis_e2()),
        geo_cl20_basis_e1()
    );

    expect_true(
        geo_witness_compile(&tree, &workspace, &compiled) == GEO_STATUS_OK,
        "compile geometric witness"
    );

    registers[0] = geo_state_from_cl20(geo_cl20_basis_e1());
    registers[1] = geo_state_from_cl20(geo_cl20_basis_e2());
    registers[2] = geo_state_from_cl20(geo_cl20_basis_e1());
    registers[3] = geo_state_zero();
    registers[4] = geo_state_zero();

    expect_true(
        geo_program_execute(&compiled.program, registers, 5) == GEO_STATUS_OK,
        "execute geometric witness"
    );
    expect_true(
        geo_cl20_near(
            registers[compiled.root_register].geometric.forward,
            expected,
            GEO_TEST_TOLERANCE
        ),
        "compiled geometric result equals direct result"
    );
    expect_true(
        geo_opposite_consistent(
            registers[compiled.root_register].geometric,
            GEO_TEST_TOLERANCE
        ),
        "compiled opposite lane remains consistent"
    );
}

static void test_invalid_trees(void) {
    const geo_witness_node_t forward_reference[] = {
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 1, 1, 0},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 0}
    };
    const geo_witness_tree_t bad_reference = {
        forward_reference,
        2,
        1,
        0
    };
    const geo_witness_node_t bad_terminal_nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 3}
    };
    const geo_witness_tree_t bad_terminal = {
        bad_terminal_nodes,
        1,
        1,
        0
    };
    const geo_witness_node_t valid_nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 0},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0, 0, 0},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0, 1, 0}
    };
    const geo_witness_tree_t valid_tree = {valid_nodes, 3, 1, 2};
    geo_instruction_t instructions[1];
    uint8_t too_small_register_map[2];
    geo_compile_workspace_t small_workspace = {
        instructions,
        1,
        too_small_register_map,
        2
    };
    geo_compiled_witness_t compiled;

    expect_true(
        geo_witness_validate(&bad_reference) == GEO_STATUS_BAD_TREE,
        "reject forward node reference"
    );
    expect_true(
        geo_witness_validate(&bad_terminal) == GEO_STATUS_BAD_TREE,
        "reject terminal outside terminal bank"
    );
    expect_true(
        geo_witness_compile(&valid_tree, &small_workspace, &compiled) ==
            GEO_STATUS_BUFFER_CAPACITY,
        "reject insufficient compiler workspace"
    );
}

int main(void) {
    test_scalar_tree();
    test_geometric_tree();
    test_invalid_trees();

    if (failures != 0) {
        fprintf(stderr, "%d test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All witness compiler tests passed.");
    return EXIT_SUCCESS;
}
