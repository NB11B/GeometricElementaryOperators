#include "geo/folding.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void expect_near(geo_real_t actual, geo_real_t expected, const char *message) {
    const double error = fabs((double)(actual - expected));
    if (error > (double)GEO_TEST_TOLERANCE) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%.17g expected=%.17g\n",
            message, (double)actual, (double)expected);
    }
}

static geo_status_t optimize_tree(
    const geo_witness_tree_t *tree,
    geo_instruction_t *instructions,
    uint8_t *node_registers,
    uint8_t *live_lanes,
    uint16_t *representatives,
    geo_optimized_witness_t *output
) {
    geo_optimizer_workspace_t workspace;

    workspace.instructions = instructions;
    workspace.instruction_capacity = 8u;
    workspace.node_registers = node_registers;
    workspace.node_register_capacity = 8u;
    workspace.live_lanes = live_lanes;
    workspace.live_lane_capacity = 8u;
    workspace.representatives = representatives;
    workspace.representative_capacity = 8u;

    return geo_witness_compile_optimized(tree, &workspace, output);
}

static void test_scalar_constant_folding(void) {
    const geo_witness_node_t nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
        {GEO_WITNESS_OMEGA, GEO_LANE_SCALAR, 0u, 1u, 0u}
    };
    const geo_witness_tree_t tree = {nodes, 3u, 2u, 2u};
    geo_instruction_t optimized_instructions[8];
    uint8_t node_registers[8];
    uint8_t live_lanes[8];
    uint16_t representatives[8];
    geo_optimized_witness_t optimized;
    const geo_state_t terminals[] = {
        geo_state_from_scalar((geo_real_t)0),
        geo_state_from_scalar((geo_real_t)1)
    };
    const uint8_t terminal_constants[] = {1u, 1u};
    geo_instruction_t folded_instructions[8];
    geo_state_t initial_registers[8];
    uint8_t old_to_new[8];
    uint8_t constant_flags[8];
    uint8_t register_kinds[8];
    geo_folding_workspace_t workspace;
    geo_folded_program_t folded;
    geo_state_t runtime_registers[8];
    geo_status_t status;

    status = optimize_tree(
        &tree,
        optimized_instructions,
        node_registers,
        live_lanes,
        representatives,
        &optimized
    );
    expect_true(status == GEO_STATUS_OK, "optimize scalar tree");

    workspace.instructions = folded_instructions;
    workspace.instruction_capacity = 8u;
    workspace.initial_registers = initial_registers;
    workspace.initial_register_capacity = 8u;
    workspace.old_to_new = old_to_new;
    workspace.old_to_new_capacity = 8u;
    workspace.constant_flags = constant_flags;
    workspace.constant_flag_capacity = 8u;
    workspace.register_kinds = register_kinds;
    workspace.register_kind_capacity = 8u;

    status = geo_program_fold_constants(
        &optimized,
        terminals,
        terminal_constants,
        2u,
        &workspace,
        &folded
    );
    expect_true(status == GEO_STATUS_OK, "fold scalar constants");
    expect_true(folded.original_instruction_count == 1u, "one original scalar instruction");
    expect_true(folded.folded_instruction_count == 0u, "scalar instruction folded away");
    expect_true(folded.folded_constant_nodes == 1u, "one constant node folded");
    expect_true(folded.scalar_register_count == 3u, "three scalar registers classified");
    expect_true(folded.estimated_typed_bytes < folded.estimated_unified_bytes,
        "typed scalar plan uses less memory");

    memcpy(runtime_registers, initial_registers,
        folded.program.register_count * sizeof(runtime_registers[0]));
    status = geo_program_execute(
        &folded.program,
        runtime_registers,
        folded.program.register_count
    );
    expect_true(status == GEO_STATUS_OK, "execute fully folded scalar program");
    expect_near(runtime_registers[folded.root_register].scalar,
        (geo_real_t)1, "folded scalar result");
}

static void test_dynamic_geometric_plan(void) {
    const geo_witness_node_t nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0u, 1u, 0u}
    };
    const geo_witness_tree_t tree = {nodes, 3u, 2u, 2u};
    geo_instruction_t optimized_instructions[8];
    uint8_t node_registers[8];
    uint8_t live_lanes[8];
    uint16_t representatives[8];
    geo_optimized_witness_t optimized;
    const geo_state_t terminals[] = {
        geo_state_from_cl20(geo_cl20_basis_e1()),
        geo_state_from_cl20(geo_cl20_basis_e2())
    };
    const uint8_t terminal_constants[] = {0u, 0u};
    geo_instruction_t folded_instructions[8];
    geo_state_t initial_registers[8];
    uint8_t old_to_new[8];
    uint8_t constant_flags[8];
    uint8_t register_kinds[8];
    geo_folding_workspace_t workspace;
    geo_folded_program_t folded;
    geo_state_t runtime_registers[8];
    geo_status_t status;

    status = optimize_tree(
        &tree,
        optimized_instructions,
        node_registers,
        live_lanes,
        representatives,
        &optimized
    );
    expect_true(status == GEO_STATUS_OK, "optimize geometric tree");

    workspace.instructions = folded_instructions;
    workspace.instruction_capacity = 8u;
    workspace.initial_registers = initial_registers;
    workspace.initial_register_capacity = 8u;
    workspace.old_to_new = old_to_new;
    workspace.old_to_new_capacity = 8u;
    workspace.constant_flags = constant_flags;
    workspace.constant_flag_capacity = 8u;
    workspace.register_kinds = register_kinds;
    workspace.register_kind_capacity = 8u;

    status = geo_program_fold_constants(
        &optimized,
        terminals,
        terminal_constants,
        2u,
        &workspace,
        &folded
    );
    expect_true(status == GEO_STATUS_OK, "plan dynamic geometric program");
    expect_true(folded.folded_instruction_count == 1u,
        "dynamic geometric instruction remains");
    expect_true(folded.folded_constant_nodes == 0u,
        "no dynamic node folded");
    expect_true(folded.geometric_register_count == 3u,
        "three geometric registers classified");
    expect_true(folded.estimated_typed_bytes < folded.estimated_unified_bytes,
        "typed geometric plan uses less memory");

    memcpy(runtime_registers, initial_registers,
        folded.program.register_count * sizeof(runtime_registers[0]));
    status = geo_program_execute(
        &folded.program,
        runtime_registers,
        folded.program.register_count
    );
    expect_true(status == GEO_STATUS_OK, "execute dynamic geometric program");
    expect_true(geo_cl20_near(
        runtime_registers[folded.root_register].geometric.forward,
        geo_cl20_basis_e12(),
        GEO_TEST_TOLERANCE
    ), "dynamic geometric result e1*e2=e12");
}

int main(void) {
    test_scalar_constant_folding();
    test_dynamic_geometric_plan();

    if (failures != 0) {
        fprintf(stderr, "%d folding assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All constant-folding and typed register tests passed.");
    return EXIT_SUCCESS;
}
