#include "geo/banked.h"

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

static void expect_real(geo_real_t actual, geo_real_t expected, const char *message) {
    if (fabs((double)(actual - expected)) > (double)GEO_TEST_TOLERANCE) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%.17g expected=%.17g\n",
            message, (double)actual, (double)expected);
    }
}

static void test_geometric_bank(void) {
    geo_state_t initial[3];
    geo_instruction_t logical_instructions[1];
    geo_folded_program_t folded;
    geo_banked_instruction_t banked_instructions[1];
    geo_banked_ref_t refs[3];
    geo_banked_plan_workspace_t workspace;
    geo_banked_program_t program;
    geo_geometric_register_t geometrics[3];
    geo_banked_storage_t storage;
    geo_state_t result;
    geo_status_t status;

    initial[0] = geo_state_from_cl20(geo_cl20_basis_e1());
    initial[1] = geo_state_from_cl20(geo_cl20_basis_e2());
    initial[2] = geo_state_zero();

    logical_instructions[0].opcode = (uint8_t)GEO_OPCODE_OMEGA;
    logical_instructions[0].destination = 2u;
    logical_instructions[0].left = 0u;
    logical_instructions[0].right = 1u;
    logical_instructions[0].requested_lanes = GEO_LANE_GEOMETRIC;

    folded.program.instructions = logical_instructions;
    folded.program.instruction_count = 1u;
    folded.program.register_count = 3u;
    folded.initial_registers = initial;
    folded.root_register = 2u;
    folded.original_instruction_count = 1u;
    folded.folded_instruction_count = 1u;
    folded.folded_constant_nodes = 0u;
    folded.scalar_register_count = 0u;
    folded.geometric_register_count = 3u;
    folded.unified_register_count = 0u;
    folded.estimated_typed_bytes = 3u * sizeof(geo_geometric_register_t);
    folded.estimated_unified_bytes = 3u * sizeof(geo_state_t);

    workspace.instructions = banked_instructions;
    workspace.instruction_capacity = 1u;
    workspace.logical_refs = refs;
    workspace.logical_ref_capacity = 3u;

    status = geo_banked_plan(&folded, &workspace, &program);
    expect_true(status == GEO_STATUS_OK, "geometric bank plan succeeds");
    expect_true(program.scalar_count == 0u, "no scalar registers");
    expect_true(program.geometric_count == 3u, "three geometric registers");
    expect_true(program.unified_count == 0u, "no unified registers");
    expect_true(program.required_bytes < 3u * sizeof(geo_state_t),
        "banked geometric storage is smaller than unified storage");

    storage.scalars = NULL;
    storage.scalar_capacity = 0u;
    storage.geometrics = geometrics;
    storage.geometric_capacity = 3u;
    storage.unified = NULL;
    storage.unified_capacity = 0u;

    status = geo_banked_initialize(&folded, &program, refs, &storage);
    expect_true(status == GEO_STATUS_OK, "geometric bank initialization succeeds");
    status = geo_banked_execute(&program, &storage);
    expect_true(status == GEO_STATUS_OK, "geometric bank execution succeeds");
    status = geo_banked_read_state(&storage, program.root, &result);
    expect_true(status == GEO_STATUS_OK, "geometric root read succeeds");
    expect_true(geo_cl20_near(
        result.geometric.forward,
        geo_cl20_basis_e12(),
        GEO_TEST_TOLERANCE
    ), "banked e1*e2 equals e12");
    expect_true(geo_opposite_is_consistent(result.geometric, GEO_TEST_TOLERANCE),
        "banked opposite lane remains consistent");
}

static void test_unified_bank(void) {
    geo_state_t initial[3];
    geo_instruction_t logical_instructions[1];
    geo_folded_program_t folded;
    geo_banked_instruction_t banked_instructions[1];
    geo_banked_ref_t refs[3];
    geo_banked_plan_workspace_t workspace;
    geo_banked_program_t program;
    geo_state_t unified[3];
    geo_banked_storage_t storage;
    geo_state_t result;
    geo_status_t status;

    initial[0] = geo_state_from_cl20(geo_cl20_basis_e1());
    initial[0].scalar = (geo_real_t)0;
    initial[0].active_lanes = GEO_LANE_ALL;
    initial[1] = geo_state_from_cl20(geo_cl20_basis_e2());
    initial[1].scalar = (geo_real_t)1;
    initial[1].active_lanes = GEO_LANE_ALL;
    initial[2] = geo_state_zero();

    logical_instructions[0].opcode = (uint8_t)GEO_OPCODE_OMEGA;
    logical_instructions[0].destination = 2u;
    logical_instructions[0].left = 0u;
    logical_instructions[0].right = 1u;
    logical_instructions[0].requested_lanes = GEO_LANE_ALL;

    folded.program.instructions = logical_instructions;
    folded.program.instruction_count = 1u;
    folded.program.register_count = 3u;
    folded.initial_registers = initial;
    folded.root_register = 2u;
    folded.original_instruction_count = 1u;
    folded.folded_instruction_count = 1u;
    folded.folded_constant_nodes = 0u;
    folded.scalar_register_count = 0u;
    folded.geometric_register_count = 0u;
    folded.unified_register_count = 3u;
    folded.estimated_typed_bytes = 3u * sizeof(geo_state_t);
    folded.estimated_unified_bytes = 3u * sizeof(geo_state_t);

    workspace.instructions = banked_instructions;
    workspace.instruction_capacity = 1u;
    workspace.logical_refs = refs;
    workspace.logical_ref_capacity = 3u;

    status = geo_banked_plan(&folded, &workspace, &program);
    expect_true(status == GEO_STATUS_OK, "unified bank plan succeeds");
    expect_true(program.unified_count == 3u, "three unified registers");

    storage.scalars = NULL;
    storage.scalar_capacity = 0u;
    storage.geometrics = NULL;
    storage.geometric_capacity = 0u;
    storage.unified = unified;
    storage.unified_capacity = 3u;

    status = geo_banked_initialize(&folded, &program, refs, &storage);
    expect_true(status == GEO_STATUS_OK, "unified initialization succeeds");
    status = geo_banked_execute(&program, &storage);
    expect_true(status == GEO_STATUS_OK, "unified execution succeeds");
    status = geo_banked_read_state(&storage, program.root, &result);
    expect_true(status == GEO_STATUS_OK, "unified root read succeeds");
    expect_real(result.scalar, (geo_real_t)1, "unified scalar lane result");
    expect_true(geo_cl20_near(
        result.geometric.forward,
        geo_cl20_basis_e12(),
        GEO_TEST_TOLERANCE
    ), "unified geometric lane result");
}

int main(void) {
    test_geometric_bank();
    test_unified_bank();

    if (failures != 0) {
        fprintf(stderr, "%d banked assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All physically banked runtime tests passed.");
    return EXIT_SUCCESS;
}
