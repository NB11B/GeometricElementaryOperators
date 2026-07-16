#include "geo/optimizer.h"

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

static void expect_cl20(geo_cl20_t actual, geo_cl20_t expected, const char *message) {
    if (!geo_cl20_near(actual, expected, GEO_TEST_TOLERANCE)) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_dead_code_and_duplicate_elimination(void) {
    const geo_witness_node_t nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0u, 1u, 0u},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0u, 1u, 0u},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 1u, 0u, 0u},
        {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 2u, 3u, 0u}
    };
    const geo_witness_tree_t tree = {
        nodes,
        sizeof(nodes) / sizeof(nodes[0]),
        2u,
        5u
    };
    geo_instruction_t instructions[4];
    uint8_t node_registers[6];
    uint8_t live_lanes[6];
    uint16_t representatives[6];
    geo_optimizer_workspace_t workspace = {
        instructions,
        sizeof(instructions) / sizeof(instructions[0]),
        node_registers,
        sizeof(node_registers) / sizeof(node_registers[0]),
        live_lanes,
        sizeof(live_lanes) / sizeof(live_lanes[0]),
        representatives,
        sizeof(representatives) / sizeof(representatives[0])
    };
    geo_optimized_witness_t optimized;
    geo_state_t registers[4];
    geo_status_t status;

    status = geo_witness_compile_optimized(&tree, &workspace, &optimized);
    expect_true(status == GEO_STATUS_OK, "optimized compilation succeeds");
    expect_true(optimized.original_instruction_count == 4u, "original instruction count");
    expect_true(optimized.optimized_instruction_count == 2u, "optimized instruction count");
    expect_true(optimized.eliminated_dead_nodes == 1u, "dead branch eliminated");
    expect_true(optimized.eliminated_duplicate_nodes == 1u, "duplicate node eliminated");
    expect_true(optimized.program.register_count == 4u, "registers compacted");

    registers[0] = geo_state_from_cl20(geo_cl20_basis_e1());
    registers[1] = geo_state_from_cl20(geo_cl20_basis_e2());
    registers[2] = geo_state_zero();
    registers[3] = geo_state_zero();

    status = geo_program_execute(&optimized.program, registers, 4u);
    expect_true(status == GEO_STATUS_OK, "optimized program executes");
    expect_cl20(
        registers[optimized.root_register].geometric.forward,
        geo_cl20_neg(geo_cl20_one()),
        "(e1 e2)(e1 e2) = -1 after optimization"
    );
}

static void test_lane_liveness(void) {
    const geo_witness_node_t nodes[] = {
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
        {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
        {GEO_WITNESS_OMEGA, GEO_LANE_ALL, 0u, 1u, 0u}
    };
    const geo_witness_tree_t tree = {nodes, 3u, 2u, 2u};
    geo_instruction_t instructions[1];
    uint8_t node_registers[3];
    uint8_t live_lanes[3];
    uint16_t representatives[3];
    geo_optimizer_workspace_t workspace = {
        instructions, 1u,
        node_registers, 3u,
        live_lanes, 3u,
        representatives, 3u
    };
    geo_optimized_witness_t optimized;
    geo_status_t status = geo_witness_compile_optimized(&tree, &workspace, &optimized);

    expect_true(status == GEO_STATUS_OK, "all-lane tree compiles");
    expect_true(instructions[0].requested_lanes == GEO_LANE_ALL, "root lane requirements preserved");
    expect_true(live_lanes[0] == GEO_LANE_ALL, "left terminal receives both live lanes");
    expect_true(live_lanes[1] == GEO_LANE_ALL, "right terminal receives both live lanes");
}

int main(void) {
    test_dead_code_and_duplicate_elimination();
    test_lane_liveness();

    if (failures != 0) {
        fprintf(stderr, "%d optimizer assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All witness optimizer tests passed.");
    return EXIT_SUCCESS;
}
