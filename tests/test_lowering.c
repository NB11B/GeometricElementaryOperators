#include "geo/lowering.h"

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

static void expect_state_near(
    geo_state_t actual,
    geo_state_t expected,
    const char *message
) {
    const int scalar_ok =
        ((actual.active_lanes & GEO_LANE_SCALAR) == 0u &&
         (expected.active_lanes & GEO_LANE_SCALAR) == 0u) ||
        (actual.scalar - expected.scalar <= GEO_TEST_TOLERANCE &&
         expected.scalar - actual.scalar <= GEO_TEST_TOLERANCE);
    const int geometric_ok = geo_cl20_near(
        actual.geometric.forward,
        expected.geometric.forward,
        GEO_TEST_TOLERANCE
    );

    if (!scalar_ok || !geometric_ok || actual.active_lanes != expected.active_lanes) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_scale_propagation(void) {
    geo_state_t initial[3];
    geo_instruction_t instructions[2];
    geo_scale_t scales[3];
    uint8_t known[3];
    geo_folded_program_t folded;
    geo_scale_workspace_t workspace;
    geo_scale_plan_t plan;
    geo_status_t status;

    initial[0] = geo_state_from_cl20(geo_cl20_basis_e1());
    initial[0].scale.numerator = 2;
    initial[0].scale.denominator = 1;
    initial[1] = geo_state_from_cl20(geo_cl20_basis_e2());
    initial[1].scale.numerator = 3;
    initial[1].scale.denominator = 5;
    initial[2] = geo_state_zero();

    instructions[0].opcode = (uint8_t)GEO_OPCODE_OMEGA;
    instructions[0].destination = 2u;
    instructions[0].left = 0u;
    instructions[0].right = 1u;
    instructions[0].requested_lanes = GEO_LANE_GEOMETRIC;

    folded.program.instructions = instructions;
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
    folded.estimated_typed_bytes = 0u;
    folded.estimated_unified_bytes = 0u;

    workspace.register_scales = scales;
    workspace.register_scale_capacity = 3u;
    workspace.known_flags = known;
    workspace.known_flag_capacity = 3u;

    status = geo_propagate_scales(&folded, &workspace, &plan);
    expect_true(status == GEO_STATUS_OK, "scale propagation succeeds");
    expect_true(plan.known_flags[2] != 0u, "result scale is known");
    expect_true(plan.register_scales[2].numerator == 6, "scale numerator propagated");
    expect_true(plan.register_scales[2].denominator == 5, "scale denominator propagated");
    expect_true(plan.propagated_geometric_results == 1u, "one geometric result propagated");
}

static void test_route_classification(void) {
    expect_true(
        geo_route_classify(geo_mat2_e11(), GEO_TEST_TOLERANCE) == GEO_ROUTE_KEEP_FIRST,
        "E11 lowers to keep-first route"
    );
    expect_true(
        geo_route_classify(geo_mat2_e12(), GEO_TEST_TOLERANCE) == GEO_ROUTE_SECOND_TO_FIRST,
        "E12 lowers to second-to-first route"
    );
    expect_true(
        geo_route_classify(geo_mat2_e21(), GEO_TEST_TOLERANCE) == GEO_ROUTE_FIRST_TO_SECOND,
        "E21 lowers to first-to-second route"
    );
    expect_true(
        geo_route_classify(geo_mat2_exchange(), GEO_TEST_TOLERANCE) == GEO_ROUTE_EXCHANGE,
        "exchange matrix lowers to exchange route"
    );
    expect_true(
        geo_route_classify(
            geo_mat2_make((geo_real_t)1, (geo_real_t)1, (geo_real_t)0, (geo_real_t)1),
            GEO_TEST_TOLERANCE
        ) == GEO_ROUTE_UNSUPPORTED,
        "non-routing matrix remains unsupported"
    );
}

static void test_route_application(void) {
    const geo_state_t first = geo_state_from_cl20(geo_cl20_basis_e1());
    const geo_state_t second = geo_state_from_cl20(geo_cl20_basis_e2());
    geo_state_t out_first;
    geo_state_t out_second;
    geo_status_t status;

    status = geo_route_apply(
        GEO_ROUTE_EXCHANGE,
        &first,
        &second,
        &out_first,
        &out_second
    );
    expect_true(status == GEO_STATUS_OK, "exchange route succeeds");
    expect_state_near(out_first, second, "exchange puts second in first lane");
    expect_state_near(out_second, first, "exchange puts first in second lane");

    status = geo_route_apply(
        GEO_ROUTE_SECOND_TO_FIRST,
        &first,
        &second,
        &out_first,
        &out_second
    );
    expect_true(status == GEO_STATUS_OK, "E12 route succeeds");
    expect_state_near(out_first, second, "E12 routes second to first");
    expect_true(out_second.active_lanes == GEO_LANE_NONE, "E12 clears second output");

    status = geo_route_apply(
        GEO_ROUTE_NEGATE,
        &first,
        &second,
        &out_first,
        &out_second
    );
    expect_true(status == GEO_STATUS_OK, "negation route succeeds");
    expect_state_near(
        out_first,
        geo_state_from_cl20(geo_cl20_neg(geo_cl20_basis_e1())),
        "negation route negates first"
    );
}

int main(void) {
    test_scale_propagation();
    test_route_classification();
    test_route_application();

    if (failures != 0) {
        fprintf(stderr, "%d test assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All lowering tests passed.");
    return EXIT_SUCCESS;
}
