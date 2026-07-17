#include "geo/banked.h"
#include "geo/fixed.h"
#include "geo/folding.h"
#include "geo/structured_program.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void expect_status(geo_status_t actual, geo_status_t expected, const char *message) {
    if (actual != expected) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%d expected=%d\n", message, (int)actual, (int)expected);
    }
}

static void expect_fixed(geo_fixed_status_t actual, geo_fixed_status_t expected, const char *message) {
    if (actual != expected) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%d expected=%d\n", message, (int)actual, (int)expected);
    }
}

static void test_null_instruction_storage(void) {
    geo_state_t omega_registers[1];
    geo_struct_value_t structured_registers[1];
    geo_banked_storage_t storage = {0};
    const geo_program_t omega = {NULL, 1u, 1u};
    const geo_struct_program_t structured = {NULL, 1u, 1u, 0u};
    geo_banked_program_t banked;

    omega_registers[0] = geo_state_from_scalar((geo_real_t)1);
    structured_registers[0] = geo_struct_value_from_scalar((geo_real_t)1);
    banked.instructions = NULL;
    banked.instruction_count = 1u;
    banked.root.kind = GEO_REGISTER_SCALAR;
    banked.root.index = 0u;
    banked.scalar_count = 0u;
    banked.geometric_count = 0u;
    banked.unified_count = 0u;
    banked.required_bytes = 0u;

    expect_status(geo_program_execute(&omega, omega_registers, 1u),
        GEO_STATUS_NULL_ARGUMENT, "Omega executor rejects missing instruction storage");
    expect_status(geo_struct_program_execute(&structured, structured_registers, 1u),
        GEO_STATUS_NULL_ARGUMENT, "structured executor rejects missing instruction storage");
    expect_status(geo_banked_execute(&banked, &storage),
        GEO_STATUS_NULL_ARGUMENT, "banked executor rejects missing instruction storage");
}

static void test_missing_terminal_image(void) {
    geo_instruction_t instruction;
    geo_optimized_witness_t optimized;
    geo_state_t terminal = geo_state_from_cl20(geo_cl20_basis_e1());
    uint8_t constant_flag = 0u;
    geo_instruction_t folded_instructions[2];
    geo_state_t initial_registers[3];
    uint8_t old_to_new[3];
    uint8_t constant_flags[3];
    uint8_t register_kinds[3];
    geo_folding_workspace_t workspace;
    geo_folded_program_t output;

    instruction.opcode = GEO_OPCODE_OMEGA;
    instruction.destination = 2u;
    instruction.left = 0u;
    instruction.right = 1u;
    instruction.requested_lanes = GEO_LANE_GEOMETRIC;

    optimized.program.instructions = &instruction;
    optimized.program.instruction_count = 1u;
    optimized.program.register_count = 3u;
    optimized.root_register = 2u;
    optimized.original_instruction_count = 1u;
    optimized.optimized_instruction_count = 1u;
    optimized.eliminated_dead_nodes = 0u;
    optimized.eliminated_duplicate_nodes = 0u;

    workspace.instructions = folded_instructions;
    workspace.instruction_capacity = 2u;
    workspace.initial_registers = initial_registers;
    workspace.initial_register_capacity = 3u;
    workspace.old_to_new = old_to_new;
    workspace.old_to_new_capacity = 3u;
    workspace.constant_flags = constant_flags;
    workspace.constant_flag_capacity = 3u;
    workspace.register_kinds = register_kinds;
    workspace.register_kind_capacity = 3u;

    expect_status(
        geo_program_fold_constants(&optimized, &terminal, &constant_flag, 1u, &workspace, &output),
        GEO_STATUS_BAD_TREE,
        "constant folding rejects incomplete terminal image"
    );
}

static void test_negative_fixed_arithmetic(void) {
    geo_fixed_t minus_one;
    geo_fixed_t two;
    geo_fixed_t result;

    expect_fixed(geo_fixed_from_double(-1.0, &minus_one), GEO_FIXED_OK, "encode -1");
    expect_fixed(geo_fixed_from_double(2.0, &two), GEO_FIXED_OK, "encode 2");
    expect_fixed(geo_fixed_div(minus_one, two, &result), GEO_FIXED_OK, "negative fixed division");
    if (fabs(geo_fixed_to_double(result) + 0.5) > 1e-6) {
        ++failures;
        fprintf(stderr, "FAIL: -1/2 fixed result %.17g\n", geo_fixed_to_double(result));
    }
    expect_fixed(geo_fixed_mul(minus_one, two, &result), GEO_FIXED_OK, "negative fixed multiplication");
    if (fabs(geo_fixed_to_double(result) + 2.0) > 1e-6) {
        ++failures;
        fprintf(stderr, "FAIL: -1*2 fixed result %.17g\n", geo_fixed_to_double(result));
    }
}

int main(void) {
    test_null_instruction_storage();
    test_missing_terminal_image();
    test_negative_fixed_arithmetic();

    if (failures != 0) {
        fprintf(stderr, "%d safety assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All malformed-program and fixed-point safety tests passed.");
    return EXIT_SUCCESS;
}
