#include "geo/folding.h"

#include <string.h>

static uint8_t geo_kind_from_lanes(uint8_t lanes) {
    const uint8_t active = (uint8_t)(lanes & GEO_LANE_ALL);

    if (active == GEO_LANE_SCALAR) {
        return (uint8_t)GEO_REGISTER_SCALAR;
    }
    if (active == GEO_LANE_GEOMETRIC) {
        return (uint8_t)GEO_REGISTER_GEOMETRIC;
    }
    if (active == GEO_LANE_ALL) {
        return (uint8_t)GEO_REGISTER_UNIFIED;
    }
    return (uint8_t)GEO_REGISTER_UNUSED;
}

static size_t geo_scalar_register_bytes(void) {
    return sizeof(geo_real_t);
}

static size_t geo_geometric_register_bytes(void) {
    return sizeof(geo_opposite_t) + sizeof(geo_scale_t);
}

geo_status_t geo_program_fold_constants(
    const geo_optimized_witness_t *input,
    const geo_state_t *terminal_values,
    const uint8_t *terminal_constant_flags,
    size_t terminal_count,
    geo_folding_workspace_t *workspace,
    geo_folded_program_t *output
) {
    size_t old_register;
    size_t instruction_index;
    size_t emitted_count = 0u;
    size_t next_register = 0u;
    size_t folded_count = 0u;
    size_t scalar_count = 0u;
    size_t geometric_count = 0u;
    size_t unified_count = 0u;
    const size_t old_register_count =
        input == NULL ? 0u : input->program.register_count;

    if (input == NULL || terminal_values == NULL ||
        terminal_constant_flags == NULL || workspace == NULL || output == NULL ||
        workspace->instructions == NULL || workspace->initial_registers == NULL ||
        workspace->old_to_new == NULL || workspace->constant_flags == NULL ||
        workspace->register_kinds == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (terminal_count > old_register_count ||
        workspace->old_to_new_capacity < old_register_count ||
        workspace->constant_flag_capacity < old_register_count ||
        workspace->register_kind_capacity < old_register_count ||
        workspace->initial_register_capacity < old_register_count ||
        workspace->instruction_capacity < input->program.instruction_count) {
        return GEO_STATUS_BUFFER_CAPACITY;
    }

    memset(workspace->constant_flags, 0, old_register_count * sizeof(uint8_t));
    memset(workspace->register_kinds, 0, old_register_count * sizeof(uint8_t));

    for (old_register = 0u; old_register < terminal_count; ++old_register) {
        workspace->old_to_new[old_register] = (uint8_t)next_register;
        workspace->initial_registers[next_register] = terminal_values[old_register];
        workspace->constant_flags[next_register] =
            terminal_constant_flags[old_register] != 0u ? 1u : 0u;
        workspace->register_kinds[next_register] =
            geo_kind_from_lanes(terminal_values[old_register].active_lanes);
        ++next_register;
    }

    for (instruction_index = 0u;
         instruction_index < input->program.instruction_count;
         ++instruction_index) {
        const geo_instruction_t source = input->program.instructions[instruction_index];
        geo_instruction_t *destination;
        uint8_t new_left;
        uint8_t new_right;
        uint8_t new_destination;

        if ((geo_opcode_t)source.opcode != GEO_OPCODE_OMEGA ||
            source.left >= old_register_count ||
            source.right >= old_register_count ||
            source.destination >= old_register_count) {
            return GEO_STATUS_BAD_OPCODE;
        }

        if (next_register >= workspace->initial_register_capacity ||
            next_register > (size_t)UINT8_MAX) {
            return GEO_STATUS_REGISTER_RANGE;
        }

        new_left = workspace->old_to_new[source.left];
        new_right = workspace->old_to_new[source.right];
        new_destination = (uint8_t)next_register;
        workspace->old_to_new[source.destination] = new_destination;
        workspace->initial_registers[next_register] = geo_state_zero();
        workspace->register_kinds[next_register] =
            geo_kind_from_lanes(source.requested_lanes);

        if (workspace->constant_flags[new_left] != 0u &&
            workspace->constant_flags[new_right] != 0u) {
            geo_status_t status = geo_omega_apply(
                &workspace->initial_registers[new_left],
                &workspace->initial_registers[new_right],
                source.requested_lanes,
                &workspace->initial_registers[new_destination]
            );
            if (status != GEO_STATUS_OK) {
                return status;
            }
            workspace->constant_flags[new_destination] = 1u;
            ++folded_count;
        } else {
            if (emitted_count >= workspace->instruction_capacity) {
                return GEO_STATUS_BUFFER_CAPACITY;
            }

            destination = &workspace->instructions[emitted_count];
            destination->opcode = source.opcode;
            destination->destination = new_destination;
            destination->left = new_left;
            destination->right = new_right;
            destination->requested_lanes = source.requested_lanes;
            workspace->constant_flags[new_destination] = 0u;
            ++emitted_count;
        }

        ++next_register;
    }

    for (old_register = 0u; old_register < next_register; ++old_register) {
        switch ((geo_register_kind_t)workspace->register_kinds[old_register]) {
            case GEO_REGISTER_SCALAR:
                ++scalar_count;
                break;
            case GEO_REGISTER_GEOMETRIC:
                ++geometric_count;
                break;
            case GEO_REGISTER_UNIFIED:
                ++unified_count;
                break;
            default:
                break;
        }
    }

    output->program.instructions = workspace->instructions;
    output->program.instruction_count = emitted_count;
    output->program.register_count = next_register;
    output->initial_registers = workspace->initial_registers;
    output->root_register = workspace->old_to_new[input->root_register];
    output->original_instruction_count = input->program.instruction_count;
    output->folded_instruction_count = emitted_count;
    output->folded_constant_nodes = folded_count;
    output->scalar_register_count = scalar_count;
    output->geometric_register_count = geometric_count;
    output->unified_register_count = unified_count;
    output->estimated_typed_bytes =
        scalar_count * geo_scalar_register_bytes() +
        geometric_count * geo_geometric_register_bytes() +
        unified_count * sizeof(geo_state_t);
    output->estimated_unified_bytes = next_register * sizeof(geo_state_t);

    return GEO_STATUS_OK;
}
