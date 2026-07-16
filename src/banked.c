#include "geo/banked.h"

#include <limits.h>

static uint8_t geo_banked_kind_from_lanes(uint8_t lanes) {
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

static uint8_t geo_banked_initial_kind(const geo_state_t *state) {
    return geo_banked_kind_from_lanes(state->active_lanes);
}

static geo_status_t geo_banked_assign_ref(
    uint8_t kind,
    size_t *scalar_count,
    size_t *geometric_count,
    size_t *unified_count,
    geo_banked_ref_t *output
) {
    size_t index;

    if (scalar_count == NULL || geometric_count == NULL ||
        unified_count == NULL || output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    output->kind = kind;
    switch ((geo_register_kind_t)kind) {
        case GEO_REGISTER_SCALAR:
            index = (*scalar_count)++;
            break;
        case GEO_REGISTER_GEOMETRIC:
            index = (*geometric_count)++;
            break;
        case GEO_REGISTER_UNIFIED:
            index = (*unified_count)++;
            break;
        default:
            return GEO_STATUS_BAD_TREE;
    }

    if (index > (size_t)UINT8_MAX) {
        return GEO_STATUS_REGISTER_RANGE;
    }
    output->index = (uint8_t)index;
    return GEO_STATUS_OK;
}

geo_status_t geo_banked_plan(
    const geo_folded_program_t *folded,
    geo_banked_plan_workspace_t *workspace,
    geo_banked_program_t *output
) {
    size_t logical;
    size_t pc;
    size_t scalar_count = 0u;
    size_t geometric_count = 0u;
    size_t unified_count = 0u;

    if (folded == NULL || workspace == NULL || output == NULL ||
        folded->initial_registers == NULL ||
        workspace->instructions == NULL || workspace->logical_refs == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (workspace->instruction_capacity < folded->program.instruction_count ||
        workspace->logical_ref_capacity < folded->program.register_count) {
        return GEO_STATUS_BUFFER_CAPACITY;
    }

    for (logical = 0u; logical < folded->program.register_count; ++logical) {
        uint8_t kind = geo_banked_initial_kind(&folded->initial_registers[logical]);
        if (kind == (uint8_t)GEO_REGISTER_UNUSED) {
            /* Runtime destinations are typed from the instruction that writes them. */
            for (pc = 0u; pc < folded->program.instruction_count; ++pc) {
                const geo_instruction_t instruction = folded->program.instructions[pc];
                if ((size_t)instruction.destination == logical) {
                    kind = geo_banked_kind_from_lanes(instruction.requested_lanes);
                    break;
                }
            }
        }
        if (kind == (uint8_t)GEO_REGISTER_UNUSED) {
            return GEO_STATUS_BAD_TREE;
        }
        {
            geo_status_t status = geo_banked_assign_ref(
                kind,
                &scalar_count,
                &geometric_count,
                &unified_count,
                &workspace->logical_refs[logical]
            );
            if (status != GEO_STATUS_OK) {
                return status;
            }
        }
    }

    for (pc = 0u; pc < folded->program.instruction_count; ++pc) {
        const geo_instruction_t source = folded->program.instructions[pc];
        geo_banked_instruction_t *destination = &workspace->instructions[pc];

        if ((size_t)source.destination >= folded->program.register_count ||
            (size_t)source.left >= folded->program.register_count ||
            (size_t)source.right >= folded->program.register_count) {
            return GEO_STATUS_REGISTER_RANGE;
        }
        if ((geo_opcode_t)source.opcode != GEO_OPCODE_OMEGA) {
            return GEO_STATUS_BAD_OPCODE;
        }

        destination->requested_lanes = source.requested_lanes;
        destination->destination = workspace->logical_refs[source.destination];
        destination->left = workspace->logical_refs[source.left];
        destination->right = workspace->logical_refs[source.right];
    }

    if ((size_t)folded->root_register >= folded->program.register_count) {
        return GEO_STATUS_REGISTER_RANGE;
    }

    output->instructions = workspace->instructions;
    output->instruction_count = folded->program.instruction_count;
    output->root = workspace->logical_refs[folded->root_register];
    output->scalar_count = scalar_count;
    output->geometric_count = geometric_count;
    output->unified_count = unified_count;
    output->required_bytes =
        scalar_count * sizeof(geo_real_t) +
        geometric_count * sizeof(geo_geometric_register_t) +
        unified_count * sizeof(geo_state_t);

    return GEO_STATUS_OK;
}

static geo_status_t geo_banked_write_initial(
    geo_banked_storage_t *storage,
    geo_banked_ref_t reference,
    const geo_state_t *state
) {
    switch ((geo_register_kind_t)reference.kind) {
        case GEO_REGISTER_SCALAR:
            if ((size_t)reference.index >= storage->scalar_capacity) {
                return GEO_STATUS_BUFFER_CAPACITY;
            }
            storage->scalars[reference.index] = state->scalar;
            return GEO_STATUS_OK;

        case GEO_REGISTER_GEOMETRIC:
            if ((size_t)reference.index >= storage->geometric_capacity) {
                return GEO_STATUS_BUFFER_CAPACITY;
            }
            storage->geometrics[reference.index].value = state->geometric;
            storage->geometrics[reference.index].scale = state->scale;
            return GEO_STATUS_OK;

        case GEO_REGISTER_UNIFIED:
            if ((size_t)reference.index >= storage->unified_capacity) {
                return GEO_STATUS_BUFFER_CAPACITY;
            }
            storage->unified[reference.index] = *state;
            return GEO_STATUS_OK;

        default:
            return GEO_STATUS_BAD_TREE;
    }
}

geo_status_t geo_banked_initialize(
    const geo_folded_program_t *folded,
    const geo_banked_program_t *program,
    const geo_banked_ref_t *logical_refs,
    geo_banked_storage_t *storage
) {
    size_t logical;

    if (folded == NULL || program == NULL || logical_refs == NULL || storage == NULL ||
        folded->initial_registers == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    if ((program->scalar_count > 0u && storage->scalars == NULL) ||
        (program->geometric_count > 0u && storage->geometrics == NULL) ||
        (program->unified_count > 0u && storage->unified == NULL)) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    if (storage->scalar_capacity < program->scalar_count ||
        storage->geometric_capacity < program->geometric_count ||
        storage->unified_capacity < program->unified_count) {
        return GEO_STATUS_BUFFER_CAPACITY;
    }

    for (logical = 0u; logical < folded->program.register_count; ++logical) {
        geo_status_t status = geo_banked_write_initial(
            storage,
            logical_refs[logical],
            &folded->initial_registers[logical]
        );
        if (status != GEO_STATUS_OK) {
            return status;
        }
    }
    return GEO_STATUS_OK;
}

geo_status_t geo_banked_read_state(
    const geo_banked_storage_t *storage,
    geo_banked_ref_t reference,
    geo_state_t *output
) {
    geo_state_t result;

    if (storage == NULL || output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    result = geo_state_zero();

    switch ((geo_register_kind_t)reference.kind) {
        case GEO_REGISTER_SCALAR:
            if (storage->scalars == NULL ||
                (size_t)reference.index >= storage->scalar_capacity) {
                return GEO_STATUS_REGISTER_RANGE;
            }
            result.scalar = storage->scalars[reference.index];
            result.active_lanes = GEO_LANE_SCALAR;
            break;

        case GEO_REGISTER_GEOMETRIC:
            if (storage->geometrics == NULL ||
                (size_t)reference.index >= storage->geometric_capacity) {
                return GEO_STATUS_REGISTER_RANGE;
            }
            result.geometric = storage->geometrics[reference.index].value;
            result.scale = storage->geometrics[reference.index].scale;
            result.active_lanes = GEO_LANE_GEOMETRIC;
            break;

        case GEO_REGISTER_UNIFIED:
            if (storage->unified == NULL ||
                (size_t)reference.index >= storage->unified_capacity) {
                return GEO_STATUS_REGISTER_RANGE;
            }
            result = storage->unified[reference.index];
            break;

        default:
            return GEO_STATUS_BAD_TREE;
    }

    *output = result;
    return GEO_STATUS_OK;
}

static geo_status_t geo_banked_write_state(
    geo_banked_storage_t *storage,
    geo_banked_ref_t reference,
    const geo_state_t *state
) {
    return geo_banked_write_initial(storage, reference, state);
}

geo_status_t geo_banked_execute(
    const geo_banked_program_t *program,
    geo_banked_storage_t *storage
) {
    size_t pc;

    if (program == NULL || storage == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    for (pc = 0u; pc < program->instruction_count; ++pc) {
        const geo_banked_instruction_t instruction = program->instructions[pc];
        geo_state_t left;
        geo_state_t right;
        geo_state_t result;
        geo_status_t status;

        status = geo_banked_read_state(storage, instruction.left, &left);
        if (status != GEO_STATUS_OK) {
            return status;
        }
        status = geo_banked_read_state(storage, instruction.right, &right);
        if (status != GEO_STATUS_OK) {
            return status;
        }
        status = geo_omega_apply(
            &left,
            &right,
            instruction.requested_lanes,
            &result
        );
        if (status != GEO_STATUS_OK) {
            return status;
        }
        status = geo_banked_write_state(storage, instruction.destination, &result);
        if (status != GEO_STATUS_OK) {
            return status;
        }
    }

    return GEO_STATUS_OK;
}
