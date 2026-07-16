#include "geo/fixed_banked.h"

#include <stddef.h>

static geo_fixed_banked_status_t geo_fixed_banked_from_omega(
    geo_fixed_omega_status_t status
) {
    switch (status) {
        case GEO_FIXED_OMEGA_OK:
            return GEO_FIXED_BANKED_OK;
        case GEO_FIXED_OMEGA_OVERFLOW:
            return GEO_FIXED_BANKED_OVERFLOW;
        case GEO_FIXED_OMEGA_DIVIDE_BY_ZERO:
            return GEO_FIXED_BANKED_DIVIDE_BY_ZERO;
        case GEO_FIXED_OMEGA_LOG_DOMAIN:
            return GEO_FIXED_BANKED_LOG_DOMAIN;
        case GEO_FIXED_OMEGA_NULL_ARGUMENT:
            return GEO_FIXED_BANKED_NULL_ARGUMENT;
        case GEO_FIXED_OMEGA_REGISTER_RANGE:
            return GEO_FIXED_BANKED_STORAGE_RANGE;
        case GEO_FIXED_OMEGA_BAD_OPCODE:
            return GEO_FIXED_BANKED_BAD_OPCODE;
        case GEO_FIXED_OMEGA_BAD_LANES:
            return GEO_FIXED_BANKED_BAD_LANES;
        case GEO_FIXED_OMEGA_INVALID_SCALE:
            return GEO_FIXED_BANKED_INVALID_SCALE;
        default:
            return GEO_FIXED_BANKED_BAD_OPCODE;
    }
}

static geo_fixed_banked_status_t geo_fixed_banked_validate_reference(
    const geo_fixed_banked_storage_t *storage,
    geo_fixed_banked_ref_t reference
) {
    if (storage == NULL) return GEO_FIXED_BANKED_NULL_ARGUMENT;

    switch ((geo_fixed_bank_kind_t)reference.kind) {
        case GEO_FIXED_BANK_SCALAR:
            if ((size_t)reference.index >= storage->scalar_capacity) {
                return GEO_FIXED_BANKED_STORAGE_RANGE;
            }
            return storage->scalars == NULL
                ? GEO_FIXED_BANKED_NULL_ARGUMENT
                : GEO_FIXED_BANKED_OK;

        case GEO_FIXED_BANK_GEOMETRIC:
            if ((size_t)reference.index >= storage->geometric_capacity) {
                return GEO_FIXED_BANKED_STORAGE_RANGE;
            }
            return storage->geometrics == NULL
                ? GEO_FIXED_BANKED_NULL_ARGUMENT
                : GEO_FIXED_BANKED_OK;

        case GEO_FIXED_BANK_UNIFIED:
            if ((size_t)reference.index >= storage->unified_capacity) {
                return GEO_FIXED_BANKED_STORAGE_RANGE;
            }
            return storage->unified == NULL
                ? GEO_FIXED_BANKED_NULL_ARGUMENT
                : GEO_FIXED_BANKED_OK;

        default:
            return GEO_FIXED_BANKED_TYPE_MISMATCH;
    }
}

static uint8_t geo_fixed_banked_lanes_for_kind(uint8_t kind) {
    switch ((geo_fixed_bank_kind_t)kind) {
        case GEO_FIXED_BANK_SCALAR:
            return GEO_FIXED_LANE_SCALAR;
        case GEO_FIXED_BANK_GEOMETRIC:
            return GEO_FIXED_LANE_GEOMETRIC;
        case GEO_FIXED_BANK_UNIFIED:
            return GEO_FIXED_LANE_ALL;
        default:
            return GEO_FIXED_LANE_NONE;
    }
}

geo_fixed_banked_status_t geo_fixed_banked_read_state(
    const geo_fixed_banked_storage_t *storage,
    geo_fixed_banked_ref_t reference,
    geo_fixed_state_t *output
) {
    geo_fixed_state_t value;
    geo_fixed_banked_status_t status;

    if (output == NULL) return GEO_FIXED_BANKED_NULL_ARGUMENT;
    status = geo_fixed_banked_validate_reference(storage, reference);
    if (status != GEO_FIXED_BANKED_OK) return status;

    value = geo_fixed_state_zero();
    switch ((geo_fixed_bank_kind_t)reference.kind) {
        case GEO_FIXED_BANK_SCALAR:
            value.scalar = storage->scalars[reference.index];
            value.active_lanes = GEO_FIXED_LANE_SCALAR;
            break;

        case GEO_FIXED_BANK_GEOMETRIC:
            value.geometric = storage->geometrics[reference.index].value;
            value.scale = storage->geometrics[reference.index].scale;
            value.active_lanes = GEO_FIXED_LANE_GEOMETRIC;
            break;

        case GEO_FIXED_BANK_UNIFIED:
            value = storage->unified[reference.index];
            if (value.active_lanes != GEO_FIXED_LANE_ALL) {
                return GEO_FIXED_BANKED_TYPE_MISMATCH;
            }
            break;

        default:
            return GEO_FIXED_BANKED_TYPE_MISMATCH;
    }

    *output = value;
    return GEO_FIXED_BANKED_OK;
}

geo_fixed_banked_status_t geo_fixed_banked_write_state(
    geo_fixed_banked_storage_t *storage,
    geo_fixed_banked_ref_t reference,
    const geo_fixed_state_t *value
) {
    geo_fixed_banked_status_t status;
    const uint8_t expected_lanes = geo_fixed_banked_lanes_for_kind(reference.kind);

    if (value == NULL) return GEO_FIXED_BANKED_NULL_ARGUMENT;
    status = geo_fixed_banked_validate_reference(storage, reference);
    if (status != GEO_FIXED_BANKED_OK) return status;
    if (expected_lanes == GEO_FIXED_LANE_NONE ||
        value->active_lanes != expected_lanes) {
        return GEO_FIXED_BANKED_TYPE_MISMATCH;
    }

    switch ((geo_fixed_bank_kind_t)reference.kind) {
        case GEO_FIXED_BANK_SCALAR:
            storage->scalars[reference.index] = value->scalar;
            break;

        case GEO_FIXED_BANK_GEOMETRIC:
            storage->geometrics[reference.index].value = value->geometric;
            storage->geometrics[reference.index].scale = value->scale;
            break;

        case GEO_FIXED_BANK_UNIFIED:
            storage->unified[reference.index] = *value;
            break;

        default:
            return GEO_FIXED_BANKED_TYPE_MISMATCH;
    }
    return GEO_FIXED_BANKED_OK;
}

geo_fixed_banked_status_t geo_fixed_banked_execute(
    const geo_fixed_banked_program_t *program,
    geo_fixed_banked_storage_t *storage
) {
    size_t pc;
    geo_fixed_banked_status_t status;

    if (program == NULL || storage == NULL) {
        return GEO_FIXED_BANKED_NULL_ARGUMENT;
    }
    if (program->instruction_count != 0u && program->instructions == NULL) {
        return GEO_FIXED_BANKED_NULL_ARGUMENT;
    }
    if (program->scalar_count > storage->scalar_capacity ||
        program->geometric_count > storage->geometric_capacity ||
        program->unified_count > storage->unified_capacity) {
        return GEO_FIXED_BANKED_STORAGE_RANGE;
    }

    status = geo_fixed_banked_validate_reference(storage, program->root);
    if (status != GEO_FIXED_BANKED_OK) return status;

    for (pc = 0u; pc < program->instruction_count; ++pc) {
        const geo_fixed_banked_instruction_t instruction = program->instructions[pc];
        geo_fixed_state_t left;
        geo_fixed_state_t right;
        geo_fixed_state_t result;
        const uint8_t destination_lanes =
            geo_fixed_banked_lanes_for_kind(instruction.destination.kind);
        geo_fixed_omega_status_t omega_status;

        if (instruction.requested_lanes == GEO_FIXED_LANE_NONE ||
            (instruction.requested_lanes & (uint8_t)~GEO_FIXED_LANE_ALL) != 0u) {
            return GEO_FIXED_BANKED_BAD_LANES;
        }
        if (destination_lanes != instruction.requested_lanes) {
            return GEO_FIXED_BANKED_TYPE_MISMATCH;
        }

        status = geo_fixed_banked_read_state(storage, instruction.left, &left);
        if (status != GEO_FIXED_BANKED_OK) return status;
        status = geo_fixed_banked_read_state(storage, instruction.right, &right);
        if (status != GEO_FIXED_BANKED_OK) return status;
        status = geo_fixed_banked_validate_reference(storage, instruction.destination);
        if (status != GEO_FIXED_BANKED_OK) return status;

        omega_status = geo_fixed_omega_apply(
            &left,
            &right,
            instruction.requested_lanes,
            &result
        );
        if (omega_status != GEO_FIXED_OMEGA_OK) {
            return geo_fixed_banked_from_omega(omega_status);
        }

        status = geo_fixed_banked_write_state(
            storage,
            instruction.destination,
            &result
        );
        if (status != GEO_FIXED_BANKED_OK) return status;
    }

    return GEO_FIXED_BANKED_OK;
}
