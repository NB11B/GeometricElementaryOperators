#include "geo/fixed_program.h"

#include <stddef.h>

static geo_fixed_cl20_t geo_fixed_program_zero(void) {
    const geo_fixed_cl20_t value = {0, 0, 0, 0};
    return value;
}

static geo_fixed_program_status_t geo_fixed_program_map_status(
    geo_fixed_status_t status
) {
    switch (status) {
        case GEO_FIXED_OK:
            return GEO_FIXED_PROGRAM_OK;
        case GEO_FIXED_OVERFLOW:
            return GEO_FIXED_PROGRAM_OVERFLOW;
        case GEO_FIXED_DIVIDE_BY_ZERO:
            return GEO_FIXED_PROGRAM_DIVIDE_BY_ZERO;
        default:
            return GEO_FIXED_PROGRAM_OVERFLOW;
    }
}

static int geo_fixed_program_target_uses_left(uint8_t target_id) {
    return target_id < (uint8_t)GEO_GEB_ZERO ||
        target_id > (uint8_t)GEO_GEB_PSEUDOSCALAR;
}

static int geo_fixed_program_target_uses_right(uint8_t target_id) {
    switch ((geo_geb_target_id_t)target_id) {
        case GEO_GEB_ADDITION:
        case GEO_GEB_SUBTRACTION:
        case GEO_GEB_GEOMETRIC_PRODUCT:
        case GEO_GEB_REVERSE_PRODUCT:
        case GEO_GEB_VECTOR_DOT:
        case GEO_GEB_VECTOR_WEDGE:
        case GEO_GEB_COMMUTATOR:
        case GEO_GEB_ANTICOMMUTATOR:
        case GEO_GEB_DISTANCE_SQUARED:
        case GEO_GEB_PROJECTION_NUMERATOR:
        case GEO_GEB_REJECTION_NUMERATOR:
        case GEO_GEB_REFLECTION_NUMERATOR:
        case GEO_GEB_ROTOR_COMPOSITION:
        case GEO_GEB_ANGLE_COSINE_NUMERATOR:
            return 1;
        default:
            return 0;
    }
}

static int geo_fixed_program_target_uses_transform(uint8_t target_id) {
    return target_id == (uint8_t)GEO_GEB_ROTOR_ACTION ||
        target_id == (uint8_t)GEO_GEB_DILATION;
}

geo_fixed_geb_result_t geo_fixed_program_value_from_cl20(geo_fixed_cl20_t value) {
    geo_fixed_geb_result_t result;
    result.kind = (uint8_t)GEO_FIXED_RESULT_CL20;
    result.as.cl20 = value;
    return result;
}

geo_fixed_geb_result_t geo_fixed_program_value_from_scalar(geo_fixed_t value) {
    geo_fixed_geb_result_t result;
    result.kind = (uint8_t)GEO_FIXED_RESULT_SCALAR;
    result.as.scalar = value;
    return result;
}

geo_fixed_program_status_t geo_fixed_program_read_cl20(
    const geo_fixed_geb_result_t *value,
    geo_fixed_cl20_t *output
) {
    geo_fixed_cl20_t normalized;
    geo_fixed_status_t status;

    if (value == NULL || output == NULL) {
        return GEO_FIXED_PROGRAM_NULL_ARGUMENT;
    }

    switch ((geo_fixed_result_kind_t)value->kind) {
        case GEO_FIXED_RESULT_CL20:
            *output = value->as.cl20;
            return GEO_FIXED_PROGRAM_OK;

        case GEO_FIXED_RESULT_SCALAR:
            normalized = geo_fixed_program_zero();
            normalized.scalar = value->as.scalar;
            *output = normalized;
            return GEO_FIXED_PROGRAM_OK;

        case GEO_FIXED_RESULT_PROJECTIVE:
            if (value->as.projective.denominator == 0) {
                return GEO_FIXED_PROGRAM_INVALID_SCALE;
            }
            status = geo_fixed_div(
                value->as.projective.represented.scalar,
                value->as.projective.denominator,
                &normalized.scalar
            );
            if (status != GEO_FIXED_OK) return geo_fixed_program_map_status(status);
            status = geo_fixed_div(
                value->as.projective.represented.e1,
                value->as.projective.denominator,
                &normalized.e1
            );
            if (status != GEO_FIXED_OK) return geo_fixed_program_map_status(status);
            status = geo_fixed_div(
                value->as.projective.represented.e2,
                value->as.projective.denominator,
                &normalized.e2
            );
            if (status != GEO_FIXED_OK) return geo_fixed_program_map_status(status);
            status = geo_fixed_div(
                value->as.projective.represented.e12,
                value->as.projective.denominator,
                &normalized.e12
            );
            if (status != GEO_FIXED_OK) return geo_fixed_program_map_status(status);
            *output = normalized;
            return GEO_FIXED_PROGRAM_OK;

        case GEO_FIXED_RESULT_UNIPOTENT:
            *output = value->as.unipotent_payload;
            return GEO_FIXED_PROGRAM_OK;

        default:
            return GEO_FIXED_PROGRAM_TYPE_MISMATCH;
    }
}

geo_fixed_program_status_t geo_fixed_program_execute(
    const geo_fixed_program_t *program,
    geo_fixed_geb_result_t *registers,
    size_t register_capacity
) {
    size_t pc;

    if (program == NULL || registers == NULL) {
        return GEO_FIXED_PROGRAM_NULL_ARGUMENT;
    }
    if (program->instruction_count != 0u && program->instructions == NULL) {
        return GEO_FIXED_PROGRAM_NULL_ARGUMENT;
    }
    if (program->register_count == 0u ||
        program->register_count > register_capacity ||
        (size_t)program->root_register >= program->register_count) {
        return GEO_FIXED_PROGRAM_REGISTER_RANGE;
    }

    for (pc = 0u; pc < program->instruction_count; ++pc) {
        const geo_fixed_program_instruction_t instruction = program->instructions[pc];
        geo_fixed_cl20_t left = geo_fixed_program_zero();
        geo_fixed_cl20_t right = geo_fixed_program_zero();
        geo_fixed_cl20_t transform = geo_fixed_program_zero();
        geo_fixed_geb_result_t result;
        geo_fixed_program_status_t program_status;
        geo_fixed_status_t arithmetic_status;

        if (geo_geb36_target_info(instruction.target_id) == NULL) {
            return GEO_FIXED_PROGRAM_BAD_TARGET;
        }
        if ((size_t)instruction.destination >= program->register_count) {
            return GEO_FIXED_PROGRAM_REGISTER_RANGE;
        }

        if (geo_fixed_program_target_uses_left(instruction.target_id)) {
            if ((size_t)instruction.left >= program->register_count) {
                return GEO_FIXED_PROGRAM_REGISTER_RANGE;
            }
            program_status = geo_fixed_program_read_cl20(
                &registers[instruction.left],
                &left
            );
            if (program_status != GEO_FIXED_PROGRAM_OK) return program_status;
        }

        if (geo_fixed_program_target_uses_right(instruction.target_id)) {
            if ((size_t)instruction.right >= program->register_count) {
                return GEO_FIXED_PROGRAM_REGISTER_RANGE;
            }
            program_status = geo_fixed_program_read_cl20(
                &registers[instruction.right],
                &right
            );
            if (program_status != GEO_FIXED_PROGRAM_OK) return program_status;
        }

        if (geo_fixed_program_target_uses_transform(instruction.target_id)) {
            if ((size_t)instruction.transform >= program->register_count) {
                return GEO_FIXED_PROGRAM_REGISTER_RANGE;
            }
            program_status = geo_fixed_program_read_cl20(
                &registers[instruction.transform],
                &transform
            );
            if (program_status != GEO_FIXED_PROGRAM_OK) return program_status;
        }

        arithmetic_status = geo_fixed_geb36_execute(
            instruction.target_id,
            left,
            right,
            transform,
            &result
        );
        if (arithmetic_status != GEO_FIXED_OK) {
            return geo_fixed_program_map_status(arithmetic_status);
        }
        if (result.kind == (uint8_t)GEO_FIXED_RESULT_PROJECTIVE &&
            result.as.projective.denominator == 0) {
            return GEO_FIXED_PROGRAM_INVALID_SCALE;
        }

        registers[instruction.destination] = result;
    }

    return GEO_FIXED_PROGRAM_OK;
}
