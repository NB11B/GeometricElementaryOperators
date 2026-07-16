#include "geo/structured_program.h"

static geo_status_t geo_struct_require_kind(
    const geo_struct_value_t *value,
    geo_struct_value_kind_t expected
) {
    if (value == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    return value->kind == (uint8_t)expected ? GEO_STATUS_OK : GEO_STATUS_BAD_OPCODE;
}

geo_struct_value_t geo_struct_value_from_cl20(geo_cl20_t value) {
    geo_struct_value_t result;
    result.kind = (uint8_t)GEO_STRUCT_VALUE_CL20;
    result.as.cl20 = value;
    return result;
}

geo_struct_value_t geo_struct_value_from_scalar(geo_real_t value) {
    geo_struct_value_t result;
    result.kind = (uint8_t)GEO_STRUCT_VALUE_SCALAR;
    result.as.scalar = value;
    return result;
}

geo_struct_value_t geo_struct_value_from_scaled_cl20(geo_scaled_cl20_t value) {
    geo_struct_value_t result;
    result.kind = (uint8_t)GEO_STRUCT_VALUE_SCALED_CL20;
    result.as.scaled_cl20 = value;
    return result;
}

geo_status_t geo_struct_program_execute(
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
) {
    size_t pc;

    if (program == NULL || registers == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    if (program->register_count > register_capacity ||
        program->root_register >= program->register_count) {
        return GEO_STATUS_REGISTER_RANGE;
    }

    for (pc = 0u; pc < program->instruction_count; ++pc) {
        const geo_struct_instruction_t instruction = program->instructions[pc];
        geo_struct_value_t left;
        geo_struct_value_t right;
        geo_struct_value_t result;
        geo_status_t status;

        if (instruction.destination >= program->register_count ||
            instruction.left >= program->register_count ||
            instruction.right >= program->register_count) {
            return GEO_STATUS_REGISTER_RANGE;
        }

        left = registers[instruction.left];
        right = registers[instruction.right];

        switch ((geo_struct_opcode_t)instruction.opcode) {
            case GEO_STRUCT_OP_COPY:
                result = left;
                break;

            case GEO_STRUCT_OP_CL20_NEGATE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_cl20(geo_cl20_neg(left.as.cl20));
                break;

            case GEO_STRUCT_OP_UNIPOTENT_ENCODE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result.kind = (uint8_t)GEO_STRUCT_VALUE_UNIPOTENT;
                result.as.unipotent = geo_unipotent_from_cl20(left.as.cl20);
                break;

            case GEO_STRUCT_OP_UNIPOTENT_COMPOSE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_UNIPOTENT);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_UNIPOTENT);
                if (status != GEO_STATUS_OK) return status;
                result.kind = (uint8_t)GEO_STRUCT_VALUE_UNIPOTENT;
                result.as.unipotent = geo_unipotent_mul(left.as.unipotent, right.as.unipotent);
                break;

            case GEO_STRUCT_OP_UNIPOTENT_EXTRACT:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_UNIPOTENT);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_cl20(geo_unipotent_extract(left.as.unipotent));
                break;

            case GEO_STRUCT_OP_ORDERED_PRODUCTS:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result.kind = (uint8_t)GEO_STRUCT_VALUE_ORDERED_PAIR;
                result.as.ordered = geo_ordered_products(left.as.cl20, right.as.cl20);
                break;

            case GEO_STRUCT_OP_HADAMARD_EXACT:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_ORDERED_PAIR);
                if (status != GEO_STATUS_OK) return status;
                result.kind = (uint8_t)GEO_STRUCT_VALUE_HADAMARD_PAIR;
                result.as.hadamard = geo_hadamard_mix_exact(left.as.ordered);
                break;

            case GEO_STRUCT_OP_HADAMARD_PROJECTIVE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_ORDERED_PAIR);
                if (status != GEO_STATUS_OK) return status;
                result.kind = (uint8_t)GEO_STRUCT_VALUE_HADAMARD_PAIR;
                result.as.hadamard = geo_hadamard_mix_projective(left.as.ordered);
                break;

            case GEO_STRUCT_OP_SELECT_SYMMETRIC:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_HADAMARD_PAIR);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scaled_cl20(left.as.hadamard.symmetric);
                break;

            case GEO_STRUCT_OP_SELECT_ANTISYMMETRIC:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_HADAMARD_PAIR);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scaled_cl20(left.as.hadamard.antisymmetric);
                break;

            case GEO_STRUCT_OP_CL20_SUBTRACT:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_cl20(geo_cl20_sub(left.as.cl20, right.as.cl20));
                break;

            case GEO_STRUCT_OP_CL20_SCALE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_SCALAR);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_cl20(geo_cl20_scale(left.as.cl20, right.as.scalar));
                break;

            case GEO_STRUCT_OP_SCALAR_EXTRACT:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scalar(left.as.cl20.scalar);
                break;

            case GEO_STRUCT_OP_SCALAR_MULTIPLY:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_SCALAR);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_SCALAR);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scalar(left.as.scalar * right.as.scalar);
                break;

            case GEO_STRUCT_OP_VECTOR_NORM_SQUARED:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scalar(geo_cl20_vector_norm_squared(left.as.cl20));
                break;

            case GEO_STRUCT_OP_DISTANCE_SQUARED:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scalar(
                    geo_cl20_vector_norm_squared(geo_cl20_sub(left.as.cl20, right.as.cl20))
                );
                break;

            case GEO_STRUCT_OP_PROJECTION_NUMERATOR:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scaled_cl20((geo_scaled_cl20_t){
                    geo_cl20_scale(right.as.cl20, geo_cl20_vector_dot(left.as.cl20, right.as.cl20)),
                    {1, 1}
                });
                break;

            case GEO_STRUCT_OP_REJECTION_NUMERATOR:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scaled_cl20((geo_scaled_cl20_t){
                    geo_cl20_sub(
                        geo_cl20_scale(left.as.cl20, geo_cl20_vector_norm_squared(right.as.cl20)),
                        geo_cl20_scale(right.as.cl20, geo_cl20_vector_dot(left.as.cl20, right.as.cl20))
                    ),
                    {1, 1}
                });
                break;

            case GEO_STRUCT_OP_REFLECTION_NUMERATOR:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scaled_cl20((geo_scaled_cl20_t){
                    geo_cl20_sub(
                        geo_cl20_scale(left.as.cl20, geo_cl20_vector_norm_squared(right.as.cl20)),
                        geo_cl20_scale(
                            right.as.cl20,
                            (geo_real_t)2 * geo_cl20_vector_dot(left.as.cl20, right.as.cl20)
                        )
                    ),
                    {1, 1}
                });
                break;

            case GEO_STRUCT_OP_VECTOR_INVERSE_PROJECTIVE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scaled_cl20((geo_scaled_cl20_t){left.as.cl20, {1, 1}});
                break;

            case GEO_STRUCT_OP_ANGLE_COSINE_NUMERATOR:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                status = geo_struct_require_kind(&right, GEO_STRUCT_VALUE_CL20);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_scalar(geo_cl20_vector_dot(left.as.cl20, right.as.cl20));
                break;

            case GEO_STRUCT_OP_SCALED_CL20_NORMALIZE:
                status = geo_struct_require_kind(&left, GEO_STRUCT_VALUE_SCALED_CL20);
                if (status != GEO_STATUS_OK) return status;
                {
                    geo_scaled_cl20_t normalized;
                    status = geo_scaled_cl20_normalize(&left.as.scaled_cl20, &normalized);
                    if (status != GEO_STATUS_OK) return status;
                    result = geo_struct_value_from_cl20(normalized.represented);
                }
                break;

            default:
                return GEO_STATUS_BAD_OPCODE;
        }

        registers[instruction.destination] = result;
    }

    return GEO_STATUS_OK;
}

geo_status_t geo_struct_read_cl20(
    const geo_struct_value_t *value,
    geo_cl20_t *output
) {
    if (value == NULL || output == NULL) return GEO_STATUS_NULL_ARGUMENT;
    if (value->kind != (uint8_t)GEO_STRUCT_VALUE_CL20) return GEO_STATUS_BAD_OPCODE;
    *output = value->as.cl20;
    return GEO_STATUS_OK;
}

geo_status_t geo_struct_read_scalar(
    const geo_struct_value_t *value,
    geo_real_t *output
) {
    if (value == NULL || output == NULL) return GEO_STATUS_NULL_ARGUMENT;
    if (value->kind != (uint8_t)GEO_STRUCT_VALUE_SCALAR) return GEO_STATUS_BAD_OPCODE;
    *output = value->as.scalar;
    return GEO_STATUS_OK;
}

geo_status_t geo_struct_read_scaled_cl20(
    const geo_struct_value_t *value,
    geo_scaled_cl20_t *output
) {
    if (value == NULL || output == NULL) return GEO_STATUS_NULL_ARGUMENT;
    if (value->kind != (uint8_t)GEO_STRUCT_VALUE_SCALED_CL20) return GEO_STATUS_BAD_OPCODE;
    *output = value->as.scaled_cl20;
    return GEO_STATUS_OK;
}
