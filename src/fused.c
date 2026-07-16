#include "geo/fused.h"

static geo_status_t require_cl20(const geo_struct_value_t *value) {
    if (value == NULL) return GEO_STATUS_NULL_ARGUMENT;
    return value->kind == (uint8_t)GEO_STRUCT_VALUE_CL20 ? GEO_STATUS_OK : GEO_STATUS_BAD_OPCODE;
}

geo_status_t geo_fused_execute(
    const geo_fused_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
) {
    size_t pc;

    if (program == NULL || registers == NULL) return GEO_STATUS_NULL_ARGUMENT;
    if (program->register_count > register_capacity ||
        program->root_register >= program->register_count) {
        return GEO_STATUS_REGISTER_RANGE;
    }

    for (pc = 0u; pc < program->instruction_count; ++pc) {
        const geo_fused_instruction_t instruction = program->instructions[pc];
        geo_struct_value_t left;
        geo_struct_value_t right;
        geo_struct_value_t result;
        geo_status_t status;

        if (instruction.destination >= program->register_count ||
            instruction.left >= program->register_count ||
            instruction.right >= program->register_count ||
            instruction.auxiliary >= program->register_count) {
            return GEO_STATUS_REGISTER_RANGE;
        }

        left = registers[instruction.left];
        right = registers[instruction.right];
        status = require_cl20(&left);
        if (status != GEO_STATUS_OK) return status;
        status = require_cl20(&right);
        if (status != GEO_STATUS_OK) return status;

        switch ((geo_fused_opcode_t)instruction.opcode) {
            case GEO_FUSED_CL20_ADD:
                result = geo_struct_value_from_cl20(geo_cl20_add(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_CL20_SUBTRACT:
                result = geo_struct_value_from_cl20(geo_cl20_sub(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_GEOMETRIC_PRODUCT:
                result = geo_struct_value_from_cl20(geo_cl20_mul(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_VECTOR_DOT:
                result = geo_struct_value_from_scalar(geo_cl20_vector_dot(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_VECTOR_WEDGE:
                result = geo_struct_value_from_cl20(geo_geb_vector_wedge(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_COMMUTATOR:
                result = geo_struct_value_from_cl20(geo_geb_commutator(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_ANTICOMMUTATOR:
                result = geo_struct_value_from_cl20(geo_geb_anticommutator(left.as.cl20, right.as.cl20));
                break;
            case GEO_FUSED_ROTOR_ACTION:
                status = require_cl20(&registers[instruction.auxiliary]);
                if (status != GEO_STATUS_OK) return status;
                result = geo_struct_value_from_cl20(
                    geo_cl20_mul(
                        geo_cl20_mul(left.as.cl20, right.as.cl20),
                        registers[instruction.auxiliary].as.cl20
                    )
                );
                break;
            case GEO_FUSED_PROJECTION_NUMERATOR:
                result = geo_struct_value_from_scaled_cl20((geo_scaled_cl20_t){
                    geo_geb_projection_numerator(left.as.cl20, right.as.cl20),
                    {1, 1}
                });
                break;
            default:
                return GEO_STATUS_BAD_OPCODE;
        }

        registers[instruction.destination] = result;
    }

    return GEO_STATUS_OK;
}

geo_status_t geo_fused_program_for_target(
    uint8_t target_id,
    geo_fused_instruction_t *instruction,
    geo_fused_program_t *program
) {
    uint8_t opcode;
    size_t register_count = 3u;

    if (instruction == NULL || program == NULL) return GEO_STATUS_NULL_ARGUMENT;

    switch ((geo_geb_target_id_t)target_id) {
        case GEO_GEB_ADDITION: opcode = GEO_FUSED_CL20_ADD; break;
        case GEO_GEB_SUBTRACTION: opcode = GEO_FUSED_CL20_SUBTRACT; break;
        case GEO_GEB_GEOMETRIC_PRODUCT: opcode = GEO_FUSED_GEOMETRIC_PRODUCT; break;
        case GEO_GEB_VECTOR_DOT: opcode = GEO_FUSED_VECTOR_DOT; break;
        case GEO_GEB_VECTOR_WEDGE: opcode = GEO_FUSED_VECTOR_WEDGE; break;
        case GEO_GEB_COMMUTATOR: opcode = GEO_FUSED_COMMUTATOR; break;
        case GEO_GEB_ANTICOMMUTATOR: opcode = GEO_FUSED_ANTICOMMUTATOR; break;
        case GEO_GEB_PROJECTION_NUMERATOR: opcode = GEO_FUSED_PROJECTION_NUMERATOR; break;
        case GEO_GEB_ROTOR_ACTION:
            opcode = GEO_FUSED_ROTOR_ACTION;
            register_count = 4u;
            break;
        default:
            return GEO_STATUS_BAD_OPCODE;
    }

    instruction->opcode = opcode;
    instruction->destination = (uint8_t)(register_count - 1u);
    instruction->left = 0u;
    instruction->right = 1u;
    instruction->auxiliary = target_id == GEO_GEB_ROTOR_ACTION ? 2u : 0u;

    program->instructions = instruction;
    program->instruction_count = 1u;
    program->register_count = register_count;
    program->root_register = instruction->destination;
    return GEO_STATUS_OK;
}
