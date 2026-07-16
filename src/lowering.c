#include "geo/lowering.h"

#include <limits.h>

static int64_t geo_lower_abs_i64(int64_t value) {
    return value < 0 ? -value : value;
}

static int64_t geo_lower_gcd_i64(int64_t a, int64_t b) {
    a = geo_lower_abs_i64(a);
    b = geo_lower_abs_i64(b);

    while (b != 0) {
        const int64_t remainder = a % b;
        a = b;
        b = remainder;
    }

    return a == 0 ? 1 : a;
}

static geo_status_t geo_lower_scale_mul(
    geo_scale_t left,
    geo_scale_t right,
    geo_scale_t *output
) {
    int64_t numerator;
    int64_t denominator;
    int64_t divisor;

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    if (left.denominator == 0 || right.denominator == 0) {
        return GEO_STATUS_SCALE_OVERFLOW;
    }

    numerator = (int64_t)left.numerator * (int64_t)right.numerator;
    denominator = (int64_t)left.denominator * (int64_t)right.denominator;

    if (denominator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }

    divisor = geo_lower_gcd_i64(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;

    if (numerator < INT32_MIN || numerator > INT32_MAX ||
        denominator < 1 || denominator > INT32_MAX) {
        return GEO_STATUS_SCALE_OVERFLOW;
    }

    output->numerator = (int32_t)numerator;
    output->denominator = (int32_t)denominator;
    return GEO_STATUS_OK;
}

static int geo_scale_is_unit(geo_scale_t scale) {
    return scale.numerator == 1 && scale.denominator == 1;
}

geo_status_t geo_propagate_scales(
    const geo_folded_program_t *program,
    geo_scale_workspace_t *workspace,
    geo_scale_plan_t *output
) {
    size_t index;
    size_t propagated = 0u;
    size_t unit_count = 0u;

    if (program == NULL || workspace == NULL || output == NULL ||
        program->initial_registers == NULL ||
        workspace->register_scales == NULL || workspace->known_flags == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (workspace->register_scale_capacity < program->program.register_count ||
        workspace->known_flag_capacity < program->program.register_count) {
        return GEO_STATUS_BUFFER_CAPACITY;
    }

    for (index = 0u; index < program->program.register_count; ++index) {
        const geo_state_t state = program->initial_registers[index];
        workspace->register_scales[index] = geo_scale_one();
        workspace->known_flags[index] = 0u;

        if ((state.active_lanes & GEO_LANE_GEOMETRIC) != 0u) {
            if (state.scale.denominator == 0) {
                return GEO_STATUS_SCALE_OVERFLOW;
            }
            workspace->register_scales[index] = state.scale;
            workspace->known_flags[index] = 1u;
        } else if ((state.active_lanes & GEO_LANE_SCALAR) != 0u) {
            workspace->known_flags[index] = 1u;
        }
    }

    for (index = 0u; index < program->program.instruction_count; ++index) {
        const geo_instruction_t instruction = program->program.instructions[index];
        const uint8_t lanes = (uint8_t)(instruction.requested_lanes & GEO_LANE_ALL);

        if (instruction.destination >= program->program.register_count ||
            instruction.left >= program->program.register_count ||
            instruction.right >= program->program.register_count) {
            return GEO_STATUS_REGISTER_RANGE;
        }

        if ((lanes & GEO_LANE_GEOMETRIC) != 0u) {
            geo_status_t status;
            if (workspace->known_flags[instruction.left] == 0u ||
                workspace->known_flags[instruction.right] == 0u) {
                workspace->known_flags[instruction.destination] = 0u;
                continue;
            }

            status = geo_lower_scale_mul(
                workspace->register_scales[instruction.left],
                workspace->register_scales[instruction.right],
                &workspace->register_scales[instruction.destination]
            );
            if (status != GEO_STATUS_OK) {
                return status;
            }
            workspace->known_flags[instruction.destination] = 1u;
            ++propagated;
        } else {
            workspace->register_scales[instruction.destination] = geo_scale_one();
            workspace->known_flags[instruction.destination] = 1u;
        }
    }

    for (index = 0u; index < program->program.register_count; ++index) {
        if (workspace->known_flags[index] != 0u &&
            geo_scale_is_unit(workspace->register_scales[index])) {
            ++unit_count;
        }
    }

    output->register_scales = workspace->register_scales;
    output->known_flags = workspace->known_flags;
    output->register_count = program->program.register_count;
    output->propagated_geometric_results = propagated;
    output->unit_scale_results = unit_count;
    return GEO_STATUS_OK;
}

geo_route_kind_t geo_route_classify(geo_mat2_t control, geo_real_t tolerance) {
    if (geo_mat2_near(control, geo_mat2_zero(), tolerance)) {
        return GEO_ROUTE_ZERO;
    }
    if (geo_mat2_near(control, geo_mat2_identity(), tolerance)) {
        return GEO_ROUTE_IDENTITY;
    }
    if (geo_mat2_near(control, geo_mat2_neg_identity(), tolerance)) {
        return GEO_ROUTE_NEGATE;
    }
    if (geo_mat2_near(control, geo_mat2_e11(), tolerance)) {
        return GEO_ROUTE_KEEP_FIRST;
    }
    if (geo_mat2_near(control, geo_mat2_e22(), tolerance)) {
        return GEO_ROUTE_KEEP_SECOND;
    }
    if (geo_mat2_near(control, geo_mat2_e12(), tolerance)) {
        return GEO_ROUTE_SECOND_TO_FIRST;
    }
    if (geo_mat2_near(control, geo_mat2_e21(), tolerance)) {
        return GEO_ROUTE_FIRST_TO_SECOND;
    }
    if (geo_mat2_near(control, geo_mat2_exchange(), tolerance)) {
        return GEO_ROUTE_EXCHANGE;
    }
    return GEO_ROUTE_UNSUPPORTED;
}

static geo_state_t geo_route_negate_state(geo_state_t input) {
    geo_state_t result = input;

    if ((input.active_lanes & GEO_LANE_SCALAR) != 0u) {
        result.scalar = -input.scalar;
    }
    if ((input.active_lanes & GEO_LANE_GEOMETRIC) != 0u) {
        result.geometric.forward = geo_cl20_neg(input.geometric.forward);
        result.geometric.reverse = geo_cl20_neg(input.geometric.reverse);
    }

    return result;
}

geo_status_t geo_route_apply(
    geo_route_kind_t route,
    const geo_state_t *first,
    const geo_state_t *second,
    geo_state_t *out_first,
    geo_state_t *out_second
) {
    const geo_state_t zero = geo_state_zero();

    if (first == NULL || second == NULL || out_first == NULL || out_second == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    switch (route) {
        case GEO_ROUTE_ZERO:
            *out_first = zero;
            *out_second = zero;
            break;
        case GEO_ROUTE_IDENTITY:
            *out_first = *first;
            *out_second = *second;
            break;
        case GEO_ROUTE_NEGATE:
            *out_first = geo_route_negate_state(*first);
            *out_second = geo_route_negate_state(*second);
            break;
        case GEO_ROUTE_KEEP_FIRST:
            *out_first = *first;
            *out_second = zero;
            break;
        case GEO_ROUTE_KEEP_SECOND:
            *out_first = zero;
            *out_second = *second;
            break;
        case GEO_ROUTE_SECOND_TO_FIRST:
            *out_first = *second;
            *out_second = zero;
            break;
        case GEO_ROUTE_FIRST_TO_SECOND:
            *out_first = zero;
            *out_second = *first;
            break;
        case GEO_ROUTE_EXCHANGE:
            *out_first = *second;
            *out_second = *first;
            break;
        default:
            return GEO_STATUS_BAD_OPCODE;
    }

    return GEO_STATUS_OK;
}
