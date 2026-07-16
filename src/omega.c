#include "geo/omega.h"

#include <limits.h>
#include <math.h>

static geo_real_t geo_exp_value(geo_real_t value) {
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    return exp(value);
#else
    return expf(value);
#endif
}

static geo_real_t geo_log_value(geo_real_t value) {
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    return log(value);
#else
    return logf(value);
#endif
}

static int64_t geo_abs_i64(int64_t value) {
    return value < 0 ? -value : value;
}

static int64_t geo_gcd_i64(int64_t a, int64_t b) {
    a = geo_abs_i64(a);
    b = geo_abs_i64(b);

    while (b != 0) {
        const int64_t remainder = a % b;
        a = b;
        b = remainder;
    }

    return a == 0 ? 1 : a;
}

static geo_status_t geo_scale_mul(
    geo_scale_t a,
    geo_scale_t b,
    geo_scale_t *output
) {
    int64_t numerator;
    int64_t denominator;
    int64_t divisor;

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (a.denominator == 0 || b.denominator == 0) {
        return GEO_STATUS_SCALE_OVERFLOW;
    }

    numerator = (int64_t)a.numerator * (int64_t)b.numerator;
    denominator = (int64_t)a.denominator * (int64_t)b.denominator;

    if (denominator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }

    divisor = geo_gcd_i64(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;

    if (
        numerator < INT32_MIN || numerator > INT32_MAX ||
        denominator < 1 || denominator > INT32_MAX
    ) {
        return GEO_STATUS_SCALE_OVERFLOW;
    }

    output->numerator = (int32_t)numerator;
    output->denominator = (int32_t)denominator;
    return GEO_STATUS_OK;
}

geo_scale_t geo_scale_one(void) {
    geo_scale_t result;
    result.numerator = 1;
    result.denominator = 1;
    return result;
}

geo_state_t geo_state_zero(void) {
    geo_state_t result;
    result.scalar = (geo_real_t)0;
    result.geometric = geo_opposite_zero();
    result.scale = geo_scale_one();
    result.active_lanes = GEO_LANE_NONE;
    return result;
}

geo_state_t geo_state_from_scalar(geo_real_t scalar) {
    geo_state_t result = geo_state_zero();
    result.scalar = scalar;
    result.active_lanes = GEO_LANE_SCALAR;
    return result;
}

geo_state_t geo_state_from_cl20(geo_cl20_t value) {
    geo_state_t result = geo_state_zero();
    result.geometric = geo_opposite_from_cl20(value);
    result.active_lanes = GEO_LANE_GEOMETRIC;
    return result;
}

geo_status_t geo_omega_apply(
    const geo_state_t *left,
    const geo_state_t *right,
    uint8_t requested_lanes,
    geo_state_t *output
) {
    geo_state_t result;
    geo_status_t status;
    const uint8_t lanes = (uint8_t)(requested_lanes & GEO_LANE_ALL);

    if (left == NULL || right == NULL || output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    result = geo_state_zero();

    if ((lanes & GEO_LANE_SCALAR) != 0u) {
        if ((left->active_lanes & GEO_LANE_SCALAR) == 0u ||
            (right->active_lanes & GEO_LANE_SCALAR) == 0u ||
            right->scalar <= (geo_real_t)0) {
            return GEO_STATUS_LOG_DOMAIN;
        }

        result.scalar = geo_exp_value(left->scalar) - geo_log_value(right->scalar);
        result.active_lanes = (uint8_t)(result.active_lanes | GEO_LANE_SCALAR);
    }

    if ((lanes & GEO_LANE_GEOMETRIC) != 0u) {
        if ((left->active_lanes & GEO_LANE_GEOMETRIC) == 0u ||
            (right->active_lanes & GEO_LANE_GEOMETRIC) == 0u) {
            return GEO_STATUS_NULL_ARGUMENT;
        }

        result.geometric = geo_opposite_mul(left->geometric, right->geometric);
        status = geo_scale_mul(left->scale, right->scale, &result.scale);
        if (status != GEO_STATUS_OK) {
            return status;
        }
        result.active_lanes = (uint8_t)(result.active_lanes | GEO_LANE_GEOMETRIC);
    }

    *output = result;
    return GEO_STATUS_OK;
}

geo_status_t geo_program_execute(
    const geo_program_t *program,
    geo_state_t *registers,
    size_t register_capacity
) {
    size_t pc;

    if (program == NULL || registers == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (program->register_count > register_capacity) {
        return GEO_STATUS_REGISTER_RANGE;
    }

    for (pc = 0; pc < program->instruction_count; ++pc) {
        const geo_instruction_t instruction = program->instructions[pc];

        if (instruction.destination >= program->register_count ||
            instruction.left >= program->register_count) {
            return GEO_STATUS_REGISTER_RANGE;
        }

        switch ((geo_opcode_t)instruction.opcode) {
            case GEO_OPCODE_COPY:
                registers[instruction.destination] = registers[instruction.left];
                break;

            case GEO_OPCODE_OMEGA:
                if (instruction.right >= program->register_count) {
                    return GEO_STATUS_REGISTER_RANGE;
                }
                {
                    const geo_state_t left_value = registers[instruction.left];
                    const geo_state_t right_value = registers[instruction.right];
                    geo_status_t status = geo_omega_apply(
                        &left_value,
                        &right_value,
                        instruction.requested_lanes,
                        &registers[instruction.destination]
                    );
                    if (status != GEO_STATUS_OK) {
                        return status;
                    }
                }
                break;

            default:
                return GEO_STATUS_BAD_OPCODE;
        }
    }

    return GEO_STATUS_OK;
}
