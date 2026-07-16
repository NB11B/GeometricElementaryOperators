#include "geo/fixed_omega.h"

#include <limits.h>
#include <stddef.h>

#define GEO_FIXED_LN2_Q30 INT32_C(744261118)
#define GEO_FIXED_EML_TERMS 12

static int geo_fixed_omega_in_range(int64_t value) {
    return value >= INT32_MIN && value <= INT32_MAX;
}

static int64_t geo_fixed_omega_scale(void) {
    return INT64_C(1) << GEO_FIXED_FRACTION_BITS;
}

static uint64_t geo_fixed_omega_abs_i64(int64_t value) {
    if (value >= 0) return (uint64_t)value;
    return (uint64_t)(-(value + INT64_C(1))) + UINT64_C(1);
}

static int64_t geo_fixed_omega_round_divide(
    int64_t numerator,
    int64_t denominator
) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    const uint64_t absolute_remainder = geo_fixed_omega_abs_i64(remainder);
    const uint64_t absolute_denominator = geo_fixed_omega_abs_i64(denominator);

    if (absolute_remainder >=
        (absolute_denominator + UINT64_C(1)) / UINT64_C(2)) {
        quotient += ((numerator < 0) != (denominator < 0)) ? -1 : 1;
    }
    return quotient;
}

static geo_fixed_omega_status_t geo_fixed_omega_checked(
    int64_t value,
    geo_fixed_t *output
) {
    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    if (!geo_fixed_omega_in_range(value)) return GEO_FIXED_OMEGA_OVERFLOW;
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OMEGA_OK;
}

static geo_fixed_omega_status_t geo_fixed_omega_add(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
) {
    return geo_fixed_omega_checked((int64_t)left + (int64_t)right, output);
}

static geo_fixed_omega_status_t geo_fixed_omega_sub(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
) {
    return geo_fixed_omega_checked((int64_t)left - (int64_t)right, output);
}

static geo_fixed_omega_status_t geo_fixed_omega_from_arithmetic(
    geo_fixed_status_t status
) {
    switch (status) {
        case GEO_FIXED_OK:
            return GEO_FIXED_OMEGA_OK;
        case GEO_FIXED_DIVIDE_BY_ZERO:
            return GEO_FIXED_OMEGA_DIVIDE_BY_ZERO;
        case GEO_FIXED_OVERFLOW:
        default:
            return GEO_FIXED_OMEGA_OVERFLOW;
    }
}

static geo_fixed_t geo_fixed_omega_constant_q30(int32_t value) {
#if GEO_FIXED_FRACTION_BITS == 30
    return (geo_fixed_t)value;
#else
    const int shift = 30 - GEO_FIXED_FRACTION_BITS;
    const int64_t rounded = geo_fixed_omega_round_divide(
        (int64_t)value,
        INT64_C(1) << shift
    );
    return (geo_fixed_t)rounded;
#endif
}

static geo_fixed_omega_status_t geo_fixed_omega_divide_integer(
    geo_fixed_t value,
    int32_t divisor,
    geo_fixed_t *output
) {
    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    if (divisor == 0) return GEO_FIXED_OMEGA_DIVIDE_BY_ZERO;
    return geo_fixed_omega_checked(
        geo_fixed_omega_round_divide((int64_t)value, (int64_t)divisor),
        output
    );
}

static geo_fixed_omega_status_t geo_fixed_omega_multiply_integer(
    geo_fixed_t value,
    int32_t multiplier,
    geo_fixed_t *output
) {
    return geo_fixed_omega_checked(
        (int64_t)value * (int64_t)multiplier,
        output
    );
}

static geo_fixed_omega_status_t geo_fixed_omega_round_integer(
    geo_fixed_t value,
    int32_t *output
) {
    const int64_t rounded = geo_fixed_omega_round_divide(
        (int64_t)value,
        geo_fixed_omega_scale()
    );
    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    if (rounded < INT32_MIN || rounded > INT32_MAX) {
        return GEO_FIXED_OMEGA_OVERFLOW;
    }
    *output = (int32_t)rounded;
    return GEO_FIXED_OMEGA_OK;
}

static geo_fixed_omega_status_t geo_fixed_omega_scale_power_two(
    geo_fixed_t value,
    int32_t exponent,
    geo_fixed_t *output
) {
    int64_t result;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    if (value == 0) {
        *output = 0;
        return GEO_FIXED_OMEGA_OK;
    }

    if (exponent >= 0) {
        if (exponent > 31) return GEO_FIXED_OMEGA_OVERFLOW;
        result = (int64_t)value * (INT64_C(1) << exponent);
        return geo_fixed_omega_checked(result, output);
    }

    if (exponent <= -63) {
        *output = 0;
        return GEO_FIXED_OMEGA_OK;
    }
    result = geo_fixed_omega_round_divide(
        (int64_t)value,
        INT64_C(1) << (-exponent)
    );
    return geo_fixed_omega_checked(result, output);
}

static uint64_t geo_fixed_omega_gcd_u64(uint64_t left, uint64_t right) {
    while (right != UINT64_C(0)) {
        const uint64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left == UINT64_C(0) ? UINT64_C(1) : left;
}

static geo_fixed_omega_status_t geo_fixed_scale_multiply(
    geo_fixed_scale_t left,
    geo_fixed_scale_t right,
    geo_fixed_scale_t *output
) {
    int64_t numerator;
    int64_t denominator;
    uint64_t divisor;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    if (left.denominator == 0 || right.denominator == 0) {
        return GEO_FIXED_OMEGA_INVALID_SCALE;
    }

    numerator = (int64_t)left.numerator * (int64_t)right.numerator;
    denominator = (int64_t)left.denominator * (int64_t)right.denominator;
    if (denominator == 0) return GEO_FIXED_OMEGA_INVALID_SCALE;
    if (denominator < 0) {
        numerator = -numerator;
        denominator = -denominator;
    }

    divisor = geo_fixed_omega_gcd_u64(
        geo_fixed_omega_abs_i64(numerator),
        (uint64_t)denominator
    );
    numerator /= (int64_t)divisor;
    denominator /= (int64_t)divisor;

    if (numerator < INT32_MIN || numerator > INT32_MAX ||
        denominator < 1 || denominator > INT32_MAX) {
        return GEO_FIXED_OMEGA_OVERFLOW;
    }

    output->numerator = (int32_t)numerator;
    output->denominator = (int32_t)denominator;
    return GEO_FIXED_OMEGA_OK;
}

geo_fixed_omega_status_t geo_fixed_eml_exp(
    geo_fixed_t input,
    geo_fixed_t *output
) {
    const geo_fixed_t one = (geo_fixed_t)geo_fixed_omega_scale();
    const geo_fixed_t ln2 = geo_fixed_omega_constant_q30(GEO_FIXED_LN2_Q30);
    geo_fixed_t scaled;
    geo_fixed_t multiple;
    geo_fixed_t remainder;
    geo_fixed_t sum = one;
    geo_fixed_t term = one;
    int32_t exponent;
    int32_t order;
    geo_fixed_status_t arithmetic_status;
    geo_fixed_omega_status_t status;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    arithmetic_status = geo_fixed_div(input, ln2, &scaled);
    if (arithmetic_status != GEO_FIXED_OK) {
        return geo_fixed_omega_from_arithmetic(arithmetic_status);
    }
    status = geo_fixed_omega_round_integer(scaled, &exponent);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    status = geo_fixed_omega_multiply_integer(ln2, exponent, &multiple);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    status = geo_fixed_omega_sub(input, multiple, &remainder);
    if (status != GEO_FIXED_OMEGA_OK) return status;

    for (order = 1; order <= GEO_FIXED_EML_TERMS; ++order) {
        arithmetic_status = geo_fixed_mul(term, remainder, &term);
        if (arithmetic_status != GEO_FIXED_OK) {
            return geo_fixed_omega_from_arithmetic(arithmetic_status);
        }
        status = geo_fixed_omega_divide_integer(term, order, &term);
        if (status != GEO_FIXED_OMEGA_OK) return status;
        status = geo_fixed_omega_add(sum, term, &sum);
        if (status != GEO_FIXED_OMEGA_OK) return status;
    }

    return geo_fixed_omega_scale_power_two(sum, exponent, output);
}

geo_fixed_omega_status_t geo_fixed_eml_log(
    geo_fixed_t input,
    geo_fixed_t *output
) {
    const geo_fixed_t one = (geo_fixed_t)geo_fixed_omega_scale();
    const geo_fixed_t ln2 = geo_fixed_omega_constant_q30(GEO_FIXED_LN2_Q30);
    uint32_t raw;
    int32_t most_significant_bit = -1;
    int32_t exponent;
    int32_t order;
    geo_fixed_t mantissa;
    geo_fixed_t numerator;
    geo_fixed_t z;
    geo_fixed_t z_squared;
    geo_fixed_t power;
    geo_fixed_t sum = 0;
    geo_fixed_t term;
    geo_fixed_t logarithm;
    geo_fixed_t exponent_term;
    int64_t denominator_wide;
    int64_t z_wide;
    geo_fixed_status_t arithmetic_status;
    geo_fixed_omega_status_t status;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    if (input <= 0) return GEO_FIXED_OMEGA_LOG_DOMAIN;

    raw = (uint32_t)input;
    while (raw != 0u) {
        ++most_significant_bit;
        raw >>= 1u;
    }
    exponent = most_significant_bit - GEO_FIXED_FRACTION_BITS + 1;

    if (exponent >= 0) {
        mantissa = (geo_fixed_t)geo_fixed_omega_round_divide(
            (int64_t)input,
            INT64_C(1) << exponent
        );
    } else {
        const int shift = -exponent;
        const int64_t shifted = (int64_t)input * (INT64_C(1) << shift);
        if (!geo_fixed_omega_in_range(shifted)) return GEO_FIXED_OMEGA_OVERFLOW;
        mantissa = (geo_fixed_t)shifted;
    }

    status = geo_fixed_omega_sub(mantissa, one, &numerator);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    denominator_wide = (int64_t)mantissa + (int64_t)one;
    if (denominator_wide <= 0) return GEO_FIXED_OMEGA_LOG_DOMAIN;
    z_wide = geo_fixed_omega_round_divide(
        (int64_t)numerator * geo_fixed_omega_scale(),
        denominator_wide
    );
    status = geo_fixed_omega_checked(z_wide, &z);
    if (status != GEO_FIXED_OMEGA_OK) return status;

    arithmetic_status = geo_fixed_mul(z, z, &z_squared);
    if (arithmetic_status != GEO_FIXED_OK) {
        return geo_fixed_omega_from_arithmetic(arithmetic_status);
    }

    power = z;
    for (order = 0; order < GEO_FIXED_EML_TERMS; ++order) {
        status = geo_fixed_omega_divide_integer(power, 2 * order + 1, &term);
        if (status != GEO_FIXED_OMEGA_OK) return status;
        status = geo_fixed_omega_add(sum, term, &sum);
        if (status != GEO_FIXED_OMEGA_OK) return status;
        if (order + 1 < GEO_FIXED_EML_TERMS) {
            arithmetic_status = geo_fixed_mul(power, z_squared, &power);
            if (arithmetic_status != GEO_FIXED_OK) {
                return geo_fixed_omega_from_arithmetic(arithmetic_status);
            }
        }
    }

    status = geo_fixed_omega_multiply_integer(sum, 2, &logarithm);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    status = geo_fixed_omega_multiply_integer(ln2, exponent, &exponent_term);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    return geo_fixed_omega_add(logarithm, exponent_term, output);
}

geo_fixed_omega_status_t geo_fixed_eml_apply(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
) {
    geo_fixed_t exponential;
    geo_fixed_t logarithm;
    geo_fixed_omega_status_t status;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    status = geo_fixed_eml_exp(left, &exponential);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    status = geo_fixed_eml_log(right, &logarithm);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    return geo_fixed_omega_sub(exponential, logarithm, output);
}

geo_fixed_omega_status_t geo_fixed_opposite_from_cl20(
    geo_fixed_cl20_t value,
    geo_fixed_opposite_t *output
) {
    geo_fixed_opposite_t result;
    geo_fixed_status_t status;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    result.forward = value;
    status = geo_fixed_cl20_reverse_checked(value, &result.reverse);
    if (status != GEO_FIXED_OK) return geo_fixed_omega_from_arithmetic(status);
    *output = result;
    return GEO_FIXED_OMEGA_OK;
}

geo_fixed_omega_status_t geo_fixed_opposite_mul(
    geo_fixed_opposite_t left,
    geo_fixed_opposite_t right,
    geo_fixed_opposite_t *output
) {
    geo_fixed_opposite_t temporary;
    geo_fixed_status_t status;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    status = geo_fixed_cl20_mul(left.forward, right.forward, &temporary.forward);
    if (status != GEO_FIXED_OK) return geo_fixed_omega_from_arithmetic(status);
    status = geo_fixed_cl20_mul(right.reverse, left.reverse, &temporary.reverse);
    if (status != GEO_FIXED_OK) return geo_fixed_omega_from_arithmetic(status);
    *output = temporary;
    return GEO_FIXED_OMEGA_OK;
}

geo_fixed_state_t geo_fixed_state_zero(void) {
    const geo_fixed_cl20_t zero = {0, 0, 0, 0};
    geo_fixed_state_t state;
    state.scalar = 0;
    state.geometric.forward = zero;
    state.geometric.reverse = zero;
    state.scale.numerator = 1;
    state.scale.denominator = 1;
    state.active_lanes = GEO_FIXED_LANE_NONE;
    return state;
}

geo_fixed_state_t geo_fixed_state_from_scalar(geo_fixed_t scalar) {
    geo_fixed_state_t state = geo_fixed_state_zero();
    state.scalar = scalar;
    state.active_lanes = GEO_FIXED_LANE_SCALAR;
    return state;
}

geo_fixed_omega_status_t geo_fixed_state_from_cl20(
    geo_fixed_cl20_t value,
    geo_fixed_state_t *output
) {
    geo_fixed_state_t state;
    geo_fixed_omega_status_t status;

    if (output == NULL) return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    state = geo_fixed_state_zero();
    status = geo_fixed_opposite_from_cl20(value, &state.geometric);
    if (status != GEO_FIXED_OMEGA_OK) return status;
    state.active_lanes = GEO_FIXED_LANE_GEOMETRIC;
    *output = state;
    return GEO_FIXED_OMEGA_OK;
}

geo_fixed_omega_status_t geo_fixed_omega_apply(
    const geo_fixed_state_t *left,
    const geo_fixed_state_t *right,
    uint8_t requested_lanes,
    geo_fixed_state_t *output
) {
    geo_fixed_state_t result;
    geo_fixed_omega_status_t status;

    if (left == NULL || right == NULL || output == NULL) {
        return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    }
    if (requested_lanes == GEO_FIXED_LANE_NONE ||
        (requested_lanes & (uint8_t)~GEO_FIXED_LANE_ALL) != 0u) {
        return GEO_FIXED_OMEGA_BAD_LANES;
    }
    if ((left->active_lanes & requested_lanes) != requested_lanes ||
        (right->active_lanes & requested_lanes) != requested_lanes) {
        return GEO_FIXED_OMEGA_BAD_LANES;
    }

    result = geo_fixed_state_zero();
    if ((requested_lanes & GEO_FIXED_LANE_SCALAR) != 0u) {
        status = geo_fixed_eml_apply(left->scalar, right->scalar, &result.scalar);
        if (status != GEO_FIXED_OMEGA_OK) return status;
    }
    if ((requested_lanes & GEO_FIXED_LANE_GEOMETRIC) != 0u) {
        status = geo_fixed_opposite_mul(
            left->geometric,
            right->geometric,
            &result.geometric
        );
        if (status != GEO_FIXED_OMEGA_OK) return status;
        status = geo_fixed_scale_multiply(left->scale, right->scale, &result.scale);
        if (status != GEO_FIXED_OMEGA_OK) return status;
    }
    result.active_lanes = requested_lanes;
    *output = result;
    return GEO_FIXED_OMEGA_OK;
}

geo_fixed_omega_status_t geo_fixed_omega_program_execute(
    const geo_fixed_omega_program_t *program,
    geo_fixed_state_t *registers,
    size_t register_capacity
) {
    size_t pc;

    if (program == NULL || registers == NULL) {
        return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    }
    if (program->instruction_count != 0u && program->instructions == NULL) {
        return GEO_FIXED_OMEGA_NULL_ARGUMENT;
    }
    if (program->register_count == 0u ||
        program->register_count > register_capacity) {
        return GEO_FIXED_OMEGA_REGISTER_RANGE;
    }

    for (pc = 0u; pc < program->instruction_count; ++pc) {
        const geo_fixed_omega_instruction_t instruction = program->instructions[pc];
        geo_fixed_state_t result;
        geo_fixed_omega_status_t status;

        if ((size_t)instruction.destination >= program->register_count ||
            (size_t)instruction.left >= program->register_count) {
            return GEO_FIXED_OMEGA_REGISTER_RANGE;
        }
        if (instruction.requested_lanes == GEO_FIXED_LANE_NONE ||
            (instruction.requested_lanes & (uint8_t)~GEO_FIXED_LANE_ALL) != 0u) {
            return GEO_FIXED_OMEGA_BAD_LANES;
        }

        switch ((geo_fixed_omega_opcode_t)instruction.opcode) {
            case GEO_FIXED_OMEGA_OPCODE_COPY:
                if ((registers[instruction.left].active_lanes &
                        instruction.requested_lanes) != instruction.requested_lanes) {
                    return GEO_FIXED_OMEGA_BAD_LANES;
                }
                result = geo_fixed_state_zero();
                if ((instruction.requested_lanes & GEO_FIXED_LANE_SCALAR) != 0u) {
                    result.scalar = registers[instruction.left].scalar;
                }
                if ((instruction.requested_lanes & GEO_FIXED_LANE_GEOMETRIC) != 0u) {
                    result.geometric = registers[instruction.left].geometric;
                    result.scale = registers[instruction.left].scale;
                }
                result.active_lanes = instruction.requested_lanes;
                break;

            case GEO_FIXED_OMEGA_OPCODE_APPLY:
                if ((size_t)instruction.right >= program->register_count) {
                    return GEO_FIXED_OMEGA_REGISTER_RANGE;
                }
                status = geo_fixed_omega_apply(
                    &registers[instruction.left],
                    &registers[instruction.right],
                    instruction.requested_lanes,
                    &result
                );
                if (status != GEO_FIXED_OMEGA_OK) return status;
                break;

            default:
                return GEO_FIXED_OMEGA_BAD_OPCODE;
        }

        registers[instruction.destination] = result;
    }

    return GEO_FIXED_OMEGA_OK;
}
