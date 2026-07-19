#include "geo/operator_kernel.h"

#include <limits.h>
#include <string.h>

static int geo_operator_dimension_valid(uint8_t dimension) {
    return dimension >= 1u && dimension <= GEO_OPERATOR_MAX_DIMENSION;
}

static int geo_operator_signature_valid(const int8_t *signature, uint8_t dimension) {
    uint8_t index;
    if (signature == NULL || !geo_operator_dimension_valid(dimension)) return 0;
    for (index = 0u; index < dimension; ++index) {
        if (signature[index] != 1 && signature[index] != -1) return 0;
    }
    return 1;
}

static int geo_operator_same_metric(
    uint8_t dimension,
    const int8_t *left,
    const int8_t *right
) {
    uint8_t index;
    if (left == NULL || right == NULL) return 0;
    for (index = 0u; index < dimension; ++index) {
        if (left[index] != right[index]) return 0;
    }
    return 1;
}

static int geo_operator_mv_f64_matches(
    const geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    return value != NULL &&
        value->dimension == dimension &&
        geo_operator_same_metric(dimension, value->signature, signature);
}

static void geo_operator_mv_f64_zero(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static int32_t geo_operator_mod_normalize(int64_t value, uint32_t modulus) {
    int64_t normalized = value % (int64_t)modulus;
    if (normalized < 0) normalized += (int64_t)modulus;
    return (int32_t)normalized;
}

static int64_t geo_operator_round_divide(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    const uint64_t abs_remainder = remainder < 0 ? (uint64_t)(-remainder) : (uint64_t)remainder;
    const uint64_t abs_denominator = denominator < 0 ? (uint64_t)(-denominator) : (uint64_t)denominator;
    if (abs_remainder >= (abs_denominator + 1u) / 2u) {
        quotient += ((numerator < 0) != (denominator < 0)) ? -1 : 1;
    }
    return quotient;
}

static geo_operator_status_t geo_operator_validate_plan(const geo_operator_plan_i32_t *plan) {
    uint8_t index;
    const size_t blade_count = plan == NULL ? 0u : geo_operator_blade_count(plan->dimension);
    if (plan == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (plan->abi_version != GEO_OPERATOR_ABI_VERSION) return GEO_OPERATOR_BAD_PLAN;
    if (!geo_operator_signature_valid(plan->signature, plan->dimension)) return GEO_OPERATOR_BAD_PLAN;
    if (plan->side != GEO_OPERATOR_SIDE_LEFT && plan->side != GEO_OPERATOR_SIDE_RIGHT) return GEO_OPERATOR_BAD_PLAN;
    if (plan->kind != GEO_OPERATOR_PLAN_FIXED_BLADE &&
        plan->kind != GEO_OPERATOR_PLAN_SPARSE_FIXED_MULTIVECTOR) return GEO_OPERATOR_BAD_PLAN;
    if (plan->term_count == 0u || plan->term_count > GEO_OPERATOR_MAX_TERMS) return GEO_OPERATOR_BAD_PLAN;
    if (plan->kind == GEO_OPERATOR_PLAN_FIXED_BLADE && plan->term_count != 1u) return GEO_OPERATOR_BAD_PLAN;
    for (index = 0u; index < plan->term_count; ++index) {
        if ((size_t)plan->terms[index].blade >= blade_count || plan->terms[index].coefficient == 0) {
            return GEO_OPERATOR_BAD_PLAN;
        }
    }
    return GEO_OPERATOR_OK;
}

static int geo_operator_plan_sign(
    const geo_operator_plan_i32_t *plan,
    uint8_t source,
    uint8_t fixed_blade
) {
    return plan->side == GEO_OPERATOR_SIDE_RIGHT
        ? geo_operator_gp_sign(source, fixed_blade, plan->signature, plan->dimension)
        : geo_operator_gp_sign(fixed_blade, source, plan->signature, plan->dimension);
}

uint32_t geo_operator_abi_version(void) {
    return GEO_OPERATOR_ABI_VERSION;
}

uint32_t geo_operator_gradient_abi_version(void) {
    return GEO_OPERATOR_GRADIENT_ABI_VERSION;
}

size_t geo_operator_blade_count(uint8_t dimension) {
    if (!geo_operator_dimension_valid(dimension)) return 0u;
    return (size_t)1u << dimension;
}

int geo_operator_gp_sign(
    uint8_t left_blade,
    uint8_t right_blade,
    const int8_t *signature,
    uint8_t dimension
) {
    int sign = 1;
    uint8_t index;
    if (!geo_operator_signature_valid(signature, dimension)) return 0;
    for (index = 0u; index < dimension; ++index) {
        const uint8_t bit = (uint8_t)(1u << index);
        if ((left_blade & bit) != 0u) {
            const uint8_t below = (uint8_t)(right_blade & (uint8_t)(bit - 1u));
            uint8_t value = below;
            uint8_t count = 0u;
            while (value != 0u) {
                value = (uint8_t)(value & (uint8_t)(value - 1u));
                ++count;
            }
            if ((count & 1u) != 0u) sign = -sign;
            if ((right_blade & bit) != 0u) sign *= signature[index];
        }
    }
    return sign;
}

geo_operator_status_t geo_operator_plan_fixed_blade_i32(
    geo_operator_plan_i32_t *plan,
    uint8_t dimension,
    const int8_t *signature,
    geo_operator_side_t side,
    uint8_t blade,
    int32_t coefficient
) {
    geo_operator_term_i32_t term;
    term.blade = blade;
    term.coefficient = coefficient;
    return geo_operator_plan_sparse_i32(plan, dimension, signature, side, &term, 1u);
}

geo_operator_status_t geo_operator_plan_sparse_i32(
    geo_operator_plan_i32_t *plan,
    uint8_t dimension,
    const int8_t *signature,
    geo_operator_side_t side,
    const geo_operator_term_i32_t *terms,
    size_t term_count
) {
    size_t index;
    size_t blade_count;
    if (plan == NULL || terms == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (!geo_operator_dimension_valid(dimension)) return GEO_OPERATOR_UNSUPPORTED_DIMENSION;
    if (!geo_operator_signature_valid(signature, dimension)) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (side != GEO_OPERATOR_SIDE_LEFT && side != GEO_OPERATOR_SIDE_RIGHT) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (term_count == 0u || term_count > GEO_OPERATOR_MAX_TERMS) return GEO_OPERATOR_TOO_MANY_TERMS;
    blade_count = geo_operator_blade_count(dimension);
    memset(plan, 0, sizeof(*plan));
    plan->abi_version = GEO_OPERATOR_ABI_VERSION;
    plan->dimension = dimension;
    memcpy(plan->signature, signature, dimension * sizeof(signature[0]));
    plan->kind = term_count == 1u ? GEO_OPERATOR_PLAN_FIXED_BLADE : GEO_OPERATOR_PLAN_SPARSE_FIXED_MULTIVECTOR;
    plan->side = (uint8_t)side;
    plan->term_count = (uint8_t)term_count;
    for (index = 0u; index < term_count; ++index) {
        size_t prior;
        if ((size_t)terms[index].blade >= blade_count || terms[index].coefficient == 0) {
            memset(plan, 0, sizeof(*plan));
            return GEO_OPERATOR_INVALID_ARGUMENT;
        }
        for (prior = 0u; prior < index; ++prior) {
            if (terms[prior].blade == terms[index].blade) {
                memset(plan, 0, sizeof(*plan));
                return GEO_OPERATOR_INVALID_ARGUMENT;
            }
        }
        plan->terms[index] = terms[index];
    }
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_apply_f64(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_f64_t *input,
    geo_operator_mv_f64_t *output
) {
    size_t source;
    size_t term_index;
    size_t blade_count;
    geo_operator_mv_f64_t result;
    geo_operator_status_t status = geo_operator_validate_plan(plan);
    if (status != GEO_OPERATOR_OK) return status;
    if (input == NULL || output == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (!geo_operator_mv_f64_matches(input, plan->dimension, plan->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(plan->dimension);
    geo_operator_mv_f64_zero(&result, plan->dimension, plan->signature);
    for (source = 0u; source < blade_count; ++source) {
        const double source_value = input->coefficients[source];
        if (source_value == 0.0) continue;
        for (term_index = 0u; term_index < plan->term_count; ++term_index) {
            const geo_operator_term_i32_t term = plan->terms[term_index];
            const uint8_t target = (uint8_t)(source ^ term.blade);
            const int sign = geo_operator_plan_sign(plan, (uint8_t)source, term.blade);
            result.coefficients[target] += source_value * (double)(sign * term.coefficient);
        }
    }
    *output = result;
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_apply_parametric_f64(
    const geo_operator_plan_i32_t *plan,
    const double *parameters,
    size_t parameter_count,
    const geo_operator_mv_f64_t *input,
    geo_operator_mv_f64_t *output
) {
    size_t source;
    size_t term_index;
    size_t blade_count;
    geo_operator_mv_f64_t result;
    geo_operator_status_t status = geo_operator_validate_plan(plan);
    if (status != GEO_OPERATOR_OK) return status;
    if (parameters == NULL || input == NULL || output == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (parameter_count != (size_t)plan->term_count) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (!geo_operator_mv_f64_matches(input, plan->dimension, plan->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(plan->dimension);
    geo_operator_mv_f64_zero(&result, plan->dimension, plan->signature);
    for (source = 0u; source < blade_count; ++source) {
        const double source_value = input->coefficients[source];
        if (source_value == 0.0) continue;
        for (term_index = 0u; term_index < plan->term_count; ++term_index) {
            const uint8_t fixed_blade = plan->terms[term_index].blade;
            const uint8_t target = (uint8_t)(source ^ fixed_blade);
            const int sign = geo_operator_plan_sign(plan, (uint8_t)source, fixed_blade);
            result.coefficients[target] += source_value * parameters[term_index] * (double)sign;
        }
    }
    *output = result;
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_apply_f64_vjp(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_f64_t *output_cotangent,
    geo_operator_mv_f64_t *input_cotangent
) {
    size_t source;
    size_t term_index;
    size_t blade_count;
    geo_operator_mv_f64_t result;
    geo_operator_status_t status = geo_operator_validate_plan(plan);
    if (status != GEO_OPERATOR_OK) return status;
    if (output_cotangent == NULL || input_cotangent == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (!geo_operator_mv_f64_matches(output_cotangent, plan->dimension, plan->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(plan->dimension);
    geo_operator_mv_f64_zero(&result, plan->dimension, plan->signature);
    for (source = 0u; source < blade_count; ++source) {
        for (term_index = 0u; term_index < plan->term_count; ++term_index) {
            const geo_operator_term_i32_t term = plan->terms[term_index];
            const uint8_t target = (uint8_t)(source ^ term.blade);
            const int sign = geo_operator_plan_sign(plan, (uint8_t)source, term.blade);
            result.coefficients[source] +=
                output_cotangent->coefficients[target] * (double)(sign * term.coefficient);
        }
    }
    *input_cotangent = result;
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_apply_parametric_f64_vjp(
    const geo_operator_plan_i32_t *plan,
    const double *parameters,
    size_t parameter_count,
    const geo_operator_mv_f64_t *input,
    const geo_operator_mv_f64_t *output_cotangent,
    geo_operator_mv_f64_t *input_cotangent,
    double *parameter_cotangents,
    size_t parameter_cotangent_count
) {
    size_t source;
    size_t term_index;
    size_t blade_count;
    double parameter_result[GEO_OPERATOR_MAX_TERMS] = {0.0};
    geo_operator_mv_f64_t input_result;
    geo_operator_status_t status = geo_operator_validate_plan(plan);
    if (status != GEO_OPERATOR_OK) return status;
    if (parameters == NULL || input == NULL || output_cotangent == NULL ||
        input_cotangent == NULL || parameter_cotangents == NULL) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    if (parameter_count != (size_t)plan->term_count ||
        parameter_cotangent_count < (size_t)plan->term_count) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    if (!geo_operator_mv_f64_matches(input, plan->dimension, plan->signature) ||
        !geo_operator_mv_f64_matches(output_cotangent, plan->dimension, plan->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(plan->dimension);
    geo_operator_mv_f64_zero(&input_result, plan->dimension, plan->signature);
    for (source = 0u; source < blade_count; ++source) {
        for (term_index = 0u; term_index < plan->term_count; ++term_index) {
            const uint8_t fixed_blade = plan->terms[term_index].blade;
            const uint8_t target = (uint8_t)(source ^ fixed_blade);
            const int sign = geo_operator_plan_sign(plan, (uint8_t)source, fixed_blade);
            const double cotangent = output_cotangent->coefficients[target];
            input_result.coefficients[source] += cotangent * parameters[term_index] * (double)sign;
            parameter_result[term_index] += cotangent * input->coefficients[source] * (double)sign;
        }
    }
    *input_cotangent = input_result;
    memcpy(
        parameter_cotangents,
        parameter_result,
        (size_t)plan->term_count * sizeof(parameter_result[0])
    );
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_apply_mod_i32(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_i32_t *input,
    uint32_t modulus,
    geo_operator_mv_i32_t *output
) {
    size_t source;
    size_t term_index;
    size_t blade_count;
    geo_operator_status_t status = geo_operator_validate_plan(plan);
    if (status != GEO_OPERATOR_OK) return status;
    if (input == NULL || output == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (modulus < 3u || modulus > (uint32_t)INT32_MAX) return GEO_OPERATOR_ZERO_MODULUS;
    if (input->dimension != plan->dimension ||
        !geo_operator_same_metric(plan->dimension, input->signature, plan->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(plan->dimension);
    memset(output, 0, sizeof(*output));
    output->dimension = plan->dimension;
    memcpy(output->signature, plan->signature, plan->dimension * sizeof(plan->signature[0]));
    for (source = 0u; source < blade_count; ++source) {
        const int32_t source_value = geo_operator_mod_normalize(input->coefficients[source], modulus);
        if (source_value == 0) continue;
        for (term_index = 0u; term_index < plan->term_count; ++term_index) {
            const geo_operator_term_i32_t term = plan->terms[term_index];
            const uint8_t target = (uint8_t)(source ^ term.blade);
            const int sign = geo_operator_plan_sign(plan, (uint8_t)source, term.blade);
            const int64_t next = (int64_t)output->coefficients[target] +
                (int64_t)source_value * (int64_t)term.coefficient * (int64_t)sign;
            output->coefficients[target] = geo_operator_mod_normalize(next, modulus);
        }
    }
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_apply_q_i32(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_i32_t *input,
    uint8_t fraction_bits,
    geo_operator_mv_i32_t *output
) {
    size_t source;
    size_t term_index;
    size_t blade_count;
    const int64_t scale = fraction_bits < 31u ? (INT64_C(1) << fraction_bits) : 0;
    geo_operator_status_t status = geo_operator_validate_plan(plan);
    if (status != GEO_OPERATOR_OK) return status;
    if (input == NULL || output == NULL || fraction_bits == 0u || fraction_bits > 30u) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    if (input->dimension != plan->dimension ||
        !geo_operator_same_metric(plan->dimension, input->signature, plan->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(plan->dimension);
    memset(output, 0, sizeof(*output));
    output->dimension = plan->dimension;
    memcpy(output->signature, plan->signature, plan->dimension * sizeof(plan->signature[0]));
    for (source = 0u; source < blade_count; ++source) {
        const int32_t source_value = input->coefficients[source];
        if (source_value == 0) continue;
        for (term_index = 0u; term_index < plan->term_count; ++term_index) {
            const geo_operator_term_i32_t term = plan->terms[term_index];
            const uint8_t target = (uint8_t)(source ^ term.blade);
            const int sign = geo_operator_plan_sign(plan, (uint8_t)source, term.blade);
            const int64_t product = (int64_t)source_value * (int64_t)term.coefficient * (int64_t)sign;
            const int64_t rounded = geo_operator_round_divide(product, scale);
            const int64_t next = (int64_t)output->coefficients[target] + rounded;
            if (next < INT32_MIN || next > INT32_MAX) return GEO_OPERATOR_OVERFLOW;
            output->coefficients[target] = (int32_t)next;
        }
    }
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_gp_f64(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    geo_operator_mv_f64_t *output
) {
    size_t left_blade;
    size_t right_blade;
    size_t blade_count;
    geo_operator_mv_f64_t result;
    if (left == NULL || right == NULL || output == NULL) return GEO_OPERATOR_INVALID_ARGUMENT;
    if (!geo_operator_dimension_valid(left->dimension) || left->dimension != right->dimension) {
        return GEO_OPERATOR_UNSUPPORTED_DIMENSION;
    }
    if (!geo_operator_signature_valid(left->signature, left->dimension) ||
        !geo_operator_same_metric(left->dimension, left->signature, right->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(left->dimension);
    geo_operator_mv_f64_zero(&result, left->dimension, left->signature);
    for (left_blade = 0u; left_blade < blade_count; ++left_blade) {
        if (left->coefficients[left_blade] == 0.0) continue;
        for (right_blade = 0u; right_blade < blade_count; ++right_blade) {
            const uint8_t target = (uint8_t)(left_blade ^ right_blade);
            const int sign = geo_operator_gp_sign(
                (uint8_t)left_blade,
                (uint8_t)right_blade,
                left->signature,
                left->dimension
            );
            result.coefficients[target] +=
                left->coefficients[left_blade] * right->coefficients[right_blade] * (double)sign;
        }
    }
    *output = result;
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_gp_f64_jvp(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    const geo_operator_mv_f64_t *left_tangent,
    const geo_operator_mv_f64_t *right_tangent,
    geo_operator_mv_f64_t *output_tangent
) {
    size_t left_blade;
    size_t right_blade;
    size_t blade_count;
    geo_operator_mv_f64_t result;
    if (left == NULL || right == NULL || left_tangent == NULL ||
        right_tangent == NULL || output_tangent == NULL) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    if (!geo_operator_dimension_valid(left->dimension) || left->dimension != right->dimension) {
        return GEO_OPERATOR_UNSUPPORTED_DIMENSION;
    }
    if (!geo_operator_signature_valid(left->signature, left->dimension) ||
        !geo_operator_same_metric(left->dimension, left->signature, right->signature) ||
        !geo_operator_mv_f64_matches(left_tangent, left->dimension, left->signature) ||
        !geo_operator_mv_f64_matches(right_tangent, left->dimension, left->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(left->dimension);
    geo_operator_mv_f64_zero(&result, left->dimension, left->signature);
    for (left_blade = 0u; left_blade < blade_count; ++left_blade) {
        for (right_blade = 0u; right_blade < blade_count; ++right_blade) {
            const uint8_t target = (uint8_t)(left_blade ^ right_blade);
            const int sign = geo_operator_gp_sign(
                (uint8_t)left_blade,
                (uint8_t)right_blade,
                left->signature,
                left->dimension
            );
            result.coefficients[target] += (
                left_tangent->coefficients[left_blade] * right->coefficients[right_blade] +
                left->coefficients[left_blade] * right_tangent->coefficients[right_blade]
            ) * (double)sign;
        }
    }
    *output_tangent = result;
    return GEO_OPERATOR_OK;
}

geo_operator_status_t geo_operator_gp_f64_vjp(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    const geo_operator_mv_f64_t *output_cotangent,
    geo_operator_mv_f64_t *left_cotangent,
    geo_operator_mv_f64_t *right_cotangent
) {
    size_t left_blade;
    size_t right_blade;
    size_t blade_count;
    geo_operator_mv_f64_t left_result;
    geo_operator_mv_f64_t right_result;
    if (left == NULL || right == NULL || output_cotangent == NULL ||
        left_cotangent == NULL || right_cotangent == NULL ||
        left_cotangent == right_cotangent) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    if (!geo_operator_dimension_valid(left->dimension) || left->dimension != right->dimension) {
        return GEO_OPERATOR_UNSUPPORTED_DIMENSION;
    }
    if (!geo_operator_signature_valid(left->signature, left->dimension) ||
        !geo_operator_same_metric(left->dimension, left->signature, right->signature) ||
        !geo_operator_mv_f64_matches(output_cotangent, left->dimension, left->signature)) {
        return GEO_OPERATOR_INVALID_ARGUMENT;
    }
    blade_count = geo_operator_blade_count(left->dimension);
    geo_operator_mv_f64_zero(&left_result, left->dimension, left->signature);
    geo_operator_mv_f64_zero(&right_result, left->dimension, left->signature);
    for (left_blade = 0u; left_blade < blade_count; ++left_blade) {
        for (right_blade = 0u; right_blade < blade_count; ++right_blade) {
            const uint8_t target = (uint8_t)(left_blade ^ right_blade);
            const int sign = geo_operator_gp_sign(
                (uint8_t)left_blade,
                (uint8_t)right_blade,
                left->signature,
                left->dimension
            );
            const double cotangent = output_cotangent->coefficients[target] * (double)sign;
            left_result.coefficients[left_blade] += cotangent * right->coefficients[right_blade];
            right_result.coefficients[right_blade] += cotangent * left->coefficients[left_blade];
        }
    }
    *left_cotangent = left_result;
    *right_cotangent = right_result;
    return GEO_OPERATOR_OK;
}
