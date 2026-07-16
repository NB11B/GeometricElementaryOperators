#include "geo/fixed_geb36.h"

#include <limits.h>
#include <stddef.h>

static geo_fixed_t geo_fixed_one_value(void) {
    return (geo_fixed_t)(INT64_C(1) << GEO_FIXED_FRACTION_BITS);
}

static geo_fixed_cl20_t geo_fixed_zero_mv(void) {
    const geo_fixed_cl20_t value = {0, 0, 0, 0};
    return value;
}

static geo_fixed_cl20_t geo_fixed_one_mv(void) {
    geo_fixed_cl20_t value = geo_fixed_zero_mv();
    value.scalar = geo_fixed_one_value();
    return value;
}

static geo_fixed_status_t geo_fixed_checked_i64(int64_t value, geo_fixed_t *output) {
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) {
        return GEO_FIXED_OVERFLOW;
    }
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t geo_fixed_mv_add(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
) {
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_checked_i64((int64_t)a.scalar + b.scalar, &output->scalar);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_checked_i64((int64_t)a.e1 + b.e1, &output->e1);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_checked_i64((int64_t)a.e2 + b.e2, &output->e2);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_checked_i64((int64_t)a.e12 + b.e12, &output->e12);
}

static geo_fixed_status_t geo_fixed_mv_sub(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
) {
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_checked_i64((int64_t)a.scalar - b.scalar, &output->scalar);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_checked_i64((int64_t)a.e1 - b.e1, &output->e1);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_checked_i64((int64_t)a.e2 - b.e2, &output->e2);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_checked_i64((int64_t)a.e12 - b.e12, &output->e12);
}

static geo_fixed_status_t geo_fixed_mv_neg(
    geo_fixed_cl20_t input,
    geo_fixed_cl20_t *output
) {
    if (output == NULL || input.scalar == INT32_MIN || input.e1 == INT32_MIN ||
        input.e2 == INT32_MIN || input.e12 == INT32_MIN) {
        return GEO_FIXED_OVERFLOW;
    }
    output->scalar = -input.scalar;
    output->e1 = -input.e1;
    output->e2 = -input.e2;
    output->e12 = -input.e12;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t geo_fixed_mv_scale(
    geo_fixed_cl20_t input,
    geo_fixed_t scale,
    geo_fixed_cl20_t *output
) {
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_mul(input.scalar, scale, &output->scalar);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mul(input.e1, scale, &output->e1);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mul(input.e2, scale, &output->e2);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_mul(input.e12, scale, &output->e12);
}

static geo_fixed_cl20_t geo_fixed_grade_involution(geo_fixed_cl20_t value) {
    if (value.e1 == INT32_MIN) value.e1 = INT32_MAX; else value.e1 = -value.e1;
    if (value.e2 == INT32_MIN) value.e2 = INT32_MAX; else value.e2 = -value.e2;
    return value;
}

static geo_fixed_cl20_t geo_fixed_clifford_conjugate(geo_fixed_cl20_t value) {
    value = geo_fixed_grade_involution(value);
    return geo_fixed_cl20_reverse(value);
}

static geo_fixed_cl20_t geo_fixed_project(geo_fixed_cl20_t value, uint8_t mask) {
    if ((mask & GEO_GRADE_SCALAR) == 0u) value.scalar = 0;
    if ((mask & GEO_GRADE_VECTOR) == 0u) {
        value.e1 = 0;
        value.e2 = 0;
    }
    if ((mask & GEO_GRADE_BIVECTOR) == 0u) value.e12 = 0;
    return value;
}

static geo_fixed_status_t geo_fixed_norm_squared(
    geo_fixed_cl20_t value,
    geo_fixed_t *output
) {
    return geo_fixed_vector_dot(value, value, output);
}

static geo_fixed_status_t geo_fixed_dot_mv(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
) {
    geo_fixed_t dot;
    geo_fixed_status_t status = geo_fixed_vector_dot(a, b, &dot);
    if (status != GEO_FIXED_OK || output == NULL) return status;
    *output = geo_fixed_zero_mv();
    output->scalar = dot;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t geo_fixed_wedge_mv(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
) {
    geo_fixed_t wedge;
    geo_fixed_status_t status = geo_fixed_vector_wedge(a, b, &wedge);
    if (status != GEO_FIXED_OK || output == NULL) return status;
    *output = geo_fixed_zero_mv();
    output->e12 = wedge;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t geo_fixed_half_sum_difference(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    int difference,
    geo_fixed_cl20_t *output
) {
    geo_fixed_cl20_t ab;
    geo_fixed_cl20_t ba;
    geo_fixed_cl20_t combined;
    geo_fixed_t half;
    geo_fixed_status_t status;
    status = geo_fixed_cl20_mul(a, b, &ab);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_cl20_mul(b, a, &ba);
    if (status != GEO_FIXED_OK) return status;
    status = difference ? geo_fixed_mv_sub(ab, ba, &combined) : geo_fixed_mv_add(ab, ba, &combined);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_div(geo_fixed_one_value(), (geo_fixed_t)(geo_fixed_one_value() * 2), &half);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_mv_scale(combined, half, output);
}

static geo_fixed_status_t geo_fixed_projection_numerator(
    geo_fixed_cl20_t vector,
    geo_fixed_cl20_t onto,
    geo_fixed_cl20_t *output
) {
    geo_fixed_t dot;
    geo_fixed_status_t status = geo_fixed_vector_dot(vector, onto, &dot);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_mv_scale(onto, dot, output);
}

static geo_fixed_status_t geo_fixed_rejection_numerator(
    geo_fixed_cl20_t vector,
    geo_fixed_cl20_t onto,
    geo_fixed_cl20_t *output
) {
    geo_fixed_t norm;
    geo_fixed_cl20_t scaled_vector;
    geo_fixed_cl20_t projection;
    geo_fixed_status_t status = geo_fixed_norm_squared(onto, &norm);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mv_scale(vector, norm, &scaled_vector);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_projection_numerator(vector, onto, &projection);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_mv_sub(scaled_vector, projection, output);
}

static geo_fixed_status_t geo_fixed_reflection_numerator(
    geo_fixed_cl20_t vector,
    geo_fixed_cl20_t normal,
    geo_fixed_cl20_t *output
) {
    geo_fixed_t norm;
    geo_fixed_t two_dot;
    geo_fixed_cl20_t scaled_vector;
    geo_fixed_cl20_t scaled_normal;
    geo_fixed_status_t status = geo_fixed_norm_squared(normal, &norm);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mv_scale(vector, norm, &scaled_vector);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_vector_dot(vector, normal, &two_dot);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_checked_i64((int64_t)two_dot * 2, &two_dot);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mv_scale(normal, two_dot, &scaled_normal);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_mv_sub(scaled_vector, scaled_normal, output);
}

static geo_fixed_status_t geo_fixed_sandwich(
    geo_fixed_cl20_t transform,
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
) {
    return geo_fixed_rotor_action(transform, value, output);
}

static void geo_fixed_set_cl20(geo_fixed_geb_result_t *output, geo_fixed_cl20_t value) {
    output->kind = (uint8_t)GEO_FIXED_RESULT_CL20;
    output->as.cl20 = value;
}

static void geo_fixed_set_scalar(geo_fixed_geb_result_t *output, geo_fixed_t value) {
    output->kind = (uint8_t)GEO_FIXED_RESULT_SCALAR;
    output->as.scalar = value;
}

static void geo_fixed_set_projective(
    geo_fixed_geb_result_t *output,
    geo_fixed_cl20_t represented,
    geo_fixed_t denominator
) {
    output->kind = (uint8_t)GEO_FIXED_RESULT_PROJECTIVE;
    output->as.projective.represented = represented;
    output->as.projective.denominator = denominator;
}

geo_fixed_status_t geo_fixed_geb36_execute(
    uint8_t target_id,
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t transform,
    geo_fixed_geb_result_t *output
) {
    geo_fixed_cl20_t result;
    geo_fixed_t scalar;
    geo_fixed_status_t status = GEO_FIXED_OK;
    if (output == NULL) return GEO_FIXED_OVERFLOW;

    switch ((geo_geb_target_id_t)target_id) {
        case GEO_GEB_ZERO:
            geo_fixed_set_cl20(output, geo_fixed_zero_mv());
            return GEO_FIXED_OK;
        case GEO_GEB_ONE:
            geo_fixed_set_cl20(output, geo_fixed_one_mv());
            return GEO_FIXED_OK;
        case GEO_GEB_MINUS_ONE:
            result = geo_fixed_one_mv();
            result.scalar = -result.scalar;
            geo_fixed_set_cl20(output, result);
            return GEO_FIXED_OK;
        case GEO_GEB_E1:
            result = geo_fixed_zero_mv(); result.e1 = geo_fixed_one_value();
            geo_fixed_set_cl20(output, result); return GEO_FIXED_OK;
        case GEO_GEB_E2:
            result = geo_fixed_zero_mv(); result.e2 = geo_fixed_one_value();
            geo_fixed_set_cl20(output, result); return GEO_FIXED_OK;
        case GEO_GEB_PSEUDOSCALAR:
            result = geo_fixed_zero_mv(); result.e12 = geo_fixed_one_value();
            geo_fixed_set_cl20(output, result); return GEO_FIXED_OK;
        case GEO_GEB_NEGATION:
            status = geo_fixed_mv_neg(a, &result); break;
        case GEO_GEB_REVERSION:
            result = geo_fixed_cl20_reverse(a); break;
        case GEO_GEB_GRADE_INVOLUTION:
            result = geo_fixed_grade_involution(a); break;
        case GEO_GEB_CLIFFORD_CONJUGATION:
            result = geo_fixed_clifford_conjugate(a); break;
        case GEO_GEB_SCALAR_PROJECTION:
            result = geo_fixed_project(a, GEO_GRADE_SCALAR); break;
        case GEO_GEB_VECTOR_PROJECTION:
            result = geo_fixed_project(a, GEO_GRADE_VECTOR); break;
        case GEO_GEB_BIVECTOR_PROJECTION:
            result = geo_fixed_project(a, GEO_GRADE_BIVECTOR); break;
        case GEO_GEB_ADDITION:
            status = geo_fixed_mv_add(a, b, &result); break;
        case GEO_GEB_SUBTRACTION:
            status = geo_fixed_mv_sub(a, b, &result); break;
        case GEO_GEB_GEOMETRIC_PRODUCT:
            status = geo_fixed_cl20_mul(a, b, &result); break;
        case GEO_GEB_REVERSE_PRODUCT:
            status = geo_fixed_cl20_mul(a, b, &result);
            if (status == GEO_FIXED_OK) result = geo_fixed_cl20_reverse(result);
            break;
        case GEO_GEB_VECTOR_DOT:
            status = geo_fixed_vector_dot(a, b, &scalar);
            if (status == GEO_FIXED_OK) { geo_fixed_set_scalar(output, scalar); return GEO_FIXED_OK; }
            return status;
        case GEO_GEB_VECTOR_WEDGE:
            status = geo_fixed_wedge_mv(a, b, &result); break;
        case GEO_GEB_COMMUTATOR:
            status = geo_fixed_half_sum_difference(a, b, 1, &result); break;
        case GEO_GEB_ANTICOMMUTATOR:
            status = geo_fixed_half_sum_difference(a, b, 0, &result); break;
        case GEO_GEB_VECTOR_NORM_SQUARED:
            status = geo_fixed_norm_squared(a, &scalar);
            if (status == GEO_FIXED_OK) { geo_fixed_set_scalar(output, scalar); return GEO_FIXED_OK; }
            return status;
        case GEO_GEB_DISTANCE_SQUARED:
            status = geo_fixed_mv_sub(a, b, &result);
            if (status == GEO_FIXED_OK) status = geo_fixed_norm_squared(result, &scalar);
            if (status == GEO_FIXED_OK) { geo_fixed_set_scalar(output, scalar); return GEO_FIXED_OK; }
            return status;
        case GEO_GEB_PROJECTION_NUMERATOR:
            status = geo_fixed_projection_numerator(a, b, &result);
            if (status == GEO_FIXED_OK) geo_fixed_set_projective(output, result, geo_fixed_one_value());
            return status;
        case GEO_GEB_REJECTION_NUMERATOR:
            status = geo_fixed_rejection_numerator(a, b, &result);
            if (status == GEO_FIXED_OK) geo_fixed_set_projective(output, result, geo_fixed_one_value());
            return status;
        case GEO_GEB_REFLECTION_NUMERATOR:
            status = geo_fixed_reflection_numerator(a, b, &result);
            if (status == GEO_FIXED_OK) geo_fixed_set_projective(output, result, geo_fixed_one_value());
            return status;
        case GEO_GEB_DUAL: {
            geo_fixed_cl20_t inverse_i = geo_fixed_zero_mv();
            inverse_i.e12 = -geo_fixed_one_value();
            status = geo_fixed_cl20_mul(a, inverse_i, &result);
            break;
        }
        case GEO_GEB_EVEN_PROJECTION:
            result = geo_fixed_project(a, (uint8_t)(GEO_GRADE_SCALAR | GEO_GRADE_BIVECTOR)); break;
        case GEO_GEB_ODD_PROJECTION:
            result = geo_fixed_project(a, GEO_GRADE_VECTOR); break;
        case GEO_GEB_ROTOR_ACTION:
            status = geo_fixed_sandwich(transform, a, &result); break;
        case GEO_GEB_ROTOR_COMPOSITION:
            status = geo_fixed_cl20_mul(a, b, &result); break;
        case GEO_GEB_ROTOR_NORM_SQUARED: {
            geo_fixed_cl20_t reverse = geo_fixed_cl20_reverse(a);
            status = geo_fixed_cl20_mul(a, reverse, &result);
            if (status == GEO_FIXED_OK) { geo_fixed_set_scalar(output, result.scalar); return GEO_FIXED_OK; }
            return status;
        }
        case GEO_GEB_DILATION:
            status = geo_fixed_sandwich(transform, a, &result); break;
        case GEO_GEB_TRANSLATION_UNIPOTENT:
            output->kind = (uint8_t)GEO_FIXED_RESULT_UNIPOTENT;
            output->as.unipotent_payload = a;
            return GEO_FIXED_OK;
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE:
            status = geo_fixed_norm_squared(a, &scalar);
            if (status == GEO_FIXED_OK) geo_fixed_set_projective(output, a, scalar);
            return status;
        case GEO_GEB_ANGLE_COSINE_NUMERATOR:
            status = geo_fixed_vector_dot(a, b, &scalar);
            if (status == GEO_FIXED_OK) { geo_fixed_set_scalar(output, scalar); return GEO_FIXED_OK; }
            return status;
        default:
            return GEO_FIXED_OVERFLOW;
    }

    if (status == GEO_FIXED_OK) geo_fixed_set_cl20(output, result);
    return status;
}
