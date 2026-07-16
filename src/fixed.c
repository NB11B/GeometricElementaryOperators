#include "geo/fixed.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>

static int geo_fixed_in_range(int64_t value) {
    return value >= INT32_MIN && value <= INT32_MAX;
}

static int64_t geo_fixed_scale_i64(void) {
    return INT64_C(1) << GEO_FIXED_FRACTION_BITS;
}

static int64_t geo_fixed_round_divide(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    const uint64_t abs_remainder = remainder < 0 ? (uint64_t)(-remainder) : (uint64_t)remainder;
    const uint64_t abs_denominator = denominator < 0 ? (uint64_t)(-denominator) : (uint64_t)denominator;

    if (abs_remainder >= (abs_denominator + 1u) / 2u) {
        quotient += ((numerator < 0) != (denominator < 0)) ? -1 : 1;
    }
    return quotient;
}

static geo_fixed_status_t geo_fixed_neg_checked(geo_fixed_t input, geo_fixed_t *output) {
    if (output == NULL || input == INT32_MIN) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)-input;
    return GEO_FIXED_OK;
}

geo_fixed_status_t geo_fixed_from_double(double value, geo_fixed_t *output) {
    const double scale = (double)geo_fixed_scale_i64();
    double scaled;
    int64_t rounded;

    if (output == NULL || !isfinite(value)) return GEO_FIXED_OVERFLOW;
    scaled = value * scale;
    if (scaled > (double)INT32_MAX || scaled < (double)INT32_MIN) return GEO_FIXED_OVERFLOW;
    rounded = (int64_t)llround(scaled);
    if (!geo_fixed_in_range(rounded)) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)rounded;
    return GEO_FIXED_OK;
}

double geo_fixed_to_double(geo_fixed_t value) {
    return (double)value / (double)geo_fixed_scale_i64();
}

geo_fixed_status_t geo_fixed_mul_rounded(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output) {
    const int64_t product = (int64_t)a * (int64_t)b;
    const int64_t rounded = geo_fixed_round_divide(product, geo_fixed_scale_i64());
    if (output == NULL || !geo_fixed_in_range(rounded)) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)rounded;
    return GEO_FIXED_OK;
}

geo_fixed_status_t geo_fixed_mul(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output) {
    return geo_fixed_mul_rounded(a, b, output);
}

geo_fixed_status_t geo_fixed_div(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output) {
    const int64_t scale = geo_fixed_scale_i64();
    int64_t numerator;
    int64_t quotient;

    if (output == NULL) return GEO_FIXED_OVERFLOW;
    if (b == 0) return GEO_FIXED_DIVIDE_BY_ZERO;
    if (a != 0 && (a > 0 ? a : -(int64_t)a) > INT64_MAX / scale) {
        return GEO_FIXED_OVERFLOW;
    }
    numerator = (int64_t)a * scale;
    quotient = geo_fixed_round_divide(numerator, (int64_t)b);
    if (!geo_fixed_in_range(quotient)) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)quotient;
    return GEO_FIXED_OK;
}

geo_fixed_t geo_fixed_saturating_add(geo_fixed_t a, geo_fixed_t b) {
    const int64_t value = (int64_t)a + (int64_t)b;
    if (value > INT32_MAX) return INT32_MAX;
    if (value < INT32_MIN) return INT32_MIN;
    return (geo_fixed_t)value;
}

geo_fixed_t geo_fixed_saturating_sub(geo_fixed_t a, geo_fixed_t b) {
    const int64_t value = (int64_t)a - (int64_t)b;
    if (value > INT32_MAX) return INT32_MAX;
    if (value < INT32_MIN) return INT32_MIN;
    return (geo_fixed_t)value;
}

geo_fixed_status_t geo_fixed_cl20_mul(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
) {
    geo_fixed_t p00, p11, p22, p33;
    geo_fixed_t p01, p10, p23, p32;
    geo_fixed_t p02, p20, p13, p31;
    geo_fixed_t p03, p30, p12, p21;
    int64_t value;

    if (output == NULL) return GEO_FIXED_OVERFLOW;

#define GEO_FIXED_MUL_OR_RETURN(x, y, out) \
    do { \
        geo_fixed_status_t s_ = geo_fixed_mul((x), (y), &(out)); \
        if (s_ != GEO_FIXED_OK) return s_; \
    } while (0)

    GEO_FIXED_MUL_OR_RETURN(a.scalar, b.scalar, p00);
    GEO_FIXED_MUL_OR_RETURN(a.e1, b.e1, p11);
    GEO_FIXED_MUL_OR_RETURN(a.e2, b.e2, p22);
    GEO_FIXED_MUL_OR_RETURN(a.e12, b.e12, p33);
    GEO_FIXED_MUL_OR_RETURN(a.scalar, b.e1, p01);
    GEO_FIXED_MUL_OR_RETURN(a.e1, b.scalar, p10);
    GEO_FIXED_MUL_OR_RETURN(a.e2, b.e12, p23);
    GEO_FIXED_MUL_OR_RETURN(a.e12, b.e2, p32);
    GEO_FIXED_MUL_OR_RETURN(a.scalar, b.e2, p02);
    GEO_FIXED_MUL_OR_RETURN(a.e2, b.scalar, p20);
    GEO_FIXED_MUL_OR_RETURN(a.e1, b.e12, p13);
    GEO_FIXED_MUL_OR_RETURN(a.e12, b.e1, p31);
    GEO_FIXED_MUL_OR_RETURN(a.scalar, b.e12, p03);
    GEO_FIXED_MUL_OR_RETURN(a.e12, b.scalar, p30);
    GEO_FIXED_MUL_OR_RETURN(a.e1, b.e2, p12);
    GEO_FIXED_MUL_OR_RETURN(a.e2, b.e1, p21);

#undef GEO_FIXED_MUL_OR_RETURN

    value = (int64_t)p00 + p11 + p22 - p33;
    if (!geo_fixed_in_range(value)) return GEO_FIXED_OVERFLOW;
    output->scalar = (geo_fixed_t)value;
    value = (int64_t)p01 + p10 - p23 + p32;
    if (!geo_fixed_in_range(value)) return GEO_FIXED_OVERFLOW;
    output->e1 = (geo_fixed_t)value;
    value = (int64_t)p02 + p20 + p13 - p31;
    if (!geo_fixed_in_range(value)) return GEO_FIXED_OVERFLOW;
    output->e2 = (geo_fixed_t)value;
    value = (int64_t)p03 + p30 + p12 - p21;
    if (!geo_fixed_in_range(value)) return GEO_FIXED_OVERFLOW;
    output->e12 = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}

geo_fixed_status_t geo_fixed_cl20_reverse_checked(
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
) {
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    *output = value;
    status = geo_fixed_neg_checked(value.e12, &output->e12);
    return status;
}

geo_fixed_cl20_t geo_fixed_cl20_reverse(geo_fixed_cl20_t value) {
    geo_fixed_cl20_t output = value;
    if (geo_fixed_cl20_reverse_checked(value, &output) != GEO_FIXED_OK) {
        output.e12 = value.e12;
    }
    return output;
}

geo_fixed_status_t geo_fixed_cl20_grade_involution_checked(
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
) {
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    *output = value;
    status = geo_fixed_neg_checked(value.e1, &output->e1);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_neg_checked(value.e2, &output->e2);
}

geo_fixed_status_t geo_fixed_cl20_clifford_conjugate_checked(
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
) {
    geo_fixed_status_t status;
    geo_fixed_cl20_t temporary;
    status = geo_fixed_cl20_grade_involution_checked(value, &temporary);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_cl20_reverse_checked(temporary, output);
}

geo_fixed_status_t geo_fixed_vector_dot(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_t *output
) {
    geo_fixed_t x;
    geo_fixed_t y;
    int64_t sum;
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_mul(a.e1, b.e1, &x);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mul(a.e2, b.e2, &y);
    if (status != GEO_FIXED_OK) return status;
    sum = (int64_t)x + (int64_t)y;
    if (!geo_fixed_in_range(sum)) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)sum;
    return GEO_FIXED_OK;
}

geo_fixed_status_t geo_fixed_vector_wedge(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_t *output
) {
    geo_fixed_t xy;
    geo_fixed_t yx;
    int64_t value;
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_mul(a.e1, b.e2, &xy);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mul(a.e2, b.e1, &yx);
    if (status != GEO_FIXED_OK) return status;
    value = (int64_t)xy - (int64_t)yx;
    if (!geo_fixed_in_range(value)) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}

geo_fixed_status_t geo_fixed_rotor_action(
    geo_fixed_cl20_t rotor,
    geo_fixed_cl20_t vector,
    geo_fixed_cl20_t *output
) {
    geo_fixed_cl20_t temporary;
    geo_fixed_cl20_t reverse;
    geo_fixed_status_t status;
    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_cl20_mul(rotor, vector, &temporary);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_cl20_reverse_checked(rotor, &reverse);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_cl20_mul(temporary, reverse, output);
}
