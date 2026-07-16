#include "geo/fixed.h"

#include <limits.h>
#include <math.h>

static int geo_fixed_in_range(int64_t value) {
    return value >= INT32_MIN && value <= INT32_MAX;
}

geo_fixed_status_t geo_fixed_from_double(double value, geo_fixed_t *output) {
    const double scale = (double)(UINT64_C(1) << GEO_FIXED_FRACTION_BITS);
    double scaled;
    int64_t rounded;

    if (output == NULL) {
        return GEO_FIXED_OVERFLOW;
    }

    scaled = value * scale;
    if (scaled > (double)INT32_MAX || scaled < (double)INT32_MIN) {
        return GEO_FIXED_OVERFLOW;
    }

    rounded = (int64_t)llround(scaled);
    if (!geo_fixed_in_range(rounded)) {
        return GEO_FIXED_OVERFLOW;
    }

    *output = (geo_fixed_t)rounded;
    return GEO_FIXED_OK;
}

double geo_fixed_to_double(geo_fixed_t value) {
    const double scale = (double)(UINT64_C(1) << GEO_FIXED_FRACTION_BITS);
    return (double)value / scale;
}

geo_fixed_status_t geo_fixed_mul(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output) {
    int64_t product;
    int64_t shifted;

    if (output == NULL) {
        return GEO_FIXED_OVERFLOW;
    }

    product = (int64_t)a * (int64_t)b;
    shifted = product >> GEO_FIXED_FRACTION_BITS;
    if (!geo_fixed_in_range(shifted)) {
        return GEO_FIXED_OVERFLOW;
    }

    *output = (geo_fixed_t)shifted;
    return GEO_FIXED_OK;
}

geo_fixed_status_t geo_fixed_div(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output) {
    int64_t numerator;
    int64_t quotient;

    if (output == NULL) {
        return GEO_FIXED_OVERFLOW;
    }
    if (b == 0) {
        return GEO_FIXED_DIVIDE_BY_ZERO;
    }

    numerator = ((int64_t)a) << GEO_FIXED_FRACTION_BITS;
    quotient = numerator / (int64_t)b;
    if (!geo_fixed_in_range(quotient)) {
        return GEO_FIXED_OVERFLOW;
    }

    *output = (geo_fixed_t)quotient;
    return GEO_FIXED_OK;
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

    if (output == NULL) {
        return GEO_FIXED_OVERFLOW;
    }

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
