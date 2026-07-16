#include "geo/eml_embedded.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

static int geo_eml_terms(geo_eml_profile_t profile) {
    switch (profile) {
        case GEO_EML_FAST: return 4;
        case GEO_EML_BALANCED: return 7;
        case GEO_EML_PRECISE: return 10;
        default: return 0;
    }
}

static geo_real_t geo_eml_exp_poly(geo_real_t r, int terms) {
    geo_real_t sum = (geo_real_t)1;
    geo_real_t term = (geo_real_t)1;
    int k;
    for (k = 1; k <= terms; ++k) {
        term *= r / (geo_real_t)k;
        sum += term;
    }
    return sum;
}

geo_eml_status_t geo_eml_exp(
    geo_real_t x,
    geo_eml_profile_t profile,
    geo_real_t *output
) {
    const int terms = geo_eml_terms(profile);
    const geo_real_t ln2 = (geo_real_t)0.693147180559945309417232121458176568;
    geo_real_t scaled;
    long exponent;
    geo_real_t remainder;
    geo_real_t approximation;

    if (output == NULL) return GEO_EML_NULL_ARGUMENT;
    if (terms == 0) return GEO_EML_DOMAIN;

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    if (x > (geo_real_t)709.0 || x < (geo_real_t)-745.0) return GEO_EML_OVERFLOW;
    scaled = x / ln2;
    exponent = lround((double)scaled);
#else
    if (x > (geo_real_t)88.0f || x < (geo_real_t)-103.0f) return GEO_EML_OVERFLOW;
    scaled = x / ln2;
    exponent = lroundf(scaled);
#endif

    remainder = x - (geo_real_t)exponent * ln2;
    approximation = geo_eml_exp_poly(remainder, terms);

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    approximation = ldexp(approximation, (int)exponent);
    if (!isfinite(approximation)) return GEO_EML_OVERFLOW;
#else
    approximation = ldexpf(approximation, (int)exponent);
    if (!isfinite((double)approximation)) return GEO_EML_OVERFLOW;
#endif

    *output = approximation;
    return GEO_EML_OK;
}

geo_eml_status_t geo_eml_log(
    geo_real_t x,
    geo_eml_profile_t profile,
    geo_real_t *output
) {
    const int terms = geo_eml_terms(profile);
    const geo_real_t ln2 = (geo_real_t)0.693147180559945309417232121458176568;
    geo_real_t mantissa;
    geo_real_t z;
    geo_real_t z2;
    geo_real_t power;
    geo_real_t sum;
    int exponent;
    int k;

    if (output == NULL) return GEO_EML_NULL_ARGUMENT;
    if (terms == 0 || x <= (geo_real_t)0) return GEO_EML_DOMAIN;

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    mantissa = frexp(x, &exponent);
#else
    mantissa = frexpf(x, &exponent);
#endif

    if (mantissa < (geo_real_t)0.7071067811865475244) {
        mantissa *= (geo_real_t)2;
        --exponent;
    }

    z = (mantissa - (geo_real_t)1) / (mantissa + (geo_real_t)1);
    z2 = z * z;
    power = z;
    sum = (geo_real_t)0;

    for (k = 0; k < terms; ++k) {
        sum += power / (geo_real_t)(2 * k + 1);
        power *= z2;
    }

    *output = (geo_real_t)2 * sum + (geo_real_t)exponent * ln2;
    return GEO_EML_OK;
}

geo_eml_status_t geo_eml_apply(
    geo_real_t x,
    geo_real_t y,
    geo_eml_profile_t profile,
    geo_real_t *output
) {
    geo_real_t exponential;
    geo_real_t logarithm;
    geo_eml_status_t status;

    if (output == NULL) return GEO_EML_NULL_ARGUMENT;

    status = geo_eml_exp(x, profile, &exponential);
    if (status != GEO_EML_OK) return status;
    status = geo_eml_log(y, profile, &logarithm);
    if (status != GEO_EML_OK) return status;

    *output = exponential - logarithm;
    return GEO_EML_OK;
}
