#include "geo/cl30.h"

#include <math.h>

static const int8_t GEO_CL30_TABLE[8][8] = {
    { 1,  2,  3,  4,  5,  6,  7,  8},
    { 2,  1,  4,  3,  6,  5,  8,  7},
    { 3, -4,  1, -2,  7, -8,  5, -6},
    { 4, -3,  2, -1,  8, -7,  6, -5},
    { 5, -6, -7,  8,  1, -2, -3,  4},
    { 6, -5, -8,  7,  2, -1, -4,  3},
    { 7,  8, -5, -6,  3,  4, -1, -2},
    { 8,  7, -6, -5,  4,  3, -2, -1}
};

geo_cl30_t geo_cl30_zero(void) {
    geo_cl30_t result = {{0}};
    return result;
}

geo_cl30_t geo_cl30_basis(unsigned blade_index) {
    geo_cl30_t result = geo_cl30_zero();
    if (blade_index < 8u) result.c[blade_index] = (geo_real_t)1;
    return result;
}

geo_cl30_t geo_cl30_add(geo_cl30_t a, geo_cl30_t b) {
    geo_cl30_t result;
    unsigned i;
    for (i = 0u; i < 8u; ++i) result.c[i] = a.c[i] + b.c[i];
    return result;
}

geo_cl30_t geo_cl30_mul(geo_cl30_t a, geo_cl30_t b) {
    geo_cl30_t result = geo_cl30_zero();
    unsigned i;
    unsigned j;
    for (i = 0u; i < 8u; ++i) {
        for (j = 0u; j < 8u; ++j) {
            const int8_t entry = GEO_CL30_TABLE[i][j];
            const unsigned index = (unsigned)(entry < 0 ? -entry : entry) - 1u;
            const geo_real_t sign = entry < 0 ? (geo_real_t)-1 : (geo_real_t)1;
            result.c[index] += sign * a.c[i] * b.c[j];
        }
    }
    return result;
}

geo_cl30_t geo_cl30_reverse(geo_cl30_t value) {
    value.c[GEO_CL30_E12] = -value.c[GEO_CL30_E12];
    value.c[GEO_CL30_E13] = -value.c[GEO_CL30_E13];
    value.c[GEO_CL30_E23] = -value.c[GEO_CL30_E23];
    value.c[GEO_CL30_E123] = -value.c[GEO_CL30_E123];
    return value;
}

bool geo_cl30_near(geo_cl30_t a, geo_cl30_t b, geo_real_t tolerance) {
    unsigned i;
    const double t = (double)tolerance;
    if (!isfinite(t) || t < 0.0) return false;
    for (i = 0u; i < 8u; ++i) {
        const double difference = (double)a.c[i] - (double)b.c[i];
        if (!isfinite(difference) || fabs(difference) > t) return false;
    }
    return true;
}
