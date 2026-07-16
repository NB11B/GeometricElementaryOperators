#ifndef GEO_FIXED_H
#define GEO_FIXED_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GEO_FIXED_FRACTION_BITS
#define GEO_FIXED_FRACTION_BITS 16
#endif

#if GEO_FIXED_FRACTION_BITS < 1 || GEO_FIXED_FRACTION_BITS > 30
#error "GEO_FIXED_FRACTION_BITS must be in the supported range 1..30 for signed 32-bit storage"
#endif

typedef int32_t geo_fixed_t;

typedef struct {
    geo_fixed_t scalar;
    geo_fixed_t e1;
    geo_fixed_t e2;
    geo_fixed_t e12;
} geo_fixed_cl20_t;

typedef enum {
    GEO_FIXED_OK = 0,
    GEO_FIXED_OVERFLOW = 1,
    GEO_FIXED_DIVIDE_BY_ZERO = 2
} geo_fixed_status_t;

geo_fixed_status_t geo_fixed_from_double(double value, geo_fixed_t *output);
double geo_fixed_to_double(geo_fixed_t value);
geo_fixed_status_t geo_fixed_mul(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output);
geo_fixed_status_t geo_fixed_mul_rounded(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output);
geo_fixed_status_t geo_fixed_div(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output);
geo_fixed_t geo_fixed_saturating_add(geo_fixed_t a, geo_fixed_t b);
geo_fixed_t geo_fixed_saturating_sub(geo_fixed_t a, geo_fixed_t b);

geo_fixed_status_t geo_fixed_cl20_mul(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
);

/* Compatibility helper. For exact error reporting use the checked form. */
geo_fixed_cl20_t geo_fixed_cl20_reverse(geo_fixed_cl20_t value);
geo_fixed_status_t geo_fixed_cl20_reverse_checked(
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
);
geo_fixed_status_t geo_fixed_cl20_grade_involution_checked(
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
);
geo_fixed_status_t geo_fixed_cl20_clifford_conjugate_checked(
    geo_fixed_cl20_t value,
    geo_fixed_cl20_t *output
);
geo_fixed_status_t geo_fixed_vector_dot(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_t *output
);
geo_fixed_status_t geo_fixed_vector_wedge(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_t *output
);
geo_fixed_status_t geo_fixed_rotor_action(
    geo_fixed_cl20_t rotor,
    geo_fixed_cl20_t vector,
    geo_fixed_cl20_t *output
);

#ifdef __cplusplus
}
#endif

#endif
