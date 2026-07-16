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
geo_fixed_status_t geo_fixed_div(geo_fixed_t a, geo_fixed_t b, geo_fixed_t *output);

geo_fixed_status_t geo_fixed_cl20_mul(
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t *output
);

#ifdef __cplusplus
}
#endif

#endif
