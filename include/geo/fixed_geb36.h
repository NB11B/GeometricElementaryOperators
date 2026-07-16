#ifndef GEO_FIXED_GEB36_H
#define GEO_FIXED_GEB36_H

#include <stdint.h>

#include "geo/fixed.h"
#include "geo/geb36.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_FIXED_RESULT_CL20 = 0,
    GEO_FIXED_RESULT_SCALAR = 1,
    GEO_FIXED_RESULT_PROJECTIVE = 2,
    GEO_FIXED_RESULT_UNIPOTENT = 3
} geo_fixed_result_kind_t;

typedef struct {
    geo_fixed_cl20_t represented;
    geo_fixed_t denominator;
} geo_fixed_projective_t;

typedef struct {
    uint8_t kind;
    union {
        geo_fixed_cl20_t cl20;
        geo_fixed_t scalar;
        geo_fixed_projective_t projective;
        geo_fixed_cl20_t unipotent_payload;
    } as;
} geo_fixed_geb_result_t;

/*
 * Executes any frozen GEB-36 target in signed Q-format arithmetic.
 * Unary targets consume a. Binary targets consume a and b. Supplied-transform
 * targets consume transform as the supplied rotor/dilation element.
 */
geo_fixed_status_t geo_fixed_geb36_execute(
    uint8_t target_id,
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t transform,
    geo_fixed_geb_result_t *output
);

#ifdef __cplusplus
}
#endif

#endif
