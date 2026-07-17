#ifndef GEO_FIXED_CONTROL_H
#define GEO_FIXED_CONTROL_H

#include "geo/fixed.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    geo_fixed_t m00;
    geo_fixed_t m01;
    geo_fixed_t m10;
    geo_fixed_t m11;
} geo_fixed_m2_t;

/* Computes the generative routing control Gc(X,Y)=XY-X. */
geo_fixed_status_t geo_fixed_control_apply(
    geo_fixed_m2_t left,
    geo_fixed_m2_t right,
    geo_fixed_m2_t *output
);

#ifdef __cplusplus
}
#endif

#endif
