#ifndef GEO_OPPOSITE_H
#define GEO_OPPOSITE_H

#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    geo_cl20_t forward;
    geo_cl20_t reverse;
} geo_opposite_t;

geo_opposite_t geo_opposite_zero(void);
geo_opposite_t geo_opposite_identity(void);
geo_opposite_t geo_opposite_from_cl20(geo_cl20_t value);
geo_opposite_t geo_opposite_mul(geo_opposite_t a, geo_opposite_t b);

bool geo_opposite_is_consistent(
    geo_opposite_t value,
    geo_real_t tolerance
);

#ifdef __cplusplus
}
#endif

#endif
