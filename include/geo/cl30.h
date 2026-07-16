#ifndef GEO_CL30_H
#define GEO_CL30_H

#include <stdbool.h>
#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    geo_real_t c[8];
} geo_cl30_t;

enum {
    GEO_CL30_SCALAR = 0,
    GEO_CL30_E1 = 1,
    GEO_CL30_E2 = 2,
    GEO_CL30_E12 = 3,
    GEO_CL30_E3 = 4,
    GEO_CL30_E13 = 5,
    GEO_CL30_E23 = 6,
    GEO_CL30_E123 = 7
};

geo_cl30_t geo_cl30_zero(void);
geo_cl30_t geo_cl30_basis(unsigned blade_index);
geo_cl30_t geo_cl30_add(geo_cl30_t a, geo_cl30_t b);
geo_cl30_t geo_cl30_mul(geo_cl30_t a, geo_cl30_t b);
geo_cl30_t geo_cl30_reverse(geo_cl30_t value);
bool geo_cl30_near(geo_cl30_t a, geo_cl30_t b, geo_real_t tolerance);

#ifdef __cplusplus
}
#endif

#endif
