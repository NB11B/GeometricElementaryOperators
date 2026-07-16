#ifndef GEO_EML_EMBEDDED_H
#define GEO_EML_EMBEDDED_H

#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_EML_FAST = 0,
    GEO_EML_BALANCED = 1,
    GEO_EML_PRECISE = 2
} geo_eml_profile_t;

typedef enum {
    GEO_EML_OK = 0,
    GEO_EML_DOMAIN = 1,
    GEO_EML_OVERFLOW = 2,
    GEO_EML_NULL_ARGUMENT = 3
} geo_eml_status_t;

geo_eml_status_t geo_eml_exp(
    geo_real_t x,
    geo_eml_profile_t profile,
    geo_real_t *output
);

geo_eml_status_t geo_eml_log(
    geo_real_t x,
    geo_eml_profile_t profile,
    geo_real_t *output
);

geo_eml_status_t geo_eml_apply(
    geo_real_t x,
    geo_real_t y,
    geo_eml_profile_t profile,
    geo_real_t *output
);

#ifdef __cplusplus
}
#endif

#endif
