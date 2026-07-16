#ifndef GEO_NATIVE_GENERATED_H
#define GEO_NATIVE_GENERATED_H

#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Straight-line kernels emitted from the Cl(2,0) product table. */
geo_cl20_t geo_native_add(geo_cl20_t a, geo_cl20_t b);
geo_real_t geo_native_vector_dot(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_native_cl20_product(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_native_rotor_action(geo_cl20_t rotor, geo_cl20_t value);

#ifdef __cplusplus
}
#endif

#endif
