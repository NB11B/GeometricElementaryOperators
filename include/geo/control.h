#ifndef GEO_CONTROL_H
#define GEO_CONTROL_H

#include <stdbool.h>

#include "geo/cl20.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    geo_real_t m00;
    geo_real_t m01;
    geo_real_t m10;
    geo_real_t m11;
} geo_mat2_t;

geo_mat2_t geo_mat2_zero(void);
geo_mat2_t geo_mat2_identity(void);
geo_mat2_t geo_mat2_neg_identity(void);
geo_mat2_t geo_mat2_e11(void);
geo_mat2_t geo_mat2_e12(void);
geo_mat2_t geo_mat2_e21(void);
geo_mat2_t geo_mat2_e22(void);
geo_mat2_t geo_mat2_exchange(void);

geo_mat2_t geo_mat2_make(
    geo_real_t m00,
    geo_real_t m01,
    geo_real_t m10,
    geo_real_t m11
);
geo_mat2_t geo_mat2_add(geo_mat2_t a, geo_mat2_t b);
geo_mat2_t geo_mat2_sub(geo_mat2_t a, geo_mat2_t b);
geo_mat2_t geo_mat2_mul(geo_mat2_t a, geo_mat2_t b);
geo_mat2_t geo_control_gc(geo_mat2_t x, geo_mat2_t y);

bool geo_mat2_near(geo_mat2_t a, geo_mat2_t b, geo_real_t tolerance);

#ifdef __cplusplus
}
#endif

#endif
