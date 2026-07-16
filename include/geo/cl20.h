#ifndef GEO_CL20_H
#define GEO_CL20_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
typedef double geo_real_t;
#else
typedef float geo_real_t;
#endif

typedef struct {
    geo_real_t scalar;
    geo_real_t e1;
    geo_real_t e2;
    geo_real_t e12;
} geo_cl20_t;

typedef enum {
    GEO_GRADE_SCALAR = 1u << 0,
    GEO_GRADE_VECTOR = 1u << 1,
    GEO_GRADE_BIVECTOR = 1u << 2,
    GEO_GRADE_ALL = GEO_GRADE_SCALAR | GEO_GRADE_VECTOR | GEO_GRADE_BIVECTOR
} geo_grade_mask_t;

geo_cl20_t geo_cl20_zero(void);
geo_cl20_t geo_cl20_one(void);
geo_cl20_t geo_cl20_basis_e1(void);
geo_cl20_t geo_cl20_basis_e2(void);
geo_cl20_t geo_cl20_basis_e12(void);
geo_cl20_t geo_cl20_make(
    geo_real_t scalar,
    geo_real_t e1,
    geo_real_t e2,
    geo_real_t e12
);

geo_cl20_t geo_cl20_add(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_cl20_sub(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_cl20_neg(geo_cl20_t value);
geo_cl20_t geo_cl20_scale(geo_cl20_t value, geo_real_t scale);
geo_cl20_t geo_cl20_mul(geo_cl20_t a, geo_cl20_t b);

geo_cl20_t geo_cl20_reverse(geo_cl20_t value);
geo_cl20_t geo_cl20_grade_involution(geo_cl20_t value);
geo_cl20_t geo_cl20_clifford_conjugate(geo_cl20_t value);
geo_cl20_t geo_cl20_project(geo_cl20_t value, uint8_t grade_mask);

geo_real_t geo_cl20_vector_dot(geo_cl20_t a, geo_cl20_t b);
geo_real_t geo_cl20_vector_wedge(geo_cl20_t a, geo_cl20_t b);
geo_real_t geo_cl20_vector_norm_squared(geo_cl20_t vector);

bool geo_cl20_near(geo_cl20_t a, geo_cl20_t b, geo_real_t tolerance);

#ifdef __cplusplus
}
#endif

#endif
