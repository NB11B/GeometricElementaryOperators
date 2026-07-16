#ifndef GEO_STRUCTURED_H
#define GEO_STRUCTURED_H

#include "geo/omega.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * U(A) = [[1, A], [0, 1]]. The diagonal and lower-left entries are implicit,
 * so the embedded implementation stores only the payload A.
 */
typedef struct {
    geo_cl20_t payload;
} geo_unipotent_t;

/* Shared ordered products used by the Jordan/Lie (dot/wedge) mixer. */
typedef struct {
    geo_cl20_t ab;
    geo_cl20_t ba;
} geo_ordered_pair_t;

/*
 * represented = scale * canonical.
 * A projective result may retain scale != 1 until metric calibration is needed.
 */
typedef struct {
    geo_cl20_t represented;
    geo_scale_t scale;
} geo_scaled_cl20_t;

typedef struct {
    geo_scaled_cl20_t symmetric;
    geo_scaled_cl20_t antisymmetric;
} geo_hadamard_pair_t;

geo_unipotent_t geo_unipotent_from_cl20(geo_cl20_t value);
geo_unipotent_t geo_unipotent_identity(void);
geo_unipotent_t geo_unipotent_mul(geo_unipotent_t left, geo_unipotent_t right);
geo_cl20_t geo_unipotent_extract(geo_unipotent_t value);

geo_ordered_pair_t geo_ordered_products(geo_cl20_t a, geo_cl20_t b);

/*
 * Integer Hadamard mixer:
 *   symmetric     = ab + ba = 2 * Jordan(a,b)
 *   antisymmetric = ab - ba = 2 * Lie(a,b)
 * Both results carry scale 2 and avoid immediate division.
 */
geo_hadamard_pair_t geo_hadamard_mix_projective(geo_ordered_pair_t ordered);

/* Exact metric mixer. Results are divided by two and carry unit scale. */
geo_hadamard_pair_t geo_hadamard_mix_exact(geo_ordered_pair_t ordered);

/* Convert represented = scale * canonical into an exact unit-scale value. */
geo_status_t geo_scaled_cl20_normalize(
    const geo_scaled_cl20_t *input,
    geo_scaled_cl20_t *output
);

/* Vector-only metric helpers supplied by the scalar functional companion. */
geo_status_t geo_vector_inverse(geo_cl20_t vector, geo_cl20_t *output);
geo_status_t geo_vector_normalize(geo_cl20_t vector, geo_cl20_t *output);
geo_status_t geo_vector_projection(
    geo_cl20_t vector,
    geo_cl20_t onto,
    geo_cl20_t *output
);
geo_status_t geo_vector_rejection(
    geo_cl20_t vector,
    geo_cl20_t onto,
    geo_cl20_t *output
);
geo_status_t geo_vector_reflection(
    geo_cl20_t vector,
    geo_cl20_t normal,
    geo_cl20_t *output
);

#ifdef __cplusplus
}
#endif

#endif
