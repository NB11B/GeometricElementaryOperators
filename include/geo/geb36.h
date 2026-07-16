#ifndef GEO_GEB36_H
#define GEO_GEB36_H

#include <stddef.h>
#include <stdint.h>

#include "geo/structured.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_GEB_EXACT = 0,
    GEO_GEB_PROJECTIVE_SCALED = 1,
    GEO_GEB_EXACT_WITH_SUPPLIED_TRANSFORM = 2
} geo_geb_closure_t;

typedef enum {
    GEO_GEB_ZERO = 1,
    GEO_GEB_ONE = 2,
    GEO_GEB_MINUS_ONE = 3,
    GEO_GEB_E1 = 4,
    GEO_GEB_E2 = 5,
    GEO_GEB_PSEUDOSCALAR = 6,
    GEO_GEB_NEGATION = 7,
    GEO_GEB_REVERSION = 8,
    GEO_GEB_GRADE_INVOLUTION = 9,
    GEO_GEB_CLIFFORD_CONJUGATION = 10,
    GEO_GEB_SCALAR_PROJECTION = 11,
    GEO_GEB_VECTOR_PROJECTION = 12,
    GEO_GEB_BIVECTOR_PROJECTION = 13,
    GEO_GEB_ADDITION = 14,
    GEO_GEB_SUBTRACTION = 15,
    GEO_GEB_GEOMETRIC_PRODUCT = 16,
    GEO_GEB_REVERSE_PRODUCT = 17,
    GEO_GEB_VECTOR_DOT = 18,
    GEO_GEB_VECTOR_WEDGE = 19,
    GEO_GEB_COMMUTATOR = 20,
    GEO_GEB_ANTICOMMUTATOR = 21,
    GEO_GEB_VECTOR_NORM_SQUARED = 22,
    GEO_GEB_DISTANCE_SQUARED = 23,
    GEO_GEB_PROJECTION_NUMERATOR = 24,
    GEO_GEB_REJECTION_NUMERATOR = 25,
    GEO_GEB_REFLECTION_NUMERATOR = 26,
    GEO_GEB_DUAL = 27,
    GEO_GEB_EVEN_PROJECTION = 28,
    GEO_GEB_ODD_PROJECTION = 29,
    GEO_GEB_ROTOR_ACTION = 30,
    GEO_GEB_ROTOR_COMPOSITION = 31,
    GEO_GEB_ROTOR_NORM_SQUARED = 32,
    GEO_GEB_DILATION = 33,
    GEO_GEB_TRANSLATION_UNIPOTENT = 34,
    GEO_GEB_VECTOR_INVERSE_PROJECTIVE = 35,
    GEO_GEB_ANGLE_COSINE_NUMERATOR = 36
} geo_geb_target_id_t;

typedef struct {
    uint8_t id;
    const char *name;
    geo_geb_closure_t closure;
    int32_t known_scale_numerator;
    int32_t known_scale_denominator;
} geo_geb_target_info_t;

const geo_geb_target_info_t *geo_geb36_manifest(size_t *count);
const geo_geb_target_info_t *geo_geb36_target_info(uint8_t id);

geo_cl20_t geo_geb_zero(void);
geo_cl20_t geo_geb_one(void);
geo_cl20_t geo_geb_minus_one(void);
geo_cl20_t geo_geb_e1(void);
geo_cl20_t geo_geb_e2(void);
geo_cl20_t geo_geb_pseudoscalar(void);
geo_cl20_t geo_geb_negation(geo_cl20_t value);
geo_cl20_t geo_geb_reversion(geo_cl20_t value);
geo_cl20_t geo_geb_grade_involution(geo_cl20_t value);
geo_cl20_t geo_geb_clifford_conjugation(geo_cl20_t value);
geo_cl20_t geo_geb_scalar_projection(geo_cl20_t value);
geo_cl20_t geo_geb_vector_projection(geo_cl20_t value);
geo_cl20_t geo_geb_bivector_projection(geo_cl20_t value);
geo_cl20_t geo_geb_addition(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_subtraction(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_geometric_product(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_reverse_product(geo_cl20_t a, geo_cl20_t b);
geo_real_t geo_geb_vector_dot(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_vector_wedge(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_commutator(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_anticommutator(geo_cl20_t a, geo_cl20_t b);
geo_real_t geo_geb_vector_norm_squared(geo_cl20_t vector);
geo_real_t geo_geb_distance_squared(geo_cl20_t a, geo_cl20_t b);
geo_cl20_t geo_geb_projection_numerator(geo_cl20_t vector, geo_cl20_t onto);
geo_cl20_t geo_geb_rejection_numerator(geo_cl20_t vector, geo_cl20_t onto);
geo_cl20_t geo_geb_reflection_numerator(geo_cl20_t vector, geo_cl20_t normal);
geo_cl20_t geo_geb_dual(geo_cl20_t value);
geo_cl20_t geo_geb_even_projection(geo_cl20_t value);
geo_cl20_t geo_geb_odd_projection(geo_cl20_t value);
geo_cl20_t geo_geb_rotor_action(geo_cl20_t rotor, geo_cl20_t value);
geo_cl20_t geo_geb_rotor_composition(geo_cl20_t left, geo_cl20_t right);
geo_real_t geo_geb_rotor_norm_squared(geo_cl20_t rotor);
geo_cl20_t geo_geb_dilation(geo_cl20_t transform, geo_cl20_t value);
geo_unipotent_t geo_geb_translation_unipotent(geo_cl20_t translation);
geo_cl20_t geo_geb_vector_inverse_projective(geo_cl20_t vector);
geo_real_t geo_geb_angle_cosine_numerator(geo_cl20_t a, geo_cl20_t b);

#ifdef __cplusplus
}
#endif

#endif
