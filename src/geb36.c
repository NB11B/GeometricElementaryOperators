#include "geo/geb36.h"

static const geo_geb_target_info_t GEO_GEB36_MANIFEST[36] = {
    {1u, "zero", GEO_GEB_EXACT, 1, 1},
    {2u, "one", GEO_GEB_EXACT, 1, 1},
    {3u, "minus_one", GEO_GEB_EXACT, 1, 1},
    {4u, "e1", GEO_GEB_EXACT, 1, 1},
    {5u, "e2", GEO_GEB_EXACT, 1, 1},
    {6u, "pseudoscalar", GEO_GEB_EXACT, 1, 1},
    {7u, "negation", GEO_GEB_EXACT, 1, 1},
    {8u, "reversion", GEO_GEB_EXACT, 1, 1},
    {9u, "grade_involution", GEO_GEB_EXACT, 1, 1},
    {10u, "clifford_conjugation", GEO_GEB_EXACT, 1, 1},
    {11u, "scalar_projection", GEO_GEB_EXACT, 1, 1},
    {12u, "vector_projection", GEO_GEB_EXACT, 1, 1},
    {13u, "bivector_projection", GEO_GEB_EXACT, 1, 1},
    {14u, "addition", GEO_GEB_EXACT, 1, 1},
    {15u, "subtraction", GEO_GEB_EXACT, 1, 1},
    {16u, "geometric_product", GEO_GEB_EXACT, 1, 1},
    {17u, "reverse_product", GEO_GEB_EXACT, 1, 1},
    {18u, "vector_dot", GEO_GEB_EXACT, 1, 1},
    {19u, "vector_wedge", GEO_GEB_EXACT, 1, 1},
    {20u, "commutator", GEO_GEB_EXACT, 1, 1},
    {21u, "anticommutator", GEO_GEB_EXACT, 1, 1},
    {22u, "vector_norm_squared", GEO_GEB_EXACT, 1, 1},
    {23u, "distance_squared", GEO_GEB_EXACT, 1, 1},
    {24u, "projection_numerator", GEO_GEB_PROJECTIVE_SCALED, 1, 1},
    {25u, "rejection_numerator", GEO_GEB_PROJECTIVE_SCALED, 1, 1},
    {26u, "reflection_numerator", GEO_GEB_PROJECTIVE_SCALED, 1, 1},
    {27u, "dual", GEO_GEB_EXACT, 1, 1},
    {28u, "even_projection", GEO_GEB_EXACT, 1, 1},
    {29u, "odd_projection", GEO_GEB_EXACT, 1, 1},
    {30u, "rotor_action", GEO_GEB_EXACT_WITH_SUPPLIED_TRANSFORM, 1, 1},
    {31u, "rotor_composition", GEO_GEB_EXACT, 1, 1},
    {32u, "rotor_norm_squared", GEO_GEB_EXACT, 1, 1},
    {33u, "dilation", GEO_GEB_EXACT_WITH_SUPPLIED_TRANSFORM, 1, 1},
    {34u, "translation_unipotent", GEO_GEB_EXACT, 1, 1},
    {35u, "vector_inverse_projective", GEO_GEB_PROJECTIVE_SCALED, 1, 1},
    {36u, "angle_cosine_numerator", GEO_GEB_PROJECTIVE_SCALED, 1, 1}
};

const geo_geb_target_info_t *geo_geb36_manifest(size_t *count) {
    if (count != NULL) {
        *count = 36u;
    }
    return GEO_GEB36_MANIFEST;
}

const geo_geb_target_info_t *geo_geb36_target_info(uint8_t id) {
    if (id < 1u || id > 36u) {
        return NULL;
    }
    return &GEO_GEB36_MANIFEST[(size_t)id - 1u];
}

geo_cl20_t geo_geb_zero(void) { return geo_cl20_zero(); }
geo_cl20_t geo_geb_one(void) { return geo_cl20_one(); }
geo_cl20_t geo_geb_minus_one(void) { return geo_cl20_neg(geo_cl20_one()); }
geo_cl20_t geo_geb_e1(void) { return geo_cl20_basis_e1(); }
geo_cl20_t geo_geb_e2(void) { return geo_cl20_basis_e2(); }
geo_cl20_t geo_geb_pseudoscalar(void) { return geo_cl20_basis_e12(); }
geo_cl20_t geo_geb_negation(geo_cl20_t value) { return geo_cl20_neg(value); }
geo_cl20_t geo_geb_reversion(geo_cl20_t value) { return geo_cl20_reverse(value); }
geo_cl20_t geo_geb_grade_involution(geo_cl20_t value) { return geo_cl20_grade_involution(value); }
geo_cl20_t geo_geb_clifford_conjugation(geo_cl20_t value) { return geo_cl20_clifford_conjugate(value); }
geo_cl20_t geo_geb_scalar_projection(geo_cl20_t value) { return geo_cl20_project(value, GEO_GRADE_SCALAR); }
geo_cl20_t geo_geb_vector_projection(geo_cl20_t value) { return geo_cl20_project(value, GEO_GRADE_VECTOR); }
geo_cl20_t geo_geb_bivector_projection(geo_cl20_t value) { return geo_cl20_project(value, GEO_GRADE_BIVECTOR); }
geo_cl20_t geo_geb_addition(geo_cl20_t a, geo_cl20_t b) { return geo_cl20_add(a, b); }
geo_cl20_t geo_geb_subtraction(geo_cl20_t a, geo_cl20_t b) { return geo_cl20_sub(a, b); }
geo_cl20_t geo_geb_geometric_product(geo_cl20_t a, geo_cl20_t b) { return geo_cl20_mul(a, b); }
geo_cl20_t geo_geb_reverse_product(geo_cl20_t a, geo_cl20_t b) { return geo_cl20_reverse(geo_cl20_mul(a, b)); }
geo_real_t geo_geb_vector_dot(geo_cl20_t a, geo_cl20_t b) { return geo_cl20_vector_dot(a, b); }

geo_cl20_t geo_geb_vector_wedge(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, geo_cl20_vector_wedge(a, b));
}

geo_cl20_t geo_geb_commutator(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_scale(
        geo_cl20_sub(geo_cl20_mul(a, b), geo_cl20_mul(b, a)),
        (geo_real_t)0.5
    );
}

geo_cl20_t geo_geb_anticommutator(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_scale(
        geo_cl20_add(geo_cl20_mul(a, b), geo_cl20_mul(b, a)),
        (geo_real_t)0.5
    );
}

geo_real_t geo_geb_vector_norm_squared(geo_cl20_t vector) {
    return geo_cl20_vector_norm_squared(vector);
}

geo_real_t geo_geb_distance_squared(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_vector_norm_squared(geo_cl20_sub(a, b));
}

geo_cl20_t geo_geb_projection_numerator(geo_cl20_t vector, geo_cl20_t onto) {
    return geo_cl20_scale(onto, geo_cl20_vector_dot(vector, onto));
}

geo_cl20_t geo_geb_rejection_numerator(geo_cl20_t vector, geo_cl20_t onto) {
    const geo_real_t norm_squared = geo_cl20_vector_norm_squared(onto);
    return geo_cl20_sub(
        geo_cl20_scale(vector, norm_squared),
        geo_geb_projection_numerator(vector, onto)
    );
}

geo_cl20_t geo_geb_reflection_numerator(geo_cl20_t vector, geo_cl20_t normal) {
    const geo_real_t norm_squared = geo_cl20_vector_norm_squared(normal);
    return geo_cl20_sub(
        geo_cl20_scale(vector, norm_squared),
        geo_cl20_scale(
            geo_geb_projection_numerator(vector, normal),
            (geo_real_t)2
        )
    );
}

geo_cl20_t geo_geb_dual(geo_cl20_t value) {
    const geo_cl20_t inverse_pseudoscalar = geo_cl20_neg(geo_cl20_basis_e12());
    return geo_cl20_mul(value, inverse_pseudoscalar);
}

geo_cl20_t geo_geb_even_projection(geo_cl20_t value) {
    return geo_cl20_project(value, (uint8_t)(GEO_GRADE_SCALAR | GEO_GRADE_BIVECTOR));
}

geo_cl20_t geo_geb_odd_projection(geo_cl20_t value) {
    return geo_cl20_project(value, GEO_GRADE_VECTOR);
}

geo_cl20_t geo_geb_rotor_action(geo_cl20_t rotor, geo_cl20_t value) {
    return geo_cl20_mul(geo_cl20_mul(rotor, value), geo_cl20_reverse(rotor));
}

geo_cl20_t geo_geb_rotor_composition(geo_cl20_t left, geo_cl20_t right) {
    return geo_cl20_mul(left, right);
}

geo_real_t geo_geb_rotor_norm_squared(geo_cl20_t rotor) {
    return geo_cl20_mul(rotor, geo_cl20_reverse(rotor)).scalar;
}

geo_cl20_t geo_geb_dilation(geo_cl20_t transform, geo_cl20_t value) {
    return geo_cl20_mul(geo_cl20_mul(transform, value), geo_cl20_reverse(transform));
}

geo_unipotent_t geo_geb_translation_unipotent(geo_cl20_t translation) {
    return geo_unipotent_from_cl20(translation);
}

geo_cl20_t geo_geb_vector_inverse_projective(geo_cl20_t vector) {
    return vector;
}

geo_real_t geo_geb_angle_cosine_numerator(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_vector_dot(a, b);
}
