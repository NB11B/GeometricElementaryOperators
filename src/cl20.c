#include "geo/cl20.h"

#include <math.h>

geo_cl20_t geo_cl20_zero(void) {
    return geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, (geo_real_t)0);
}

geo_cl20_t geo_cl20_one(void) {
    return geo_cl20_make((geo_real_t)1, (geo_real_t)0, (geo_real_t)0, (geo_real_t)0);
}

geo_cl20_t geo_cl20_basis_e1(void) {
    return geo_cl20_make((geo_real_t)0, (geo_real_t)1, (geo_real_t)0, (geo_real_t)0);
}

geo_cl20_t geo_cl20_basis_e2(void) {
    return geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)1, (geo_real_t)0);
}

geo_cl20_t geo_cl20_basis_e12(void) {
    return geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, (geo_real_t)1);
}

geo_cl20_t geo_cl20_make(
    const geo_real_t scalar,
    const geo_real_t e1,
    const geo_real_t e2,
    const geo_real_t e12
) {
    const geo_cl20_t result = {
        .scalar = scalar,
        .e1 = e1,
        .e2 = e2,
        .e12 = e12
    };
    return result;
}

geo_cl20_t geo_cl20_add(const geo_cl20_t a, const geo_cl20_t b) {
    return geo_cl20_make(
        a.scalar + b.scalar,
        a.e1 + b.e1,
        a.e2 + b.e2,
        a.e12 + b.e12
    );
}

geo_cl20_t geo_cl20_sub(const geo_cl20_t a, const geo_cl20_t b) {
    return geo_cl20_make(
        a.scalar - b.scalar,
        a.e1 - b.e1,
        a.e2 - b.e2,
        a.e12 - b.e12
    );
}

geo_cl20_t geo_cl20_neg(const geo_cl20_t value) {
    return geo_cl20_make(-value.scalar, -value.e1, -value.e2, -value.e12);
}

geo_cl20_t geo_cl20_scale(const geo_cl20_t value, const geo_real_t scale) {
    return geo_cl20_make(
        value.scalar * scale,
        value.e1 * scale,
        value.e2 * scale,
        value.e12 * scale
    );
}

geo_cl20_t geo_cl20_mul(const geo_cl20_t a, const geo_cl20_t b) {
    return geo_cl20_make(
        (a.scalar * b.scalar) +
        (a.e1 * b.e1) +
        (a.e2 * b.e2) -
        (a.e12 * b.e12),

        (a.scalar * b.e1) +
        (a.e1 * b.scalar) -
        (a.e2 * b.e12) +
        (a.e12 * b.e2),

        (a.scalar * b.e2) +
        (a.e2 * b.scalar) +
        (a.e1 * b.e12) -
        (a.e12 * b.e1),

        (a.scalar * b.e12) +
        (a.e12 * b.scalar) +
        (a.e1 * b.e2) -
        (a.e2 * b.e1)
    );
}

geo_cl20_t geo_cl20_reverse(const geo_cl20_t value) {
    return geo_cl20_make(value.scalar, value.e1, value.e2, -value.e12);
}

geo_cl20_t geo_cl20_grade_involution(const geo_cl20_t value) {
    return geo_cl20_make(value.scalar, -value.e1, -value.e2, value.e12);
}

geo_cl20_t geo_cl20_clifford_conjugate(const geo_cl20_t value) {
    return geo_cl20_make(value.scalar, -value.e1, -value.e2, -value.e12);
}

geo_cl20_t geo_cl20_project(const geo_cl20_t value, const uint8_t grade_mask) {
    return geo_cl20_make(
        (grade_mask & GEO_GRADE_SCALAR) != 0u ? value.scalar : (geo_real_t)0,
        (grade_mask & GEO_GRADE_VECTOR) != 0u ? value.e1 : (geo_real_t)0,
        (grade_mask & GEO_GRADE_VECTOR) != 0u ? value.e2 : (geo_real_t)0,
        (grade_mask & GEO_GRADE_BIVECTOR) != 0u ? value.e12 : (geo_real_t)0
    );
}

geo_real_t geo_cl20_vector_dot(const geo_cl20_t a, const geo_cl20_t b) {
    return (a.e1 * b.e1) + (a.e2 * b.e2);
}

geo_real_t geo_cl20_vector_wedge(const geo_cl20_t a, const geo_cl20_t b) {
    return (a.e1 * b.e2) - (a.e2 * b.e1);
}

geo_real_t geo_cl20_vector_norm_squared(const geo_cl20_t vector) {
    return geo_cl20_vector_dot(vector, vector);
}

bool geo_cl20_near(
    const geo_cl20_t a,
    const geo_cl20_t b,
    const geo_real_t tolerance
) {
    return
        fabs((double)(a.scalar - b.scalar)) <= (double)tolerance &&
        fabs((double)(a.e1 - b.e1)) <= (double)tolerance &&
        fabs((double)(a.e2 - b.e2)) <= (double)tolerance &&
        fabs((double)(a.e12 - b.e12)) <= (double)tolerance;
}
