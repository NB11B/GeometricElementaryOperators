#include "geo/fixed_geb36.h"

#include <limits.h>
#include <stddef.h>

static geo_fixed_t one_value(void) {
    return (geo_fixed_t)(INT64_C(1) << GEO_FIXED_FRACTION_BITS);
}

static geo_fixed_cl20_t zero_mv(void) {
    const geo_fixed_cl20_t value = {0, 0, 0, 0};
    return value;
}

static geo_fixed_cl20_t one_mv(void) {
    geo_fixed_cl20_t value = zero_mv();
    value.scalar = one_value();
    return value;
}

static geo_fixed_status_t checked_i64(int64_t value, geo_fixed_t *output) {
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) return GEO_FIXED_OVERFLOW;
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t mv_add(geo_fixed_cl20_t a, geo_fixed_cl20_t b, geo_fixed_cl20_t *o) {
    geo_fixed_status_t s;
    if (o == NULL) return GEO_FIXED_OVERFLOW;
    s = checked_i64((int64_t)a.scalar + b.scalar, &o->scalar); if (s != GEO_FIXED_OK) return s;
    s = checked_i64((int64_t)a.e1 + b.e1, &o->e1); if (s != GEO_FIXED_OK) return s;
    s = checked_i64((int64_t)a.e2 + b.e2, &o->e2); if (s != GEO_FIXED_OK) return s;
    return checked_i64((int64_t)a.e12 + b.e12, &o->e12);
}

static geo_fixed_status_t mv_sub(geo_fixed_cl20_t a, geo_fixed_cl20_t b, geo_fixed_cl20_t *o) {
    geo_fixed_status_t s;
    if (o == NULL) return GEO_FIXED_OVERFLOW;
    s = checked_i64((int64_t)a.scalar - b.scalar, &o->scalar); if (s != GEO_FIXED_OK) return s;
    s = checked_i64((int64_t)a.e1 - b.e1, &o->e1); if (s != GEO_FIXED_OK) return s;
    s = checked_i64((int64_t)a.e2 - b.e2, &o->e2); if (s != GEO_FIXED_OK) return s;
    return checked_i64((int64_t)a.e12 - b.e12, &o->e12);
}

static geo_fixed_status_t mv_neg(geo_fixed_cl20_t v, geo_fixed_cl20_t *o) {
    if (o == NULL || v.scalar == INT32_MIN || v.e1 == INT32_MIN ||
        v.e2 == INT32_MIN || v.e12 == INT32_MIN) return GEO_FIXED_OVERFLOW;
    o->scalar = -v.scalar; o->e1 = -v.e1; o->e2 = -v.e2; o->e12 = -v.e12;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t mv_scale(geo_fixed_cl20_t v, geo_fixed_t k, geo_fixed_cl20_t *o) {
    geo_fixed_status_t s;
    if (o == NULL) return GEO_FIXED_OVERFLOW;
    s = geo_fixed_mul(v.scalar, k, &o->scalar); if (s != GEO_FIXED_OK) return s;
    s = geo_fixed_mul(v.e1, k, &o->e1); if (s != GEO_FIXED_OK) return s;
    s = geo_fixed_mul(v.e2, k, &o->e2); if (s != GEO_FIXED_OK) return s;
    return geo_fixed_mul(v.e12, k, &o->e12);
}

static geo_fixed_cl20_t project(geo_fixed_cl20_t v, uint8_t mask) {
    if ((mask & GEO_GRADE_SCALAR) == 0u) v.scalar = 0;
    if ((mask & GEO_GRADE_VECTOR) == 0u) { v.e1 = 0; v.e2 = 0; }
    if ((mask & GEO_GRADE_BIVECTOR) == 0u) v.e12 = 0;
    return v;
}

static geo_fixed_status_t norm2(geo_fixed_cl20_t v, geo_fixed_t *o) {
    return geo_fixed_vector_dot(v, v, o);
}

static geo_fixed_status_t dot_mv(geo_fixed_cl20_t a, geo_fixed_cl20_t b, geo_fixed_cl20_t *o) {
    geo_fixed_t d;
    geo_fixed_status_t s = geo_fixed_vector_dot(a, b, &d);
    if (s != GEO_FIXED_OK || o == NULL) return s;
    *o = zero_mv(); o->scalar = d; return GEO_FIXED_OK;
}

static geo_fixed_status_t wedge_mv(geo_fixed_cl20_t a, geo_fixed_cl20_t b, geo_fixed_cl20_t *o) {
    geo_fixed_t w;
    geo_fixed_status_t s = geo_fixed_vector_wedge(a, b, &w);
    if (s != GEO_FIXED_OK || o == NULL) return s;
    *o = zero_mv(); o->e12 = w; return GEO_FIXED_OK;
}

static geo_fixed_status_t half_sum_difference(
    geo_fixed_cl20_t a, geo_fixed_cl20_t b, int difference, geo_fixed_cl20_t *o
) {
    geo_fixed_cl20_t ab, ba, combined;
    geo_fixed_status_t s;
    const geo_fixed_t half = (geo_fixed_t)(one_value() / 2);
    s = geo_fixed_cl20_mul(a, b, &ab); if (s != GEO_FIXED_OK) return s;
    s = geo_fixed_cl20_mul(b, a, &ba); if (s != GEO_FIXED_OK) return s;
    s = difference ? mv_sub(ab, ba, &combined) : mv_add(ab, ba, &combined);
    if (s != GEO_FIXED_OK) return s;
    return mv_scale(combined, half, o);
}

static geo_fixed_status_t projection_num(geo_fixed_cl20_t v, geo_fixed_cl20_t onto, geo_fixed_cl20_t *o) {
    geo_fixed_t d;
    geo_fixed_status_t s = geo_fixed_vector_dot(v, onto, &d);
    if (s != GEO_FIXED_OK) return s;
    return mv_scale(onto, d, o);
}

static geo_fixed_status_t rejection_num(geo_fixed_cl20_t v, geo_fixed_cl20_t onto, geo_fixed_cl20_t *o) {
    geo_fixed_t n;
    geo_fixed_cl20_t sv, p;
    geo_fixed_status_t s = norm2(onto, &n);
    if (s != GEO_FIXED_OK) return s;
    s = mv_scale(v, n, &sv); if (s != GEO_FIXED_OK) return s;
    s = projection_num(v, onto, &p); if (s != GEO_FIXED_OK) return s;
    return mv_sub(sv, p, o);
}

static geo_fixed_status_t reflection_num(geo_fixed_cl20_t v, geo_fixed_cl20_t nrm, geo_fixed_cl20_t *o) {
    geo_fixed_t n, d2;
    geo_fixed_cl20_t sv, sn;
    geo_fixed_status_t s = norm2(nrm, &n);
    if (s != GEO_FIXED_OK) return s;
    s = mv_scale(v, n, &sv); if (s != GEO_FIXED_OK) return s;
    s = geo_fixed_vector_dot(v, nrm, &d2); if (s != GEO_FIXED_OK) return s;
    s = checked_i64((int64_t)d2 * 2, &d2); if (s != GEO_FIXED_OK) return s;
    s = mv_scale(nrm, d2, &sn); if (s != GEO_FIXED_OK) return s;
    return mv_sub(sv, sn, o);
}

static void set_cl20(geo_fixed_geb_result_t *o, geo_fixed_cl20_t v) {
    o->kind = (uint8_t)GEO_FIXED_RESULT_CL20; o->as.cl20 = v;
}
static void set_scalar(geo_fixed_geb_result_t *o, geo_fixed_t v) {
    o->kind = (uint8_t)GEO_FIXED_RESULT_SCALAR; o->as.scalar = v;
}
static void set_projective(geo_fixed_geb_result_t *o, geo_fixed_cl20_t v, geo_fixed_t d) {
    o->kind = (uint8_t)GEO_FIXED_RESULT_PROJECTIVE;
    o->as.projective.represented = v; o->as.projective.denominator = d;
}

geo_fixed_status_t geo_fixed_geb36_execute(
    uint8_t target_id,
    geo_fixed_cl20_t a,
    geo_fixed_cl20_t b,
    geo_fixed_cl20_t transform,
    geo_fixed_geb_result_t *output
) {
    geo_fixed_cl20_t r = zero_mv(), t = zero_mv();
    geo_fixed_t s = 0;
    geo_fixed_status_t st = GEO_FIXED_OK;
    if (output == NULL) return GEO_FIXED_OVERFLOW;

    switch ((geo_geb_target_id_t)target_id) {
        case GEO_GEB_ZERO: set_cl20(output, zero_mv()); return GEO_FIXED_OK;
        case GEO_GEB_ONE: set_cl20(output, one_mv()); return GEO_FIXED_OK;
        case GEO_GEB_MINUS_ONE:
            r = one_mv(); r.scalar = -r.scalar; set_cl20(output, r); return GEO_FIXED_OK;
        case GEO_GEB_E1: r = zero_mv(); r.e1 = one_value(); set_cl20(output, r); return GEO_FIXED_OK;
        case GEO_GEB_E2: r = zero_mv(); r.e2 = one_value(); set_cl20(output, r); return GEO_FIXED_OK;
        case GEO_GEB_PSEUDOSCALAR: r = zero_mv(); r.e12 = one_value(); set_cl20(output, r); return GEO_FIXED_OK;
        case GEO_GEB_NEGATION: st = mv_neg(a, &r); break;
        case GEO_GEB_REVERSION: st = geo_fixed_cl20_reverse_checked(a, &r); break;
        case GEO_GEB_GRADE_INVOLUTION: st = geo_fixed_cl20_grade_involution_checked(a, &r); break;
        case GEO_GEB_CLIFFORD_CONJUGATION: st = geo_fixed_cl20_clifford_conjugate_checked(a, &r); break;
        case GEO_GEB_SCALAR_PROJECTION: r = project(a, GEO_GRADE_SCALAR); break;
        case GEO_GEB_VECTOR_PROJECTION: r = project(a, GEO_GRADE_VECTOR); break;
        case GEO_GEB_BIVECTOR_PROJECTION: r = project(a, GEO_GRADE_BIVECTOR); break;
        case GEO_GEB_ADDITION: st = mv_add(a, b, &r); break;
        case GEO_GEB_SUBTRACTION: st = mv_sub(a, b, &r); break;
        case GEO_GEB_GEOMETRIC_PRODUCT: st = geo_fixed_cl20_mul(a, b, &r); break;
        case GEO_GEB_REVERSE_PRODUCT:
            st = geo_fixed_cl20_mul(a, b, &t);
            if (st == GEO_FIXED_OK) st = geo_fixed_cl20_reverse_checked(t, &r);
            break;
        case GEO_GEB_VECTOR_DOT:
            st = geo_fixed_vector_dot(a, b, &s); if (st == GEO_FIXED_OK) set_scalar(output, s); return st;
        case GEO_GEB_VECTOR_WEDGE: st = wedge_mv(a, b, &r); break;
        case GEO_GEB_COMMUTATOR: st = half_sum_difference(a, b, 1, &r); break;
        case GEO_GEB_ANTICOMMUTATOR: st = half_sum_difference(a, b, 0, &r); break;
        case GEO_GEB_VECTOR_NORM_SQUARED:
            st = norm2(a, &s); if (st == GEO_FIXED_OK) set_scalar(output, s); return st;
        case GEO_GEB_DISTANCE_SQUARED:
            st = mv_sub(a, b, &r); if (st == GEO_FIXED_OK) st = norm2(r, &s);
            if (st == GEO_FIXED_OK) set_scalar(output, s); return st;
        case GEO_GEB_PROJECTION_NUMERATOR:
            st = projection_num(a, b, &r); if (st == GEO_FIXED_OK) set_projective(output, r, one_value()); return st;
        case GEO_GEB_REJECTION_NUMERATOR:
            st = rejection_num(a, b, &r); if (st == GEO_FIXED_OK) set_projective(output, r, one_value()); return st;
        case GEO_GEB_REFLECTION_NUMERATOR:
            st = reflection_num(a, b, &r); if (st == GEO_FIXED_OK) set_projective(output, r, one_value()); return st;
        case GEO_GEB_DUAL:
            t = zero_mv(); t.e12 = -one_value(); st = geo_fixed_cl20_mul(a, t, &r); break;
        case GEO_GEB_EVEN_PROJECTION: r = project(a, (uint8_t)(GEO_GRADE_SCALAR | GEO_GRADE_BIVECTOR)); break;
        case GEO_GEB_ODD_PROJECTION: r = project(a, GEO_GRADE_VECTOR); break;
        case GEO_GEB_ROTOR_ACTION: st = geo_fixed_rotor_action(transform, a, &r); break;
        case GEO_GEB_ROTOR_COMPOSITION: st = geo_fixed_cl20_mul(a, b, &r); break;
        case GEO_GEB_ROTOR_NORM_SQUARED:
            st = geo_fixed_cl20_reverse_checked(a, &t);
            if (st == GEO_FIXED_OK) st = geo_fixed_cl20_mul(a, t, &r);
            if (st == GEO_FIXED_OK) set_scalar(output, r.scalar); return st;
        case GEO_GEB_DILATION: st = geo_fixed_rotor_action(transform, a, &r); break;
        case GEO_GEB_TRANSLATION_UNIPOTENT:
            output->kind = (uint8_t)GEO_FIXED_RESULT_UNIPOTENT;
            output->as.unipotent_payload = a; return GEO_FIXED_OK;
        case GEO_GEB_VECTOR_INVERSE_PROJECTIVE:
            st = norm2(a, &s); if (st == GEO_FIXED_OK) set_projective(output, a, s); return st;
        case GEO_GEB_ANGLE_COSINE_NUMERATOR:
            st = geo_fixed_vector_dot(a, b, &s); if (st == GEO_FIXED_OK) set_scalar(output, s); return st;
        default: return GEO_FIXED_OVERFLOW;
    }

    if (st == GEO_FIXED_OK) set_cl20(output, r);
    return st;
}
