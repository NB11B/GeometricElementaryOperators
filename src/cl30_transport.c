#include "geo/cl30_transport.h"
#include <math.h>

geo_cl30_rotor_t geo_cl30_rotor_exp(geo_real_t b12, geo_real_t b23, geo_real_t b31) {
    geo_real_t theta_sq = b12 * b12 + b23 * b23 + b31 * b31;
    geo_real_t theta = (geo_real_t)sqrt((double)theta_sq);
    geo_cl30_rotor_t r;

    if (theta < (geo_real_t)1e-7) {
        r.s = (geo_real_t)1.0 - (geo_real_t)0.125 * theta_sq;
        geo_real_t k = (geo_real_t)-0.5 + (geo_real_t)(1.0 / 48.0) * theta_sq;
        r.b12 = k * b12;
        r.b23 = k * b23;
        r.b31 = k * b31;
    } else {
        geo_real_t half_theta = (geo_real_t)0.5 * theta;
        r.s = (geo_real_t)cos((double)half_theta);
        geo_real_t k = -(geo_real_t)sin((double)half_theta) / theta;
        r.b12 = k * b12;
        r.b23 = k * b23;
        r.b31 = k * b31;
    }

    return geo_cl30_rotor_normalize(r);
}

geo_cl30_rotor_t geo_cl30_rotor_normalize(geo_cl30_rotor_t r) {
    geo_real_t norm_sq = r.s * r.s + r.b12 * r.b12 + r.b23 * r.b23 + r.b31 * r.b31;
    if (norm_sq > (geo_real_t)0.0) {
        geo_real_t inv_norm = (geo_real_t)(1.0 / sqrt((double)norm_sq));
        r.s *= inv_norm;
        r.b12 *= inv_norm;
        r.b23 *= inv_norm;
        r.b31 *= inv_norm;
    } else {
        r.s = (geo_real_t)1.0;
        r.b12 = (geo_real_t)0.0;
        r.b23 = (geo_real_t)0.0;
        r.b31 = (geo_real_t)0.0;
    }
    return r;
}

geo_cl30_rotor_t geo_cl30_rotor_reverse(geo_cl30_rotor_t r) {
    r.b12 = -r.b12;
    r.b23 = -r.b23;
    r.b31 = -r.b31;
    return r;
}

geo_cl30_rotor_t geo_cl30_rotor_mul(geo_cl30_rotor_t r1, geo_cl30_rotor_t r2) {
    geo_cl30_rotor_t out;
    out.s   = r1.s * r2.s   - r1.b12 * r2.b12 - r1.b23 * r2.b23 - r1.b31 * r2.b31;
    out.b12 = r1.s * r2.b12 + r1.b12 * r2.s   - r1.b23 * r2.b31 + r1.b31 * r2.b23;
    out.b23 = r1.s * r2.b23 + r1.b23 * r2.s   - r1.b31 * r2.b12 + r1.b12 * r2.b31;
    out.b31 = r1.s * r2.b31 + r1.b31 * r2.s   - r1.b12 * r2.b23 + r1.b23 * r2.b12;
    return geo_cl30_rotor_normalize(out);
}

geo_so3_matrix_t geo_cl30_rotor_to_so3(geo_cl30_rotor_t r) {
    geo_so3_matrix_t m;
    geo_real_t s = r.s, b12 = r.b12, b23 = r.b23, b31 = r.b31;

    m.m[0][0] = (geo_real_t)1.0 - (geo_real_t)2.0 * (b31 * b31 + b12 * b12);
    m.m[0][1] = (geo_real_t)2.0 * (b23 * b31 + s * b12);
    m.m[0][2] = (geo_real_t)2.0 * (b23 * b12 - s * b31);

    m.m[1][0] = (geo_real_t)2.0 * (b23 * b31 - s * b12);
    m.m[1][1] = (geo_real_t)1.0 - (geo_real_t)2.0 * (b23 * b23 + b12 * b12);
    m.m[1][2] = (geo_real_t)2.0 * (b31 * b12 + s * b23);

    m.m[2][0] = (geo_real_t)2.0 * (b23 * b12 + s * b31);
    m.m[2][1] = (geo_real_t)2.0 * (b31 * b12 - s * b23);
    m.m[2][2] = (geo_real_t)1.0 - (geo_real_t)2.0 * (b23 * b23 + b31 * b31);

    return m;
}

geo_cl30_vec3_t geo_so3_apply_vec3(geo_so3_matrix_t m, geo_cl30_vec3_t v) {
    geo_cl30_vec3_t out;
    out.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z;
    out.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z;
    out.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z;
    return out;
}

geo_cl30_vec3_t geo_cl30_rotor_apply_vec3(geo_cl30_rotor_t r, geo_cl30_vec3_t v) {
    geo_so3_matrix_t m = geo_cl30_rotor_to_so3(r);
    return geo_so3_apply_vec3(m, v);
}

geo_real_t geo_cl30_rotor_norm_residual(geo_cl30_rotor_t r) {
    geo_real_t norm_sq = r.s * r.s + r.b12 * r.b12 + r.b23 * r.b23 + r.b31 * r.b31;
    return (geo_real_t)fabs((double)norm_sq - 1.0);
}

void cl30_rotor_normalize_batch(const float* bivectors, float* rotors, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t in_off = i * 3;
        int64_t out_off = i * 4;
        geo_cl30_rotor_t r = geo_cl30_rotor_exp(bivectors[in_off], bivectors[in_off + 1], bivectors[in_off + 2]);
        rotors[out_off] = r.s;
        rotors[out_off + 1] = r.b12;
        rotors[out_off + 2] = r.b23;
        rotors[out_off + 3] = r.b31;
    }
}

void cl30_rotor_compose_batch(const float* r1, const float* r2, float* out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t off = i * 4;
        geo_cl30_rotor_t a = { r1[off], r1[off + 1], r1[off + 2], r1[off + 3] };
        geo_cl30_rotor_t b = { r2[off], r2[off + 1], r2[off + 2], r2[off + 3] };
        geo_cl30_rotor_t res = geo_cl30_rotor_mul(a, b);
        out[off] = res.s;
        out[off + 1] = res.b12;
        out[off + 2] = res.b23;
        out[off + 3] = res.b31;
    }
}

void cl30_rotor_reverse_batch(const float* r, float* out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t off = i * 4;
        geo_cl30_rotor_t val = { r[off], r[off + 1], r[off + 2], r[off + 3] };
        geo_cl30_rotor_t rev = geo_cl30_rotor_reverse(val);
        out[off] = rev.s;
        out[off + 1] = rev.b12;
        out[off + 2] = rev.b23;
        out[off + 3] = rev.b31;
    }
}

void cl30_rotor_apply_batch(const float* rotors, const float* vectors, float* out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t r_off = i * 4;
        int64_t v_off = i * 3;
        geo_cl30_rotor_t r = { rotors[r_off], rotors[r_off + 1], rotors[r_off + 2], rotors[r_off + 3] };
        geo_cl30_vec3_t v = { vectors[v_off], vectors[v_off + 1], vectors[v_off + 2] };
        geo_cl30_vec3_t res = geo_cl30_rotor_apply_vec3(r, v);
        out[v_off] = res.x;
        out[v_off + 1] = res.y;
        out[v_off + 2] = res.z;
    }
}

void cl30_so3_matrix_apply_batch(const float* matrices, const float* vectors, float* out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t m_off = i * 9;
        int64_t v_off = i * 3;
        geo_so3_matrix_t m;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                m.m[r][c] = matrices[m_off + r * 3 + c];
            }
        }
        geo_cl30_vec3_t v = { vectors[v_off], vectors[v_off + 1], vectors[v_off + 2] };
        geo_cl30_vec3_t res = geo_so3_apply_vec3(m, v);
        out[v_off] = res.x;
        out[v_off + 1] = res.y;
        out[v_off + 2] = res.z;
    }
}
