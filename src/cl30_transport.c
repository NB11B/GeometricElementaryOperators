#include "geo/cl30_transport.h"
#include <math.h>
#include <stddef.h>

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

int geo_cl30_rotor_exp_f32(
    const float *bivectors,
    float *rotors,
    int64_t count)
{
    if (!bivectors || !rotors) return -1;
    if (count < 0) return -1;

    for (int64_t i = 0; i < count; ++i) {
        int64_t in_off = i * 3;
        int64_t out_off = i * 4;
        float b1 = bivectors[in_off];
        float b2 = bivectors[in_off + 1];
        float b3 = bivectors[in_off + 2];

        if (!isfinite(b1) || !isfinite(b2) || !isfinite(b3)) return -2;

        geo_cl30_rotor_t r = geo_cl30_rotor_exp(b1, b2, b3);
        rotors[out_off]     = r.s;
        rotors[out_off + 1] = r.b12;
        rotors[out_off + 2] = r.b23;
        rotors[out_off + 3] = r.b31;
    }
    return 0;
}

int geo_cl30_rotor_apply_f32(
    const float *rotors,
    const float *vectors,
    float *outputs,
    int64_t count)
{
    if (!rotors || !vectors || !outputs) return -1;
    if (count < 0) return -1;

    for (int64_t i = 0; i < count; ++i) {
        int64_t r_off = i * 4;
        int64_t v_off = i * 3;

        float s   = rotors[r_off];
        float b12 = rotors[r_off + 1];
        float b23 = rotors[r_off + 2];
        float b31 = rotors[r_off + 3];

        float vx = vectors[v_off];
        float vy = vectors[v_off + 1];
        float vz = vectors[v_off + 2];

        if (!isfinite(s) || !isfinite(b12) || !isfinite(b23) || !isfinite(b31) ||
            !isfinite(vx) || !isfinite(vy) || !isfinite(vz)) return -2;

        geo_cl30_rotor_t r = { s, b12, b23, b31 };
        geo_cl30_vec3_t v = { vx, vy, vz };
        geo_cl30_vec3_t res = geo_cl30_rotor_apply_vec3(r, v);

        outputs[v_off]     = res.x;
        outputs[v_off + 1] = res.y;
        outputs[v_off + 2] = res.z;
    }
    return 0;
}

int geo_cl30_rotor_apply_vjp_f32(
    const float *bivectors,
    const float *vectors,
    const float *grad_outputs,
    float *grad_bivectors,
    float *grad_vectors,
    int64_t count)
{
    if (!bivectors || !vectors || !grad_outputs || !grad_bivectors || !grad_vectors) return -1;
    if (count < 0) return -1;

    for (int64_t i = 0; i < count; ++i) {
        int64_t b_off = i * 3;
        int64_t v_off = i * 3;

        float b1 = bivectors[b_off];
        float b2 = bivectors[b_off + 1];
        float b3 = bivectors[b_off + 2];

        float vx = vectors[v_off];
        float vy = vectors[v_off + 1];
        float vz = vectors[v_off + 2];

        float gx = grad_outputs[v_off];
        float gy = grad_outputs[v_off + 1];
        float gz = grad_outputs[v_off + 2];

        if (!isfinite(b1) || !isfinite(b2) || !isfinite(b3) ||
            !isfinite(vx) || !isfinite(vy) || !isfinite(vz) ||
            !isfinite(gx) || !isfinite(gy) || !isfinite(gz)) return -2;

        // 1. Exponentiate to rotor
        float theta_sq = b1 * b1 + b2 * b2 + b3 * b3;
        float theta = (float)sqrt((double)theta_sq);
        float s, k;
        float ds_dtheta, dk_dtheta;

        if (theta < 1e-7f) {
            s = 1.0f - 0.125f * theta_sq;
            k = -0.5f + (1.0f / 48.0f) * theta_sq;
            ds_dtheta = -0.25f * theta;
            dk_dtheta = (1.0f / 24.0f) * theta;
        } else {
            float half_theta = 0.5f * theta;
            s = (float)cos((double)half_theta);
            k = -(float)sin((double)half_theta) / theta;
            ds_dtheta = -0.5f * (float)sin((double)half_theta);
            dk_dtheta = (-0.5f * (float)cos((double)half_theta) * theta + (float)sin((double)half_theta)) / (theta * theta);
        }

        float rb1 = k * b1;
        float rb2 = k * b2;
        float rb3 = k * b3;

        // 2. SO(3) matrix entries
        float m00 = 1.0f - 2.0f * (rb3 * rb3 + rb1 * rb1);
        float m01 = 2.0f * (rb2 * rb3 + s * rb1);
        float m02 = 2.0f * (rb2 * rb1 - s * rb3);

        float m10 = 2.0f * (rb2 * rb3 - s * rb1);
        float m11 = 1.0f - 2.0f * (rb2 * rb2 + rb1 * rb1);
        float m12 = 2.0f * (rb3 * rb1 + s * rb2);

        float m20 = 2.0f * (rb2 * rb1 + s * rb3);
        float m21 = 2.0f * (rb3 * rb1 - s * rb2);
        float m22 = 1.0f - 2.0f * (rb2 * rb2 + rb3 * rb3);

        // 3. Vector gradient: M^T * g
        grad_vectors[v_off]     = m00 * gx + m10 * gy + m20 * gz;
        grad_vectors[v_off + 1] = m01 * gx + m11 * gy + m21 * gz;
        grad_vectors[v_off + 2] = m02 * gx + m12 * gy + m22 * gz;

        // 4. Rotor gradients dL/ds, dL/drb1, dL/drb2, dL/drb3
        float dL_ds = 2.0f * (gx * vy * rb1 - gx * vz * rb3 - gy * vx * rb1 + gy * vz * rb2 + gz * vx * rb3 - gz * vy * rb2);
        float dL_drb1 = -4.0f * rb1 * (gx * vx + gy * vy) + 2.0f * s * (gx * vy - gy * vx) + 2.0f * rb2 * (gx * vz + gz * vx) + 2.0f * rb3 * (gy * vz + gz * vy);
        float dL_drb2 = -4.0f * rb2 * (gy * vy + gz * vz) + 2.0f * s * (gy * vz - gz * vy) + 2.0f * rb1 * (gx * vz + gz * vx) + 2.0f * rb3 * (gx * vy + gy * vx);
        float dL_drb3 = -4.0f * rb3 * (gx * vx + gz * vz) + 2.0f * s * (gz * vx - gx * vz) + 2.0f * rb1 * (gy * vz + gz * vy) + 2.0f * rb2 * (gx * vy + gy * vx);

        // 5. Bivector gradient dL/db_i
        float dL_dot_b = dL_drb1 * b1 + dL_drb2 * b2 + dL_drb3 * b3;

        if (theta < 1e-7f) {
            grad_bivectors[b_off]     = -0.25f * dL_ds * b1 + k * dL_drb1 + (1.0f / 24.0f) * dL_dot_b * b1;
            grad_bivectors[b_off + 1] = -0.25f * dL_ds * b2 + k * dL_drb2 + (1.0f / 24.0f) * dL_dot_b * b2;
            grad_bivectors[b_off + 2] = -0.25f * dL_ds * b3 + k * dL_drb3 + (1.0f / 24.0f) * dL_dot_b * b3;
        } else {
            float inv_theta = 1.0f / theta;
            float term_s = dL_ds * ds_dtheta * inv_theta;
            float term_k = dL_dot_b * dk_dtheta * inv_theta;

            grad_bivectors[b_off]     = term_s * b1 + k * dL_drb1 + term_k * b1;
            grad_bivectors[b_off + 1] = term_s * b2 + k * dL_drb2 + term_k * b2;
            grad_bivectors[b_off + 2] = term_s * b3 + k * dL_drb3 + term_k * b3;
        }
    }
    return 0;
}

int geo_cl30_rotor_certificate_f32(
    const float *rotors,
    float *max_residual,
    int64_t count)
{
    if (!rotors || !max_residual) return -1;
    if (count < 0) return -1;

    float max_res = 0.0f;
    for (int64_t i = 0; i < count; ++i) {
        int64_t off = i * 4;
        float s   = rotors[off];
        float b12 = rotors[off + 1];
        float b23 = rotors[off + 2];
        float b31 = rotors[off + 3];

        if (!isfinite(s) || !isfinite(b12) || !isfinite(b23) || !isfinite(b31)) return -2;

        float norm_sq = s * s + b12 * b12 + b23 * b23 + b31 * b31;
        float res = (float)fabs((double)norm_sq - 1.0);
        if (res > max_res) max_res = res;
    }
    *max_residual = max_res;
    if (max_res > 1e-6f) return 1; // Certificate failure
    return 0;
}

void cl30_rotor_normalize_batch(const float* bivectors, float* rotors, int64_t count) {
    geo_cl30_rotor_exp_f32(bivectors, rotors, count);
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
    geo_cl30_rotor_apply_f32(rotors, vectors, out, count);
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
