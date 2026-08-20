/**
 * @file cl30_karcher.c
 * @brief Implementation of Cl(3,0) / Spin(3) Rotor and Riemannian Karcher Mean Engine.
 */

#include "geo/cl30_karcher.h"
#include <math.h>

#define EPSILON 1e-8f
#define PI 3.14159265358979323846f

geo_rotor3_t geo_rotor3_from_bivector(float b0, float b1, float b2) {
    float norm_sq = b0 * b0 + b1 * b1 + b2 * b2;
    float norm = sqrtf(norm_sq);
    geo_rotor3_t r;
    if (norm < EPSILON) {
        r.s = 1.0f;
        r.b0 = 0.0f;
        r.b1 = 0.0f;
        r.b2 = 0.0f;
        return r;
    }
    float half_angle = 0.5f * norm;
    float sin_val = sinf(half_angle);
    r.s = cosf(half_angle);
    r.b0 = -(b0 / norm) * sin_val;
    r.b1 = -(b1 / norm) * sin_val;
    r.b2 = -(b2 / norm) * sin_val;
    return r;
}

geo_rotor3_t geo_rotor3_multiply(geo_rotor3_t r1, geo_rotor3_t r2) {
    // Quaternion multiplication isomorphism:
    // (s1 + v1)(s2 + v2) = (s1*s2 - v1.v2) + (s1*v2 + s2*v1 + v1 x v2)
    geo_rotor3_t r;
    r.s = r1.s * r2.s - r1.b0 * r2.b0 - r1.b1 * r2.b1 - r1.b2 * r2.b2;
    r.b0 = r1.s * r2.b0 + r1.b0 * r2.s + (r1.b1 * r2.b2 - r1.b2 * r2.b1);
    r.b1 = r1.s * r2.b1 + r1.b1 * r2.s + (r1.b2 * r2.b0 - r1.b0 * r2.b2);
    r.b2 = r1.s * r2.b2 + r1.b2 * r2.s + (r1.b0 * r2.b1 - r1.b1 * r2.b0);
    return r;
}

geo_rotor3_t geo_rotor3_reverse(geo_rotor3_t r) {
    geo_rotor3_t res;
    res.s = r.s;
    res.b0 = -r.b0;
    res.b1 = -r.b1;
    res.b2 = -r.b2;
    return res;
}

geo_rotor3_t geo_rotor3_normalize(geo_rotor3_t r) {
    float norm_sq = r.s * r.s + r.b0 * r.b0 + r.b1 * r.b1 + r.b2 * r.b2;
    float norm = sqrtf(norm_sq);
    if (norm < EPSILON) {
        geo_rotor3_t id = {1.0f, 0.0f, 0.0f, 0.0f};
        return id;
    }
    float inv = 1.0f / norm;
    geo_rotor3_t res;
    res.s = r.s * inv;
    res.b0 = r.b0 * inv;
    res.b1 = r.b1 * inv;
    res.b2 = r.b2 * inv;
    return res;
}

float geo_rotor3_dot(geo_rotor3_t r1, geo_rotor3_t r2) {
    return r1.s * r2.s + r1.b0 * r2.b0 + r1.b1 * r2.b1 + r1.b2 * r2.b2;
}

geo_vec3_t geo_rotor3_log(geo_rotor3_t r) {
    // r = cos(theta) + u * sin(theta) -> Log(R) = theta * u
    geo_rotor3_t u_r = geo_rotor3_normalize(r);
    float cos_theta = u_r.s;
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;

    float vec_norm = sqrtf(u_r.b0 * u_r.b0 + u_r.b1 * u_r.b1 + u_r.b2 * u_r.b2);
    geo_vec3_t v;
    if (vec_norm < EPSILON) {
        v.x = 0.0f;
        v.y = 0.0f;
        v.z = 0.0f;
        return v;
    }
    float theta = acosf(cos_theta);
    float scale = theta / vec_norm;
    v.x = u_r.b0 * scale;
    v.y = u_r.b1 * scale;
    v.z = u_r.b2 * scale;
    return v;
}

geo_rotor3_t geo_rotor3_exp(geo_vec3_t v) {
    float theta = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    geo_rotor3_t r;
    if (theta < EPSILON) {
        r.s = 1.0f;
        r.b0 = 0.0f;
        r.b1 = 0.0f;
        r.b2 = 0.0f;
        return r;
    }
    float sin_theta = sinf(theta);
    r.s = cosf(theta);
    r.b0 = (v.x / theta) * sin_theta;
    r.b1 = (v.y / theta) * sin_theta;
    r.b2 = (v.z / theta) * sin_theta;
    return r;
}

geo_vec3_t geo_vec3_slerp(geo_vec3_t v0, geo_vec3_t v1, float t) {
    float dot = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    if (dot > 0.9999f) {
        // Nearly collinear
        geo_vec3_t res;
        res.x = (1.0f - t) * v0.x + t * v1.x;
        res.y = (1.0f - t) * v0.y + t * v1.y;
        res.z = (1.0f - t) * v0.z + t * v1.z;
        float norm = sqrtf(res.x * res.x + res.y * res.y + res.z * res.z);
        if (norm > EPSILON) {
            res.x /= norm; res.y /= norm; res.z /= norm;
        }
        return res;
    }
    if (dot < -0.9999f) {
        dot = -0.9999f;
    }
    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    float s0 = sinf((1.0f - t) * theta) / sin_theta;
    float s1 = sinf(t * theta) / sin_theta;
    geo_vec3_t res;
    res.x = s0 * v0.x + s1 * v1.x;
    res.y = s0 * v0.y + s1 * v1.y;
    res.z = s0 * v0.z + s1 * v1.z;
    return res;
}

geo_rotor3_t geo_rotor3_slerp(geo_rotor3_t r0, geo_rotor3_t r1, float t) {
    float dot = geo_rotor3_dot(r0, r1);
    geo_rotor3_t r1_aligned = r1;
    if (dot < 0.0f) {
        // Antipodal shortest arc
        r1_aligned.s = -r1.s;
        r1_aligned.b0 = -r1.b0;
        r1_aligned.b1 = -r1.b1;
        r1_aligned.b2 = -r1.b2;
        dot = -dot;
    }
    if (dot > 0.9999f) {
        geo_rotor3_t res;
        res.s = (1.0f - t) * r0.s + t * r1_aligned.s;
        res.b0 = (1.0f - t) * r0.b0 + t * r1_aligned.b0;
        res.b1 = (1.0f - t) * r0.b1 + t * r1_aligned.b1;
        res.b2 = (1.0f - t) * r0.b2 + t * r1_aligned.b2;
        return geo_rotor3_normalize(res);
    }
    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    float s0 = sinf((1.0f - t) * theta) / sin_theta;
    float s1 = sinf(t * theta) / sin_theta;
    geo_rotor3_t res;
    res.s = s0 * r0.s + s1 * r1_aligned.s;
    res.b0 = s0 * r0.b0 + s1 * r1_aligned.b0;
    res.b1 = s0 * r0.b1 + s1 * r1_aligned.b1;
    res.b2 = s0 * r0.b2 + s1 * r1_aligned.b2;
    return geo_rotor3_normalize(res);
}

geo_vec3_t geo_rotor3_rotate_vec(geo_rotor3_t r, geo_vec3_t v) {
    // Standard quaternion rotation: v' = v + 2*s*(q_v x v) + 2*(q_v x (q_v x v))
    // where q_v = (b0, b1, b2).
    float qx = r.b0, qy = r.b1, qz = r.b2;
    float s = r.s;

    // t = 2 * (q_v x v)
    float tx = 2.0f * (qy * v.z - qz * v.y);
    float ty = 2.0f * (qz * v.x - qx * v.z);
    float tz = 2.0f * (qx * v.y - qy * v.x);

    // v' = v + s * t + (q_v x t)
    geo_vec3_t v_prime;
    v_prime.x = v.x + s * tx + (qy * tz - qz * ty);
    v_prime.y = v.y + s * ty + (qz * tx - qx * tz);
    v_prime.z = v.z + s * tz + (qx * ty - qy * tx);
    return v_prime;
}

geo_rotor3_t geo_karcher_mean_spin3(
    const geo_rotor3_t* rotors,
    const float* weights,
    size_t count,
    geo_rotor3_t init_rotor
) {
    if (count == 0 || rotors == NULL || weights == NULL) {
        return geo_rotor3_normalize(init_rotor);
    }

    geo_rotor3_t G = geo_rotor3_normalize(init_rotor);

    // Exactly 4 iterations
    for (int k = 0; k < 4; ++k) {
        geo_rotor3_t G_rev = geo_rotor3_reverse(G);
        geo_vec3_t delta_sum = {0.0f, 0.0f, 0.0f};

        for (size_t e = 0; e < count; ++e) {
            float w_e = weights[e];
            if (w_e <= 0.0f) continue;

            geo_rotor3_t G_e = rotors[e];
            // Antipodal alignment
            if (geo_rotor3_dot(G, G_e) < 0.0f) {
                G_e.s = -G_e.s;
                G_e.b0 = -G_e.b0;
                G_e.b1 = -G_e.b1;
                G_e.b2 = -G_e.b2;
            }

            // Relative rotor: G_rel = G_rev * G_e
            geo_rotor3_t G_rel = geo_rotor3_multiply(G_rev, G_e);
            geo_vec3_t log_rel = geo_rotor3_log(G_rel);

            delta_sum.x += w_e * log_rel.x;
            delta_sum.y += w_e * log_rel.y;
            delta_sum.z += w_e * log_rel.z;
        }

        // Update G^[k+1] = G^[k] * Exp(delta^[k])
        geo_rotor3_t step = geo_rotor3_exp(delta_sum);
        G = geo_rotor3_normalize(geo_rotor3_multiply(G, step));
    }

    return G;
}
