#include "geo/cl30_incidence.h"
#include <math.h>
#include <string.h>

#define GEO_CL30_CERTIFICATE_THRESHOLD 1e-6f

static inline int is_finite_f(float v) {
    return isfinite(v);
}

typedef struct {
    float s;
    float b12;
    float b23;
    float b31;
} rotor4_t;

static inline rotor4_t rotor_exp(float b1, float b2, float b3) {
    float theta_sq = b1 * b1 + b2 * b2 + b3 * b3;
    float theta = (float)sqrt((double)theta_sq);
    rotor4_t r;

    if (theta < 1e-7f) {
        r.s = 1.0f - 0.125f * theta_sq;
        float k = -0.5f + (1.0f / 48.0f) * theta_sq;
        r.b12 = k * b1;
        r.b23 = k * b2;
        r.b31 = k * b3;
    } else {
        float half_theta = 0.5f * theta;
        r.s = (float)cos((double)half_theta);
        float k = -(float)sin((double)half_theta) / theta;
        r.b12 = k * b1;
        r.b23 = k * b2;
        r.b31 = k * b3;
    }

    float norm_sq = r.s * r.s + r.b12 * r.b12 + r.b23 * r.b23 + r.b31 * r.b31;
    if (norm_sq > 0.0f) {
        float inv_norm = 1.0f / (float)sqrt((double)norm_sq);
        r.s *= inv_norm;
        r.b12 *= inv_norm;
        r.b23 *= inv_norm;
        r.b31 *= inv_norm;
    } else {
        r.s = 1.0f;
        r.b12 = 0.0f;
        r.b23 = 0.0f;
        r.b31 = 0.0f;
    }
    return r;
}

static inline void rotor_to_so3(rotor4_t r, float m[3][3]) {
    float s = r.s, b12 = r.b12, b23 = r.b23, b31 = r.b31;

    m[0][0] = 1.0f - 2.0f * (b31 * b31 + b12 * b12);
    m[0][1] = 2.0f * (b23 * b31 + s * b12);
    m[0][2] = 2.0f * (b23 * b12 - s * b31);

    m[1][0] = 2.0f * (b23 * b31 - s * b12);
    m[1][1] = 1.0f - 2.0f * (b23 * b23 + b12 * b12);
    m[1][2] = 2.0f * (b31 * b12 + s * b23);

    m[2][0] = 2.0f * (b23 * b12 + s * b31);
    m[2][1] = 2.0f * (b31 * b12 - s * b23);
    m[2][2] = 1.0f - 2.0f * (b23 * b23 + b31 * b31);
}

static inline void apply_so3(const float m[3][3], const float v[3], float out[3]) {
    out[0] = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
    out[1] = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2];
    out[2] = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2];
}

size_t geo_cl30_incidence_workspace_bytes(const geo_cl30_incidence_shape* shape) {
    if (!shape) return 0;
    if (shape->relation_count <= 0 || shape->channel_count <= 0) return 0;
    return (size_t)shape->relation_count * (size_t)shape->channel_count * sizeof(rotor4_t);
}

geo_cl30_incidence_status geo_cl30_incidence_forward_f32(
    const geo_cl30_incidence_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    const float* inverse_degrees,
    float* node_aggregates,
    void* workspace,
    size_t workspace_bytes,
    float* max_rotor_residual
) {
    if (!shape || !relation_bivectors || !node_states || !inverse_degrees || !node_aggregates) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }
    if (shape->batch_count <= 0 || shape->node_count <= 0 || shape->relation_count <= 0 || shape->channel_count <= 0) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }
    if (shape->edge_count < 0) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }
    if (shape->edge_count > 0 && (!edge_batch || !edge_source || !edge_relation || !edge_destination)) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }

    size_t needed_workspace = geo_cl30_incidence_workspace_bytes(shape);
    if (workspace_bytes < needed_workspace || !workspace) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }

    int32_t B = shape->batch_count;
    int32_t N = shape->node_count;
    int32_t E = shape->edge_count;
    int32_t R = shape->relation_count;
    int32_t C = shape->channel_count;

    rotor4_t* cached_rotors = (rotor4_t*)workspace;
    float max_residual = 0.0f;

    // 1. Exponentiate & validate all relation rotors
    for (int32_t r = 0; r < R; ++r) {
        for (int32_t c = 0; c < C; ++c) {
            int64_t b_idx = ((int64_t)r * C + c) * 3;
            float b1 = relation_bivectors[b_idx];
            float b2 = relation_bivectors[b_idx + 1];
            float b3 = relation_bivectors[b_idx + 2];

            if (!is_finite_f(b1) || !is_finite_f(b2) || !is_finite_f(b3)) {
                return GEO_CL30_INCIDENCE_NONFINITE;
            }

            rotor4_t rot = rotor_exp(b1, b2, b3);
            float norm_sq = rot.s * rot.s + rot.b12 * rot.b12 + rot.b23 * rot.b23 + rot.b31 * rot.b31;
            float res = (float)fabs((double)norm_sq - 1.0);
            if (res > max_residual) {
                max_residual = res;
            }
            if (res > GEO_CL30_CERTIFICATE_THRESHOLD) {
                return GEO_CL30_INCIDENCE_CERTIFICATE_FAILURE;
            }
            cached_rotors[r * C + c] = rot;
        }
    }

    if (max_rotor_residual) {
        *max_rotor_residual = max_residual;
    }

    // 2. Zero node aggregates: shape [B, N, C, 3]
    size_t total_out_floats = (size_t)B * (size_t)N * (size_t)C * 3;
    memset(node_aggregates, 0, total_out_floats * sizeof(float));

    // 3. Accumulate edge messages
    for (int32_t e = 0; e < E; ++e) {
        int32_t b = edge_batch[e];
        int32_t u = edge_source[e];
        int32_t r = edge_relation[e];
        int32_t v = edge_destination[e];

        if (b < 0 || b >= B || u < 0 || u >= N || r < 0 || r >= R || v < 0 || v >= N) {
            return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
        }

        for (int32_t c = 0; c < C; ++c) {
            int64_t src_idx = (((int64_t)b * N + u) * C + c) * 3;
            float vx = node_states[src_idx];
            float vy = node_states[src_idx + 1];
            float vz = node_states[src_idx + 2];

            if (!is_finite_f(vx) || !is_finite_f(vy) || !is_finite_f(vz)) {
                return GEO_CL30_INCIDENCE_NONFINITE;
            }

            rotor4_t rot = cached_rotors[r * C + c];
            float m[3][3];
            rotor_to_so3(rot, m);

            float vec_in[3] = { vx, vy, vz };
            float vec_out[3];
            apply_so3(m, vec_in, vec_out);

            if (!is_finite_f(vec_out[0]) || !is_finite_f(vec_out[1]) || !is_finite_f(vec_out[2])) {
                return GEO_CL30_INCIDENCE_NONFINITE;
            }

            int64_t dst_idx = (((int64_t)b * N + v) * C + c) * 3;
            node_aggregates[dst_idx]     += vec_out[0];
            node_aggregates[dst_idx + 1] += vec_out[1];
            node_aggregates[dst_idx + 2] += vec_out[2];
        }
    }

    // 4. In-degree normalization: node_aggregates[b, v, c, k] *= inverse_degrees[b, v]
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t v = 0; v < N; ++v) {
            float inv_deg = inverse_degrees[b * N + v];
            if (!is_finite_f(inv_deg)) {
                return GEO_CL30_INCIDENCE_NONFINITE;
            }
            for (int32_t c = 0; c < C; ++c) {
                int64_t dst_idx = (((int64_t)b * N + v) * C + c) * 3;
                node_aggregates[dst_idx]     *= inv_deg;
                node_aggregates[dst_idx + 1] *= inv_deg;
                node_aggregates[dst_idx + 2] *= inv_deg;
            }
        }
    }

    return GEO_CL30_INCIDENCE_OK;
}

geo_cl30_incidence_status geo_cl30_incidence_vjp_f32(
    const geo_cl30_incidence_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    const float* inverse_degrees,
    const float* grad_node_aggregates,
    float* grad_relation_bivectors,
    float* grad_node_states,
    void* workspace,
    size_t workspace_bytes
) {
    if (!shape || !relation_bivectors || !node_states || !inverse_degrees ||
        !grad_node_aggregates || !grad_relation_bivectors || !grad_node_states) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }
    if (shape->batch_count <= 0 || shape->node_count <= 0 || shape->relation_count <= 0 || shape->channel_count <= 0) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }
    if (shape->edge_count < 0) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }
    if (shape->edge_count > 0 && (!edge_batch || !edge_source || !edge_relation || !edge_destination)) {
        return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
    }

    int32_t B = shape->batch_count;
    int32_t N = shape->node_count;
    int32_t E = shape->edge_count;
    int32_t R = shape->relation_count;
    int32_t C = shape->channel_count;

    (void)workspace;
    (void)workspace_bytes;

    // Zero gradient accumulators
    memset(grad_relation_bivectors, 0, (size_t)R * (size_t)C * 3 * sizeof(float));
    memset(grad_node_states, 0, (size_t)B * (size_t)N * (size_t)C * 3 * sizeof(float));

    // Accumulate VJPs across edges
    for (int32_t e = 0; e < E; ++e) {
        int32_t b = edge_batch[e];
        int32_t u = edge_source[e];
        int32_t r = edge_relation[e];
        int32_t v = edge_destination[e];

        if (b < 0 || b >= B || u < 0 || u >= N || r < 0 || r >= R || v < 0 || v >= N) {
            return GEO_CL30_INCIDENCE_INVALID_ARGUMENT;
        }

        float inv_deg = inverse_degrees[b * N + v];
        if (!is_finite_f(inv_deg)) {
            return GEO_CL30_INCIDENCE_NONFINITE;
        }

        for (int32_t c = 0; c < C; ++c) {
            int64_t b_idx = ((int64_t)r * C + c) * 3;
            float b1 = relation_bivectors[b_idx];
            float b2 = relation_bivectors[b_idx + 1];
            float b3 = relation_bivectors[b_idx + 2];

            int64_t src_idx = (((int64_t)b * N + u) * C + c) * 3;
            float vx = node_states[src_idx];
            float vy = node_states[src_idx + 1];
            float vz = node_states[src_idx + 2];

            int64_t dst_idx = (((int64_t)b * N + v) * C + c) * 3;
            float gx = grad_node_aggregates[dst_idx]     * inv_deg;
            float gy = grad_node_aggregates[dst_idx + 1] * inv_deg;
            float gz = grad_node_aggregates[dst_idx + 2] * inv_deg;

            if (!is_finite_f(b1) || !is_finite_f(b2) || !is_finite_f(b3) ||
                !is_finite_f(vx) || !is_finite_f(vy) || !is_finite_f(vz) ||
                !is_finite_f(gx) || !is_finite_f(gy) || !is_finite_f(gz)) {
                return GEO_CL30_INCIDENCE_NONFINITE;
            }

            // Exponentiate & compute derivatives
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

            // SO(3) matrix
            float m00 = 1.0f - 2.0f * (rb3 * rb3 + rb1 * rb1);
            float m01 = 2.0f * (rb2 * rb3 + s * rb1);
            float m02 = 2.0f * (rb2 * rb1 - s * rb3);

            float m10 = 2.0f * (rb2 * rb3 - s * rb1);
            float m11 = 1.0f - 2.0f * (rb2 * rb2 + rb1 * rb1);
            float m12 = 2.0f * (rb3 * rb1 + s * rb2);

            float m20 = 2.0f * (rb2 * rb1 + s * rb3);
            float m21 = 2.0f * (rb3 * rb1 - s * rb2);
            float m22 = 1.0f - 2.0f * (rb2 * rb2 + rb3 * rb3);

            // Vector gradient: M^T * g
            float g_vx = m00 * gx + m10 * gy + m20 * gz;
            float g_vy = m01 * gx + m11 * gy + m21 * gz;
            float g_vz = m02 * gx + m12 * gy + m22 * gz;

            grad_node_states[src_idx]     += g_vx;
            grad_node_states[src_idx + 1] += g_vy;
            grad_node_states[src_idx + 2] += g_vz;

            // Rotor gradients dL/ds, dL/drb1, dL/drb2, dL/drb3
            float dL_ds   = 2.0f * (gx * vy * rb1 - gx * vz * rb3 - gy * vx * rb1 + gy * vz * rb2 + gz * vx * rb3 - gz * vy * rb2);
            float dL_drb1 = -4.0f * rb1 * (gx * vx + gy * vy) + 2.0f * s * (gx * vy - gy * vx) + 2.0f * rb2 * (gx * vz + gz * vx) + 2.0f * rb3 * (gy * vz + gz * vy);
            float dL_drb2 = -4.0f * rb2 * (gy * vy + gz * vz) + 2.0f * s * (gy * vz - gz * vy) + 2.0f * rb1 * (gx * vz + gz * vx) + 2.0f * rb3 * (gx * vy + gy * vx);
            float dL_drb3 = -4.0f * rb3 * (gx * vx + gz * vz) + 2.0f * s * (gz * vx - gx * vz) + 2.0f * rb1 * (gy * vz + gz * vy) + 2.0f * rb2 * (gx * vy + gy * vx);

            // Bivector gradient dL/db_i
            float dL_dot_b = dL_drb1 * b1 + dL_drb2 * b2 + dL_drb3 * b3;
            float gb1, gb2, gb3;

            if (theta < 1e-7f) {
                gb1 = -0.25f * dL_ds * b1 + k * dL_drb1 + (1.0f / 24.0f) * dL_dot_b * b1;
                gb2 = -0.25f * dL_ds * b2 + k * dL_drb2 + (1.0f / 24.0f) * dL_dot_b * b2;
                gb3 = -0.25f * dL_ds * b3 + k * dL_drb3 + (1.0f / 24.0f) * dL_dot_b * b3;
            } else {
                float inv_theta = 1.0f / theta;
                float term_s = dL_ds * ds_dtheta * inv_theta;
                float term_k = dL_dot_b * dk_dtheta * inv_theta;

                gb1 = term_s * b1 + k * dL_drb1 + term_k * b1;
                gb2 = term_s * b2 + k * dL_drb2 + term_k * b2;
                gb3 = term_s * b3 + k * dL_drb3 + term_k * b3;
            }

            grad_relation_bivectors[b_idx]     += gb1;
            grad_relation_bivectors[b_idx + 1] += gb2;
            grad_relation_bivectors[b_idx + 2] += gb3;
        }
    }

    return GEO_CL30_INCIDENCE_OK;
}
