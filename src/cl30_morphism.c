#include "geo/cl30_morphism.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPS_NORM 1e-8f
#define EPS_SQ 1e-14f

static inline float geo_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return (float)exp((double)x);
    return (float)log1p(exp((double)x));
}

static inline float geo_sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + (float)exp(-(double)x));
}

static inline float geo_vec3_dot(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline void geo_vec3_cross(const float* a, const float* b, float* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static inline float geo_vec3_normalize(const float* in, float* out) {
    float norm_sq = in[0] * in[0] + in[1] * in[1] + in[2] * in[2];
    float denom = (float)sqrt((double)(norm_sq + EPS_NORM));
    float inv = 1.0f / denom;
    out[0] = in[0] * inv;
    out[1] = in[1] * inv;
    out[2] = in[2] * inv;
    return denom;
}

static inline void geo_bivector_to_so3(const float* b, float* m, float* out_res) {
    float b1 = b[0], b2 = b[1], b3 = b[2];
    float theta_sq = b1 * b1 + b2 * b2 + b3 * b3;
    float theta = (float)sqrt((double)theta_sq);
    float s, rb1, rb2, rb3;

    if (theta < 1e-7f) {
        s = 1.0f - 0.125f * theta_sq;
        float k = -0.5f + (1.0f / 48.0f) * theta_sq;
        rb1 = k * b1;
        rb2 = k * b2;
        rb3 = k * b3;
    } else {
        float half_theta = 0.5f * theta;
        s = (float)cos((double)half_theta);
        float k = -(float)sin((double)half_theta) / theta;
        rb1 = k * b1;
        rb2 = k * b2;
        rb3 = k * b3;
    }

    float norm_sq = s * s + rb1 * rb1 + rb2 * rb2 + rb3 * rb3;
    float res = (float)fabs((double)(norm_sq - 1.0f));
    if (out_res && res > *out_res) *out_res = res;

    if (norm_sq > 0.0f) {
        float inv_norm = 1.0f / (float)sqrt((double)norm_sq);
        s *= inv_norm;
        rb1 *= inv_norm;
        rb2 *= inv_norm;
        rb3 *= inv_norm;
    }

    m[0] = 1.0f - 2.0f * (rb3 * rb3 + rb1 * rb1);
    m[1] = 2.0f * (rb2 * rb3 + s * rb1);
    m[2] = 2.0f * (rb2 * rb1 - s * rb3);

    m[3] = 2.0f * (rb2 * rb3 - s * rb1);
    m[4] = 1.0f - 2.0f * (rb2 * rb2 + rb1 * rb1);
    m[5] = 2.0f * (rb3 * rb1 + s * rb2);

    m[6] = 2.0f * (rb2 * rb1 + s * rb3);
    m[7] = 2.0f * (rb3 * rb1 - s * rb2);
    m[8] = 1.0f - 2.0f * (rb2 * rb2 + rb3 * rb3);
}

size_t geo_cl30_morphism_workspace_bytes(const geo_cl30_morphism_shape* shape) {
    if (!shape) return 0;
    size_t R = (size_t)shape->relation_count;
    size_t C = (size_t)shape->channel_count;
    size_t E = (size_t)shape->edge_count;
    size_t B = (size_t)shape->batch_count;
    size_t N = (size_t)shape->node_count;

    size_t so3_bytes = R * C * 9 * sizeof(float);
    size_t alpha_bytes = E * C * sizeof(float);
    size_t beta_bytes = E * C * sizeof(float);
    size_t weight_bytes = B * N * C * sizeof(float);
    size_t msg_bytes = B * N * C * 3 * sizeof(float);
    size_t t_bytes = E * C * 3 * sizeof(float);

    return so3_bytes + alpha_bytes + beta_bytes + weight_bytes + msg_bytes + t_bytes + 1024;
}

geo_cl30_morphism_status geo_cl30_morphism_forward_f32(
    const geo_cl30_morphism_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const float* goal_queries,
    const float* applicability_params,
    const float* compatibility_params,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    int32_t lesion_mode,
    float* node_aggregates,
    float* edge_alphas,
    float* edge_betas,
    float* gate_denominators,
    void* workspace,
    size_t workspace_bytes,
    geo_cl30_morphism_telemetry* telemetry
) {
    if (!shape || !relation_bivectors || !node_states || !goal_queries ||
        !applicability_params || !compatibility_params ||
        !edge_batch || !edge_source || !edge_relation || !edge_destination ||
        !node_aggregates || !workspace) {
        return GEO_CL30_MORPHISM_INVALID_ARGUMENT;
    }

    int32_t B = shape->batch_count;
    int32_t N = shape->node_count;
    int32_t E = shape->edge_count;
    int32_t R = shape->relation_count;
    int32_t C = shape->channel_count;

    size_t req_bytes = geo_cl30_morphism_workspace_bytes(shape);
    if (workspace_bytes < req_bytes) return GEO_CL30_MORPHISM_INVALID_ARGUMENT;

    float* so3_matrices = (float*)workspace;
    float* local_alphas = so3_matrices + (R * C * 9);
    float* local_betas = local_alphas + (E * C);
    float* sum_weights = local_betas + (E * C);
    float* sum_msgs = sum_weights + (B * N * C);

    float w_a = applicability_params[0];
    float w_ar = applicability_params[1];
    float b_a = applicability_params[2];

    float w_s = compatibility_params[0];
    float theta_b = compatibility_params[1];
    float w_b = geo_softplus(theta_b);
    float w_q = compatibility_params[2];
    float w_tau = compatibility_params[3];
    float b_beta = compatibility_params[4];

    float max_res = 0.0f;
    for (int32_t r = 0; r < R; ++r) {
        for (int32_t c = 0; c < C; ++c) {
            const float* b_ptr = relation_bivectors + (r * C + c) * 3;
            float b_eff[3] = { b_ptr[0], b_ptr[1], b_ptr[2] };
            if (lesion_mode == 3) {
                /* Commutative lesion: project onto e12 plane */
                b_eff[1] = 0.0f;
                b_eff[2] = 0.0f;
            } else if (lesion_mode == 4) {
                /* Identity lesion */
                b_eff[0] = 0.0f;
                b_eff[1] = 0.0f;
                b_eff[2] = 0.0f;
            }
            float* m_ptr = so3_matrices + (r * C + c) * 9;
            geo_bivector_to_so3(b_eff, m_ptr, &max_res);
        }
    }

    if (max_res > 1e-5f) return GEO_CL30_MORPHISM_CERTIFICATE_FAILURE;

    memset(sum_weights, 0, B * N * C * sizeof(float));
    memset(sum_msgs, 0, B * N * C * 3 * sizeof(float));

    float alpha_sum = 0.0f, alpha_sq_sum = 0.0f, alpha_min = 1.0f, alpha_max = 0.0f;
    float beta_sum = 0.0f, beta_sq_sum = 0.0f, beta_min = 1.0f, beta_max = 0.0f;
    float prod_sum = 0.0f, prod_sq_sum = 0.0f;
    int32_t a_sat_low = 0, a_sat_high = 0, b_sat_low = 0, b_sat_high = 0;
    int32_t total_evals = E * C;

    for (int32_t e = 0; e < E; ++e) {
        int32_t b_idx = edge_batch[e];
        int32_t u_idx = edge_source[e];
        int32_t r_idx = edge_relation[e];
        int32_t v_idx = edge_destination[e];

        for (int32_t c = 0; c < C; ++c) {
            const float* h_u = node_states + ((b_idx * N + u_idx) * C + c) * 3;
            const float* h_v = node_states + ((b_idx * N + v_idx) * C + c) * 3;
            const float* q_vec = goal_queries + (b_idx * C + c) * 3;
            const float* m_rot = so3_matrices + (r_idx * C + c) * 9;

            /* 1. Rotor transport: t = M_r * h_u */
            float t_vec[3];
            t_vec[0] = m_rot[0] * h_u[0] + m_rot[1] * h_u[1] + m_rot[2] * h_u[2];
            t_vec[1] = m_rot[3] * h_u[0] + m_rot[4] * h_u[1] + m_rot[5] * h_u[2];
            t_vec[2] = m_rot[6] * h_u[0] + m_rot[7] * h_u[1] + m_rot[8] * h_u[2];

            /* 2. Normalizations */
            float h_u_hat[3], t_hat[3], h_v_hat[3], q_hat[3];
            geo_vec3_normalize(h_u, h_u_hat);
            geo_vec3_normalize(t_vec, t_hat);
            geo_vec3_normalize(h_v, h_v_hat);
            geo_vec3_normalize(q_vec, q_hat);

            /* 3. Applicability */
            float alpha = 1.0f;
            if (lesion_mode != 1) {
                float dot_hu_q = geo_vec3_dot(h_u_hat, q_hat);
                float dot_t_q = geo_vec3_dot(t_hat, q_hat);
                float raw_alpha = w_a * dot_hu_q + w_ar * dot_t_q + b_a;
                alpha = geo_sigmoid(raw_alpha);
            }

            /* 4. Compatibility */
            float beta = 1.0f;
            if (lesion_mode != 2) {
                float dot_hv_t = geo_vec3_dot(h_v_hat, t_hat);
                float cross_hv_t[3];
                geo_vec3_cross(h_v_hat, t_hat, cross_hv_t);
                float cross_sq = geo_vec3_dot(cross_hv_t, cross_hv_t);
                float bivector_norm = (float)sqrt((double)(cross_sq + 1e-12f));
                float dot_q_t = geo_vec3_dot(q_hat, t_hat);
                float pseudoscalar_dual = geo_vec3_dot(cross_hv_t, q_hat);

                float raw_beta = w_s * dot_hv_t - w_b * bivector_norm + w_q * dot_q_t + w_tau * pseudoscalar_dual + b_beta;
                beta = geo_sigmoid(raw_beta);
            }

            local_alphas[e * C + c] = alpha;
            local_betas[e * C + c] = beta;
            if (edge_alphas) edge_alphas[e * C + c] = alpha;
            if (edge_betas) edge_betas[e * C + c] = beta;

            float weight = (lesion_mode == 5) ? 0.0f : (alpha * beta);
            sum_weights[(b_idx * N + v_idx) * C + c] += weight;

            float* dst_msg = sum_msgs + ((b_idx * N + v_idx) * C + c) * 3;
            dst_msg[0] += weight * t_vec[0];
            dst_msg[1] += weight * t_vec[1];
            dst_msg[2] += weight * t_vec[2];

            /* Telemetry accumulation */
            alpha_sum += alpha;
            alpha_sq_sum += alpha * alpha;
            if (alpha < alpha_min) alpha_min = alpha;
            if (alpha > alpha_max) alpha_max = alpha;
            if (alpha <= 0.01f) a_sat_low++;
            if (alpha >= 0.99f) a_sat_high++;

            beta_sum += beta;
            beta_sq_sum += beta * beta;
            if (beta < beta_min) beta_min = beta;
            if (beta > beta_max) beta_max = beta;
            if (beta <= 0.01f) b_sat_low++;
            if (beta >= 0.99f) b_sat_high++;

            prod_sum += weight;
            prod_sq_sum += weight * weight;
        }
    }

    float denom_min = 1e9f, denom_max = 0.0f;
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t n = 0; n < N; ++n) {
            for (int32_t c = 0; c < C; ++c) {
                float w = sum_weights[(b * N + n) * C + c];
                float denom = (w > 1.0f) ? w : 1.0f;
                if (gate_denominators) gate_denominators[(b * N + n) * C + c] = denom;
                if (denom < denom_min) denom_min = denom;
                if (denom > denom_max) denom_max = denom;

                float inv_denom = 1.0f / denom;
                float* out_ptr = node_aggregates + ((b * N + n) * C + c) * 3;
                const float* msg_ptr = sum_msgs + ((b * N + n) * C + c) * 3;

                out_ptr[0] = msg_ptr[0] * inv_denom;
                out_ptr[1] = msg_ptr[1] * inv_denom;
                out_ptr[2] = msg_ptr[2] * inv_denom;

                if (!isfinite(out_ptr[0]) || !isfinite(out_ptr[1]) || !isfinite(out_ptr[2])) {
                    return GEO_CL30_MORPHISM_NONFINITE;
                }
            }
        }
    }

    if (telemetry && total_evals > 0) {
        float inv_tot = 1.0f / (float)total_evals;
        telemetry->max_rotor_residual = max_res;
        telemetry->alpha_mean = alpha_sum * inv_tot;
        telemetry->alpha_std = (float)sqrt((double)fmaxf(0.0f, alpha_sq_sum * inv_tot - (alpha_sum * inv_tot) * (alpha_sum * inv_tot)));
        telemetry->alpha_min = alpha_min;
        telemetry->alpha_max = alpha_max;
        telemetry->beta_mean = beta_sum * inv_tot;
        telemetry->beta_std = (float)sqrt((double)fmaxf(0.0f, beta_sq_sum * inv_tot - (beta_sum * inv_tot) * (beta_sum * inv_tot)));
        telemetry->beta_min = beta_min;
        telemetry->beta_max = beta_max;
        telemetry->gate_product_mean = prod_sum * inv_tot;
        telemetry->gate_product_std = (float)sqrt((double)fmaxf(0.0f, prod_sq_sum * inv_tot - (prod_sum * inv_tot) * (prod_sum * inv_tot)));
        telemetry->gate_denominator_min = denom_min;
        telemetry->gate_denominator_max = denom_max;
        telemetry->alpha_sat_low_rate = (float)a_sat_low * inv_tot;
        telemetry->alpha_sat_high_rate = (float)a_sat_high * inv_tot;
        telemetry->beta_sat_low_rate = (float)b_sat_low * inv_tot;
        telemetry->beta_sat_high_rate = (float)b_sat_high * inv_tot;
    }

    return GEO_CL30_MORPHISM_OK;
}

geo_cl30_morphism_status geo_cl30_morphism_vjp_f32(
    const geo_cl30_morphism_shape* shape,
    const float* relation_bivectors,
    const float* node_states,
    const float* goal_queries,
    const float* applicability_params,
    const float* compatibility_params,
    const int32_t* edge_batch,
    const int32_t* edge_source,
    const int32_t* edge_relation,
    const int32_t* edge_destination,
    const float* grad_node_aggregates,
    float* grad_relation_bivectors,
    float* grad_node_states,
    float* grad_goal_queries,
    float* grad_applicability_params,
    float* grad_compatibility_params,
    void* workspace,
    size_t workspace_bytes
) {
    if (!shape || !relation_bivectors || !node_states || !goal_queries ||
        !applicability_params || !compatibility_params ||
        !edge_batch || !edge_source || !edge_relation || !edge_destination ||
        !grad_node_aggregates || !grad_relation_bivectors || !grad_node_states ||
        !grad_goal_queries || !grad_applicability_params || !grad_compatibility_params ||
        !workspace) {
        return GEO_CL30_MORPHISM_INVALID_ARGUMENT;
    }

    int32_t B = shape->batch_count;
    int32_t N = shape->node_count;
    int32_t E = shape->edge_count;
    int32_t R = shape->relation_count;
    int32_t C = shape->channel_count;

    size_t req_bytes = geo_cl30_morphism_workspace_bytes(shape);
    if (workspace_bytes < req_bytes) return GEO_CL30_MORPHISM_INVALID_ARGUMENT;

    float* so3_matrices = (float*)workspace;
    float* local_alphas = so3_matrices + (R * C * 9);
    float* local_betas = local_alphas + (E * C);
    float* sum_weights = local_betas + (E * C);
    float* sum_msgs = sum_weights + (B * N * C);

    float w_a = applicability_params[0];
    float w_ar = applicability_params[1];
    float b_a = applicability_params[2];

    float w_s = compatibility_params[0];
    float theta_b = compatibility_params[1];
    float w_b = geo_softplus(theta_b);
    float d_wb_d_thetab = geo_sigmoid(theta_b);
    float w_q = compatibility_params[2];
    float w_tau = compatibility_params[3];
    float b_beta = compatibility_params[4];

    /* Zero out gradients */
    memset(grad_relation_bivectors, 0, R * C * 3 * sizeof(float));
    memset(grad_node_states, 0, B * N * C * 3 * sizeof(float));
    memset(grad_goal_queries, 0, B * C * 3 * sizeof(float));
    memset(grad_applicability_params, 0, 3 * sizeof(float));
    memset(grad_compatibility_params, 0, 5 * sizeof(float));

    /* Recompute forward values in workspace */
    float max_res = 0.0f;
    for (int32_t r = 0; r < R; ++r) {
        for (int32_t c = 0; c < C; ++c) {
            const float* b_ptr = relation_bivectors + (r * C + c) * 3;
            float* m_ptr = so3_matrices + (r * C + c) * 9;
            geo_bivector_to_so3(b_ptr, m_ptr, &max_res);
        }
    }

    memset(sum_weights, 0, B * N * C * sizeof(float));
    memset(sum_msgs, 0, B * N * C * 3 * sizeof(float));

    for (int32_t e = 0; e < E; ++e) {
        int32_t b_idx = edge_batch[e];
        int32_t u_idx = edge_source[e];
        int32_t r_idx = edge_relation[e];
        int32_t v_idx = edge_destination[e];

        for (int32_t c = 0; c < C; ++c) {
            const float* h_u = node_states + ((b_idx * N + u_idx) * C + c) * 3;
            const float* h_v = node_states + ((b_idx * N + v_idx) * C + c) * 3;
            const float* q_vec = goal_queries + (b_idx * C + c) * 3;
            const float* m_rot = so3_matrices + (r_idx * C + c) * 9;

            float t_vec[3];
            t_vec[0] = m_rot[0] * h_u[0] + m_rot[1] * h_u[1] + m_rot[2] * h_u[2];
            t_vec[1] = m_rot[3] * h_u[0] + m_rot[4] * h_u[1] + m_rot[5] * h_u[2];
            t_vec[2] = m_rot[6] * h_u[0] + m_rot[7] * h_u[1] + m_rot[8] * h_u[2];

            float h_u_hat[3], t_hat[3], h_v_hat[3], q_hat[3];
            geo_vec3_normalize(h_u, h_u_hat);
            geo_vec3_normalize(t_vec, t_hat);
            geo_vec3_normalize(h_v, h_v_hat);
            geo_vec3_normalize(q_vec, q_hat);

            float dot_hu_q = geo_vec3_dot(h_u_hat, q_hat);
            float dot_t_q = geo_vec3_dot(t_hat, q_hat);
            float raw_alpha = w_a * dot_hu_q + w_ar * dot_t_q + b_a;
            float alpha = geo_sigmoid(raw_alpha);

            float dot_hv_t = geo_vec3_dot(h_v_hat, t_hat);
            float cross_hv_t[3];
            geo_vec3_cross(h_v_hat, t_hat, cross_hv_t);
            float cross_sq = geo_vec3_dot(cross_hv_t, cross_hv_t);
            float bivector_norm = (float)sqrt((double)(cross_sq + 1e-12f));
            float dot_q_t = geo_vec3_dot(q_hat, t_hat);
            float pseudoscalar_dual = geo_vec3_dot(cross_hv_t, q_hat);

            float raw_beta = w_s * dot_hv_t - w_b * bivector_norm + w_q * dot_q_t + w_tau * pseudoscalar_dual + b_beta;
            float beta = geo_sigmoid(raw_beta);

            local_alphas[e * C + c] = alpha;
            local_betas[e * C + c] = beta;

            float weight = alpha * beta;
            sum_weights[(b_idx * N + v_idx) * C + c] += weight;

            float* dst_msg = sum_msgs + ((b_idx * N + v_idx) * C + c) * 3;
            dst_msg[0] += weight * t_vec[0];
            dst_msg[1] += weight * t_vec[1];
            dst_msg[2] += weight * t_vec[2];
        }
    }

    /* Backpropagate through aggregation to each edge */
    for (int32_t e = 0; e < E; ++e) {
        int32_t b_idx = edge_batch[e];
        int32_t u_idx = edge_source[e];
        int32_t r_idx = edge_relation[e];
        int32_t v_idx = edge_destination[e];

        for (int32_t c = 0; c < C; ++c) {
            const float* h_u = node_states + ((b_idx * N + u_idx) * C + c) * 3;
            const float* h_v = node_states + ((b_idx * N + v_idx) * C + c) * 3;
            const float* q_vec = goal_queries + (b_idx * C + c) * 3;
            const float* m_rot = so3_matrices + (r_idx * C + c) * 9;
            const float* grad_agg = grad_node_aggregates + ((b_idx * N + v_idx) * C + c) * 3;

            float w_total = sum_weights[(b_idx * N + v_idx) * C + c];
            float denom = (w_total > 1.0f) ? w_total : 1.0f;
            float inv_denom = 1.0f / denom;

            float t_vec[3];
            t_vec[0] = m_rot[0] * h_u[0] + m_rot[1] * h_u[1] + m_rot[2] * h_u[2];
            t_vec[1] = m_rot[3] * h_u[0] + m_rot[4] * h_u[1] + m_rot[5] * h_u[2];
            t_vec[2] = m_rot[6] * h_u[0] + m_rot[7] * h_u[1] + m_rot[8] * h_u[2];

            float h_u_hat[3], t_hat[3], h_v_hat[3], q_hat[3];
            float s_hu = geo_vec3_normalize(h_u, h_u_hat);
            float s_t = geo_vec3_normalize(t_vec, t_hat);
            float s_hv = geo_vec3_normalize(h_v, h_v_hat);
            float s_q = geo_vec3_normalize(q_vec, q_hat);

            float alpha = local_alphas[e * C + c];
            float beta = local_betas[e * C + c];
            float weight = alpha * beta;

            /* Gradient of a_tilde w.r.t m_e and weight */
            float grad_me[3];
            grad_me[0] = grad_agg[0] * inv_denom;
            grad_me[1] = grad_agg[1] * inv_denom;
            grad_me[2] = grad_agg[2] * inv_denom;

            float grad_weight = 0.0f;
            if (w_total > 1.0f) {
                const float* msg_sum = sum_msgs + ((b_idx * N + v_idx) * C + c) * 3;
                float dot_grad_sum = grad_agg[0] * msg_sum[0] + grad_agg[1] * msg_sum[1] + grad_agg[2] * msg_sum[2];
                grad_weight = -dot_grad_sum * (inv_denom * inv_denom);
            }

            /* m_e = weight * t_vec */
            float grad_t_from_me[3];
            grad_t_from_me[0] = weight * grad_me[0];
            grad_t_from_me[1] = weight * grad_me[1];
            grad_t_from_me[2] = weight * grad_me[2];

            float dot_me_t = grad_me[0] * t_vec[0] + grad_me[1] * t_vec[1] + grad_me[2] * t_vec[2];
            float grad_weight_total = grad_weight + dot_me_t;

            /* weight = alpha * beta */
            float grad_alpha = grad_weight_total * beta;
            float grad_beta = grad_weight_total * alpha;

            /* raw_alpha and raw_beta derivatives */
            float grad_raw_alpha = grad_alpha * alpha * (1.0f - alpha);
            float grad_raw_beta = grad_beta * beta * (1.0f - beta);

            /* Applicability parameter gradients */
            float dot_hu_q = geo_vec3_dot(h_u_hat, q_hat);
            float dot_t_q = geo_vec3_dot(t_hat, q_hat);
            grad_applicability_params[0] += grad_raw_alpha * dot_hu_q;
            grad_applicability_params[1] += grad_raw_alpha * dot_t_q;
            grad_applicability_params[2] += grad_raw_alpha;

            /* Compatibility parameter gradients */
            float dot_hv_t = geo_vec3_dot(h_v_hat, t_hat);
            float cross_hv_t[3];
            geo_vec3_cross(h_v_hat, t_hat, cross_hv_t);
            float cross_sq = geo_vec3_dot(cross_hv_t, cross_hv_t);
            float bivector_norm = (float)sqrt((double)(cross_sq + 1e-12f));
            float dot_q_t = geo_vec3_dot(q_hat, t_hat);
            float pseudoscalar_dual = geo_vec3_dot(cross_hv_t, q_hat);

            grad_compatibility_params[0] += grad_raw_beta * dot_hv_t;
            grad_compatibility_params[1] += grad_raw_beta * (-bivector_norm * d_wb_d_thetab);
            grad_compatibility_params[2] += grad_raw_beta * dot_q_t;
            grad_compatibility_params[3] += grad_raw_beta * pseudoscalar_dual;
            grad_compatibility_params[4] += grad_raw_beta;

            /* Gradients w.r.t normalized vectors */
            float grad_hu_hat[3] = {0.0f, 0.0f, 0.0f};
            float grad_t_hat[3] = {0.0f, 0.0f, 0.0f};
            float grad_hv_hat[3] = {0.0f, 0.0f, 0.0f};
            float grad_q_hat[3] = {0.0f, 0.0f, 0.0f};

            /* from alpha */
            grad_hu_hat[0] += grad_raw_alpha * w_a * q_hat[0];
            grad_hu_hat[1] += grad_raw_alpha * w_a * q_hat[1];
            grad_hu_hat[2] += grad_raw_alpha * w_a * q_hat[2];

            grad_t_hat[0] += grad_raw_alpha * w_ar * q_hat[0];
            grad_t_hat[1] += grad_raw_alpha * w_ar * q_hat[1];
            grad_t_hat[2] += grad_raw_alpha * w_ar * q_hat[2];

            grad_q_hat[0] += grad_raw_alpha * (w_a * h_u_hat[0] + w_ar * t_hat[0]);
            grad_q_hat[1] += grad_raw_alpha * (w_a * h_u_hat[1] + w_ar * t_hat[1]);
            grad_q_hat[2] += grad_raw_alpha * (w_a * h_u_hat[2] + w_ar * t_hat[2]);

            /* from beta: dot_hv_t */
            grad_hv_hat[0] += grad_raw_beta * w_s * t_hat[0];
            grad_hv_hat[1] += grad_raw_beta * w_s * t_hat[1];
            grad_hv_hat[2] += grad_raw_beta * w_s * t_hat[2];

            grad_t_hat[0] += grad_raw_beta * w_s * h_v_hat[0];
            grad_t_hat[1] += grad_raw_beta * w_s * h_v_hat[1];
            grad_t_hat[2] += grad_raw_beta * w_s * h_v_hat[2];

            /* from beta: bivector mismatch norm & pseudoscalar */
            float grad_cross[3];
            float inv_biv_norm = (bivector_norm > 1e-6f) ? (1.0f / bivector_norm) : 0.0f;
            float factor_biv = grad_raw_beta * (-w_b) * inv_biv_norm;
            grad_cross[0] = factor_biv * cross_hv_t[0] + grad_raw_beta * w_tau * q_hat[0];
            grad_cross[1] = factor_biv * cross_hv_t[1] + grad_raw_beta * w_tau * q_hat[1];
            grad_cross[2] = factor_biv * cross_hv_t[2] + grad_raw_beta * w_tau * q_hat[2];

            /* cross product derivatives: cross = h_v_hat x t_hat */
            float g_hv_cross[3], g_t_cross[3];
            geo_vec3_cross(t_hat, grad_cross, g_hv_cross);
            geo_vec3_cross(grad_cross, h_v_hat, g_t_cross);

            grad_hv_hat[0] += g_hv_cross[0];
            grad_hv_hat[1] += g_hv_cross[1];
            grad_hv_hat[2] += g_hv_cross[2];

            grad_t_hat[0] += g_t_cross[0];
            grad_t_hat[1] += g_t_cross[1];
            grad_t_hat[2] += g_t_cross[2];

            /* from beta: dot_q_t & pseudoscalar w.r.t q_hat */
            grad_q_hat[0] += grad_raw_beta * (w_q * t_hat[0] + w_tau * cross_hv_t[0]);
            grad_q_hat[1] += grad_raw_beta * (w_q * t_hat[1] + w_tau * cross_hv_t[1]);
            grad_q_hat[2] += grad_raw_beta * (w_q * t_hat[2] + w_tau * cross_hv_t[2]);

            grad_t_hat[0] += grad_raw_beta * w_q * q_hat[0];
            grad_t_hat[1] += grad_raw_beta * w_q * q_hat[1];
            grad_t_hat[2] += grad_raw_beta * w_q * q_hat[2];

            /* Backpropagate through vector normalization */
            /* grad_x = (grad_x_hat - (grad_x_hat . x_hat) * x_hat) / s_x */
            float dot_ghu_hu = geo_vec3_dot(grad_hu_hat, h_u_hat);
            float dot_gt_t = geo_vec3_dot(grad_t_hat, t_hat);
            float dot_ghv_hv = geo_vec3_dot(grad_hv_hat, h_v_hat);
            float dot_gq_q = geo_vec3_dot(grad_q_hat, q_hat);

            float grad_hu_unnorm[3];
            grad_hu_unnorm[0] = (grad_hu_hat[0] - dot_ghu_hu * h_u_hat[0]) / s_hu;
            grad_hu_unnorm[1] = (grad_hu_hat[1] - dot_ghu_hu * h_u_hat[1]) / s_hu;
            grad_hu_unnorm[2] = (grad_hu_hat[2] - dot_ghu_hu * h_u_hat[2]) / s_hu;

            float grad_hv_unnorm[3];
            grad_hv_unnorm[0] = (grad_hv_hat[0] - dot_ghv_hv * h_v_hat[0]) / s_hv;
            grad_hv_unnorm[1] = (grad_hv_hat[1] - dot_ghv_hv * h_v_hat[1]) / s_hv;
            grad_hv_unnorm[2] = (grad_hv_hat[2] - dot_ghv_hv * h_v_hat[2]) / s_hv;

            float grad_q_unnorm[3];
            grad_q_unnorm[0] = (grad_q_hat[0] - dot_gq_q * q_hat[0]) / s_q;
            grad_q_unnorm[1] = (grad_q_hat[1] - dot_gq_q * q_hat[1]) / s_q;
            grad_q_unnorm[2] = (grad_q_hat[2] - dot_gq_q * q_hat[2]) / s_q;

            float grad_t_total[3];
            grad_t_total[0] = grad_t_from_me[0] + (grad_t_hat[0] - dot_gt_t * t_hat[0]) / s_t;
            grad_t_total[1] = grad_t_from_me[1] + (grad_t_hat[1] - dot_gt_t * t_hat[1]) / s_t;
            grad_t_total[2] = grad_t_from_me[2] + (grad_t_hat[2] - dot_gt_t * t_hat[2]) / s_t;

            /* Accumulate node_states and goal_queries gradients */
            float* g_hu_dst = grad_node_states + ((b_idx * N + u_idx) * C + c) * 3;
            float* g_hv_dst = grad_node_states + ((b_idx * N + v_idx) * C + c) * 3;
            float* g_q_dst = grad_goal_queries + (b_idx * C + c) * 3;

            g_hv_dst[0] += grad_hv_unnorm[0];
            g_hv_dst[1] += grad_hv_unnorm[1];
            g_hv_dst[2] += grad_hv_unnorm[2];

            g_q_dst[0] += grad_q_unnorm[0];
            g_q_dst[1] += grad_q_unnorm[1];
            g_q_dst[2] += grad_q_unnorm[2];

            /* t = M_r * h_u */
            g_hu_dst[0] += grad_hu_unnorm[0] + (m_rot[0] * grad_t_total[0] + m_rot[3] * grad_t_total[1] + m_rot[6] * grad_t_total[2]);
            g_hu_dst[1] += grad_hu_unnorm[1] + (m_rot[1] * grad_t_total[0] + m_rot[4] * grad_t_total[1] + m_rot[7] * grad_t_total[2]);
            g_hu_dst[2] += grad_hu_unnorm[2] + (m_rot[2] * grad_t_total[0] + m_rot[5] * grad_t_total[1] + m_rot[8] * grad_t_total[2]);

            /* Gradient w.r.t bivector B_r via rotor chain rule */
            const float* b_vec = relation_bivectors + (r_idx * C + c) * 3;
            float* g_b_dst = grad_relation_bivectors + (r_idx * C + c) * 3;

            float b1 = b_vec[0], b2 = b_vec[1], b3 = b_vec[2];
            float theta_sq = b1 * b1 + b2 * b2 + b3 * b3;
            float theta = (float)sqrt((double)theta_sq);
            float s, k;

            if (theta < 1e-7f) {
                s = 1.0f - 0.125f * theta_sq;
                k = -0.5f + (1.0f / 48.0f) * theta_sq;
            } else {
                float half_theta = 0.5f * theta;
                s = (float)cos((double)half_theta);
                k = -(float)sin((double)half_theta) / theta;
            }

            float rb1 = k * b1, rb2 = k * b2, rb3 = k * b3;
            float norm_sq = s * s + rb1 * rb1 + rb2 * rb2 + rb3 * rb3;
            if (norm_sq > 0.0f) {
                float inv_norm = 1.0f / (float)sqrt((double)norm_sq);
                s *= inv_norm;
                rb1 *= inv_norm;
                rb2 *= inv_norm;
                rb3 *= inv_norm;
            }

            /* Grad w.r.t M_r: g_M = grad_t_total * h_u^T */
            float g_M[9];
            g_M[0] = grad_t_total[0] * h_u[0]; g_M[1] = grad_t_total[0] * h_u[1]; g_M[2] = grad_t_total[0] * h_u[2];
            g_M[3] = grad_t_total[1] * h_u[0]; g_M[4] = grad_t_total[1] * h_u[1]; g_M[5] = grad_t_total[1] * h_u[2];
            g_M[6] = grad_t_total[2] * h_u[0]; g_M[7] = grad_t_total[2] * h_u[1]; g_M[8] = grad_t_total[2] * h_u[2];

            /* Chain rule to rotor quaternion (s, rb1, rb2, rb3) */
            float g_s = 2.0f * (g_M[1] * rb1 - g_M[2] * rb3 - g_M[3] * rb1 + g_M[5] * rb2 + g_M[6] * rb3 - g_M[7] * rb2);
            float g_rb1 = -4.0f * (g_M[0] + g_M[4]) * rb1 + 2.0f * (g_M[1] * s + g_M[2] * rb2 - g_M[3] * s + g_M[5] * rb3 + g_M[6] * rb2 + g_M[7] * rb3);
            float g_rb2 = -4.0f * (g_M[4] + g_M[8]) * rb2 + 2.0f * (g_M[1] * rb3 + g_M[2] * rb1 + g_M[3] * rb3 + g_M[5] * s + g_M[6] * rb1 - g_M[7] * s);
            float g_rb3 = -4.0f * (g_M[0] + g_M[8]) * rb3 + 2.0f * (g_M[1] * rb2 - g_M[2] * s + g_M[3] * rb2 + g_M[5] * rb1 + g_M[6] * s + g_M[7] * rb1);

            /* Chain rule to bivector components (b1, b2, b3) */
            float dot_g_rb = g_rb1 * b1 + g_rb2 * b2 + g_rb3 * b3;
            float ds_dtheta = -0.5f * (float)sin((double)(0.5f * theta));
            float dk_dtheta = (theta > 1e-7f) ? (-(0.5f * (float)cos((double)(0.5f * theta)) * theta - (float)sin((double)(0.5f * theta))) / (theta * theta)) : (1.0f / 24.0f);
            float inv_theta = (theta > 1e-7f) ? (1.0f / theta) : 0.0f;

            g_b_dst[0] += k * g_rb1 + (g_s * ds_dtheta + dot_g_rb * dk_dtheta) * b1 * inv_theta;
            g_b_dst[1] += k * g_rb2 + (g_s * ds_dtheta + dot_g_rb * dk_dtheta) * b2 * inv_theta;
            g_b_dst[2] += k * g_rb3 + (g_s * ds_dtheta + dot_g_rb * dk_dtheta) * b3 * inv_theta;
        }
    }

    return GEO_CL30_MORPHISM_OK;
}
