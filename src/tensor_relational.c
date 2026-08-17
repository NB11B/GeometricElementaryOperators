#include "geo/tensor_relational.h"

#include <math.h>
#include <string.h>
#include <float.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_EXP(x) exp(x)
#define GEO_LOG(x) log(x)
#define GEO_FABS(x) fabs(x)
#define GEO_SQRT(x) sqrt(x)
#define GEO_MAX_REAL DBL_MAX
#else
#define GEO_EXP(x) expf(x)
#define GEO_LOG(x) logf(x)
#define GEO_FABS(x) fabsf(x)
#define GEO_SQRT(x) sqrtf(x)
#define GEO_MAX_REAL FLT_MAX
#endif

uint32_t geo_relational_abi_version(void) {
    return GEO_RELATIONAL_ABI_VERSION;
}

const char *geo_relational_status_string(geo_relational_status status) {
    switch (status) {
        case GEO_RELATIONAL_OK: return "GEO_RELATIONAL_OK";
        case GEO_RELATIONAL_INVALID_ARGUMENT: return "GEO_RELATIONAL_INVALID_ARGUMENT";
        case GEO_RELATIONAL_OVERFLOW: return "GEO_RELATIONAL_OVERFLOW";
        case GEO_RELATIONAL_INSUFFICIENT_WORKSPACE: return "GEO_RELATIONAL_INSUFFICIENT_WORKSPACE";
        case GEO_RELATIONAL_NUMERIC_FAILURE: return "GEO_RELATIONAL_NUMERIC_FAILURE";
        case GEO_RELATIONAL_CONSTRAINT_FAILURE: return "GEO_RELATIONAL_CONSTRAINT_FAILURE";
        case GEO_RELATIONAL_CUDA_ERROR: return "GEO_RELATIONAL_CUDA_ERROR";
        case GEO_RELATIONAL_BACKEND_UNAVAILABLE: return "GEO_RELATIONAL_BACKEND_UNAVAILABLE";
        case GEO_RELATIONAL_UNSUPPORTED: return "GEO_RELATIONAL_UNSUPPORTED";
        default: return "UNKNOWN_STATUS";
    }
}

size_t geo_relational_projection_workspace_elements(
    size_t matrix_count,
    size_t streams,
    uint32_t iterations,
    int backward
) {
    if (streams == 0 || matrix_count == 0) return 0;
    size_t per_matrix_steps = backward ? ((size_t)iterations + 1) : 1;
    return matrix_count * per_matrix_steps * streams * streams;
}

static geo_real_t logsumexp_row(const geo_real_t *row, size_t n) {
    geo_real_t max_val = -GEO_MAX_REAL;
    for (size_t j = 0; j < n; ++j) {
        if (row[j] > max_val) max_val = row[j];
    }
    if (isnan(max_val) || isinf(max_val)) return max_val;
    geo_real_t sum = (geo_real_t)0.0;
    for (size_t j = 0; j < n; ++j) {
        sum += GEO_EXP(row[j] - max_val);
    }
    return max_val + GEO_LOG(sum);
}

static geo_real_t logsumexp_col(const geo_real_t *mat, size_t col, size_t streams, size_t stride) {
    geo_real_t max_val = -GEO_MAX_REAL;
    for (size_t i = 0; i < streams; ++i) {
        geo_real_t val = mat[i * stride + col];
        if (val > max_val) max_val = val;
    }
    if (isnan(max_val) || isinf(max_val)) return max_val;
    geo_real_t sum = (geo_real_t)0.0;
    for (size_t i = 0; i < streams; ++i) {
        sum += GEO_EXP(mat[i * stride + col] - max_val);
    }
    return max_val + GEO_LOG(sum);
}

geo_relational_status geo_relational_certify(
    const geo_real_t *relationship,
    geo_real_t tolerance,
    const geo_relational_shape *shape,
    geo_relational_certificate *certificates
) {
    if (!relationship || !shape || !certificates) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (shape->streams == 0 || shape->streams > GEO_RELATIONAL_MAX_STREAMS) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (shape->matrix_count == 0) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t P = shape->streams;
    size_t M = shape->matrix_count;

    for (size_t m = 0; m < M; ++m) {
        const geo_real_t *H = relationship + m * P * P;
        geo_relational_certificate *cert = certificates + m;
        memset(cert, 0, sizeof(geo_relational_certificate));
        cert->abi_version = GEO_RELATIONAL_ABI_VERSION;
        cert->streams = P;
        cert->certificate_tolerance = tolerance;

        int finite = 1;
        geo_real_t min_e = GEO_MAX_REAL;
        geo_real_t max_e = -GEO_MAX_REAL;
        geo_real_t max_row_err = (geo_real_t)0.0;
        geo_real_t max_col_err = (geo_real_t)0.0;
        geo_real_t fwd_gain = (geo_real_t)0.0;
        geo_real_t bwd_gain = (geo_real_t)0.0;

        for (size_t i = 0; i < P * P; ++i) {
            geo_real_t val = H[i];
            if (isnan(val) || isinf(val)) { finite = 0; break; }
            if (val < min_e) min_e = val;
            if (val > max_e) max_e = val;
        }

        if (!finite) {
            cert->finite = 0;
            cert->accepted = 0;
            continue;
        }

        cert->finite = 1;
        cert->minimum_entry = min_e;
        cert->maximum_entry = max_e;

        // Row sums & fwd gain
        for (size_t p = 0; p < P; ++p) {
            geo_real_t rsum = (geo_real_t)0.0;
            geo_real_t rabs = (geo_real_t)0.0;
            for (size_t q = 0; q < P; ++q) {
                geo_real_t v = H[p * P + q];
                rsum += v;
                rabs += GEO_FABS(v);
            }
            geo_real_t rerr = GEO_FABS(rsum - (geo_real_t)1.0);
            if (rerr > max_row_err) max_row_err = rerr;
            if (rabs > fwd_gain) fwd_gain = rabs;
        }

        // Col sums & bwd gain
        for (size_t q = 0; q < P; ++q) {
            geo_real_t csum = (geo_real_t)0.0;
            geo_real_t cabs = (geo_real_t)0.0;
            for (size_t p = 0; p < P; ++p) {
                geo_real_t v = H[p * P + q];
                csum += v;
                cabs += GEO_FABS(v);
            }
            geo_real_t cerr = GEO_FABS(csum - (geo_real_t)1.0);
            if (cerr > max_col_err) max_col_err = cerr;
            if (cabs > bwd_gain) bwd_gain = cabs;
        }

        cert->maximum_row_sum_error = max_row_err;
        cert->maximum_column_sum_error = max_col_err;
        cert->forward_amax_gain = fwd_gain;
        cert->backward_amax_gain = bwd_gain;

        // Identity distance
        geo_real_t d_I_sq = (geo_real_t)0.0;
        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < P; ++q) {
                geo_real_t target = (p == q) ? (geo_real_t)1.0 : (geo_real_t)0.0;
                geo_real_t diff = H[p * P + q] - target;
                d_I_sq += diff * diff;
            }
        }
        cert->identity_distance_frobenius = GEO_SQRT(d_I_sq);

        // Consensus distance
        geo_real_t d_J_sq = (geo_real_t)0.0;
        geo_real_t inv_P = (geo_real_t)1.0 / (geo_real_t)P;
        for (size_t i = 0; i < P * P; ++i) {
            geo_real_t diff = H[i] - inv_P;
            d_J_sq += diff * diff;
        }
        cert->consensus_distance_frobenius = GEO_SQRT(d_J_sq);

        cert->nonnegative = (min_e >= -tolerance);
        cert->row_balanced = (max_row_err <= tolerance);
        cert->column_balanced = (max_col_err <= tolerance);
        cert->accepted = cert->nonnegative && cert->row_balanced && cert->column_balanced &&
                         (fwd_gain <= (geo_real_t)1.0 + tolerance) && (bwd_gain <= (geo_real_t)1.0 + tolerance);
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_project_forward(
    const geo_real_t *logits,
    geo_real_t *relationship,
    geo_real_t *workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options,
    geo_relational_certificate *certificates
) {
    if (!logits || !relationship || !workspace || !shape || !options) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (shape->streams == 0 || shape->streams > GEO_RELATIONAL_MAX_STREAMS) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (shape->matrix_count == 0) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (options->iterations > GEO_RELATIONAL_MAX_SINKHORN_ITERATIONS) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t req_ws = geo_relational_projection_workspace_elements(shape->matrix_count, shape->streams, options->iterations, 1);
    if (workspace_elements < req_ws) return GEO_RELATIONAL_INSUFFICIENT_WORKSPACE;

    size_t P = shape->streams;
    size_t M = shape->matrix_count;
    uint32_t K = options->iterations;

    for (size_t m = 0; m < M; ++m) {
        const geo_real_t *A = logits + m * P * P;
        geo_real_t *H = relationship + m * P * P;
        geo_real_t *ws_m = workspace + m * (K + 1) * P * P;

        // Check non-finite
        if (options->fail_on_nonfinite) {
            for (size_t i = 0; i < P * P; ++i) {
                if (isnan(A[i]) || isinf(A[i])) return GEO_RELATIONAL_NUMERIC_FAILURE;
            }
        }

        // Copy A to step 0 workspace
        memcpy(ws_m, A, P * P * sizeof(geo_real_t));

        for (uint32_t k = 0; k < K; ++k) {
            const geo_real_t *curr = ws_m + k * P * P;
            geo_real_t *next = ws_m + (k + 1) * P * P;

            // Temporary step buffer
            geo_real_t Z_row[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];

            // 1. Row norm
            for (size_t p = 0; p < P; ++p) {
                geo_real_t lse = logsumexp_row(curr + p * P, P);
                if (isnan(lse) || isinf(lse)) return GEO_RELATIONAL_NUMERIC_FAILURE;
                for (size_t q = 0; q < P; ++q) {
                    Z_row[p * P + q] = curr[p * P + q] - lse;
                }
            }

            // 2. Col norm
            for (size_t q = 0; q < P; ++q) {
                geo_real_t lse = logsumexp_col(Z_row, q, P, P);
                if (isnan(lse) || isinf(lse)) return GEO_RELATIONAL_NUMERIC_FAILURE;
                for (size_t p = 0; p < P; ++p) {
                    next[p * P + q] = Z_row[p * P + q] - lse;
                }
            }
        }

        // Output S = exp(Z_K)
        const geo_real_t *Z_K = ws_m + K * P * P;
        for (size_t i = 0; i < P * P; ++i) {
            H[i] = GEO_EXP(Z_K[i]);
        }
    }

    if (certificates) {
        geo_relational_status cert_status = geo_relational_certify(relationship, options->epsilon, shape, certificates);
        if (cert_status != GEO_RELATIONAL_OK) return cert_status;

        if (options->require_certificate) {
            for (size_t m = 0; m < M; ++m) {
                if (!certificates[m].accepted) return GEO_RELATIONAL_CONSTRAINT_FAILURE;
            }
        }
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_project_vjp(
    const geo_real_t *logits,
    const geo_real_t *relationship_cotangent,
    geo_real_t *logits_cotangent,
    geo_real_t *workspace,
    size_t workspace_elements,
    const geo_relational_shape *shape,
    const geo_relational_projection_options *options
) {
    if (!logits || !relationship_cotangent || !logits_cotangent || !workspace || !shape || !options)
        return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (shape->streams == 0 || shape->streams > GEO_RELATIONAL_MAX_STREAMS) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (shape->matrix_count == 0) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (options->iterations > GEO_RELATIONAL_MAX_SINKHORN_ITERATIONS) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t req_ws = geo_relational_projection_workspace_elements(shape->matrix_count, shape->streams, options->iterations, 1);
    if (workspace_elements < req_ws) return GEO_RELATIONAL_INSUFFICIENT_WORKSPACE;

    size_t P = shape->streams;
    size_t M = shape->matrix_count;
    uint32_t K = options->iterations;

    // First ensure forward trajectory is in workspace by running forward
    geo_relational_status fwd_status = geo_relational_project_forward(
        logits, logits_cotangent, workspace, workspace_elements, shape, options, NULL
    );
    if (fwd_status != GEO_RELATIONAL_OK) return fwd_status;

    for (size_t m = 0; m < M; ++m) {
        const geo_real_t *cot_S = relationship_cotangent + m * P * P;
        geo_real_t *cot_A = logits_cotangent + m * P * P;
        const geo_real_t *ws_m = workspace + m * (K + 1) * P * P;

        const geo_real_t *Z_K = ws_m + K * P * P;
        geo_real_t dZ[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];

        // S_bar * S
        for (size_t i = 0; i < P * P; ++i) {
            dZ[i] = cot_S[i] * GEO_EXP(Z_K[i]);
        }

        // Reverse iterations
        for (int k = (int)K - 1; k >= 0; --k) {
            const geo_real_t *Z_prev = ws_m + k * P * P;

            // Recompute Z_row for step k
            geo_real_t Z_row[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];
            for (size_t p = 0; p < P; ++p) {
                geo_real_t lse = logsumexp_row(Z_prev + p * P, P);
                for (size_t q = 0; q < P; ++q) {
                    Z_row[p * P + q] = Z_prev[p * P + q] - lse;
                }
            }

            // Reverse Column normalization
            // dZ_row[p, q] = dZ[p, q] - softmax(Z_row)[p, q] * sum_p' dZ[p', q]
            geo_real_t dZ_row[GEO_RELATIONAL_MAX_STREAMS * GEO_RELATIONAL_MAX_STREAMS];
            for (size_t q = 0; q < P; ++q) {
                geo_real_t sum_col_dZ = (geo_real_t)0.0;
                for (size_t p = 0; p < P; ++p) {
                    sum_col_dZ += dZ[p * P + q];
                }
                geo_real_t col_lse = logsumexp_col(Z_row, q, P, P);
                for (size_t p = 0; p < P; ++p) {
                    geo_real_t sm = GEO_EXP(Z_row[p * P + q] - col_lse);
                    dZ_row[p * P + q] = dZ[p * P + q] - sm * sum_col_dZ;
                }
            }

            // Reverse Row normalization
            // dZ_prev[p, q] = dZ_row[p, q] - softmax(Z_prev)[p, q] * sum_q' dZ_row[p, q']
            for (size_t p = 0; p < P; ++p) {
                geo_real_t sum_row_dZ = (geo_real_t)0.0;
                for (size_t q = 0; q < P; ++q) {
                    sum_row_dZ += dZ_row[p * P + q];
                }
                geo_real_t row_lse = logsumexp_row(Z_prev + p * P, P);
                for (size_t q = 0; q < P; ++q) {
                    geo_real_t sm = GEO_EXP(Z_prev[p * P + q] - row_lse);
                    dZ[p * P + q] = dZ_row[p * P + q] - sm * sum_row_dZ;
                }
            }
        }

        memcpy(cot_A, dZ, P * P * sizeof(geo_real_t));
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_identity_gate_forward(
    const geo_real_t *projected_relationship,
    const geo_real_t *gate,
    geo_real_t *effective_relationship,
    const geo_relational_shape *shape
) {
    if (!projected_relationship || !gate || !effective_relationship || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;
    size_t P = shape->streams;
    size_t M = shape->matrix_count;

    for (size_t m = 0; m < M; ++m) {
        geo_real_t g = gate[m];
        const geo_real_t *S = projected_relationship + m * P * P;
        geo_real_t *H = effective_relationship + m * P * P;

        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < P; ++q) {
                geo_real_t eye_val = (p == q) ? (geo_real_t)1.0 : (geo_real_t)0.0;
                H[p * P + q] = ((geo_real_t)1.0 - g) * eye_val + g * S[p * P + q];
            }
        }
    }
    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_identity_gate_vjp(
    const geo_real_t *projected_relationship,
    const geo_real_t *gate,
    const geo_real_t *effective_relationship_cotangent,
    geo_real_t *projected_relationship_cotangent,
    geo_real_t *gate_cotangent,
    const geo_relational_shape *shape
) {
    if (!projected_relationship || !gate || !effective_relationship_cotangent ||
        !projected_relationship_cotangent || !gate_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t P = shape->streams;
    size_t M = shape->matrix_count;

    for (size_t m = 0; m < M; ++m) {
        geo_real_t g = gate[m];
        const geo_real_t *S = projected_relationship + m * P * P;
        const geo_real_t *cot_H = effective_relationship_cotangent + m * P * P;
        geo_real_t *cot_S = projected_relationship_cotangent + m * P * P;

        geo_real_t sum_g_cot = (geo_real_t)0.0;

        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < P; ++q) {
                geo_real_t eye_val = (p == q) ? (geo_real_t)1.0 : (geo_real_t)0.0;
                geo_real_t h_cot = cot_H[p * P + q];
                cot_S[p * P + q] = g * h_cot;
                sum_g_cot += h_cot * (S[p * P + q] - eye_val);
            }
        }
        gate_cotangent[m] = sum_g_cot;
    }
    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_mix_forward(
    const geo_real_t *state,
    const geo_real_t *relationship,
    geo_real_t *output,
    const geo_relational_shape *shape
) {
    if (!state || !relationship || !output || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;
    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    size_t M = shape->matrix_count;
    if (M != 1 && M != G) return GEO_RELATIONAL_INVALID_ARGUMENT;

    for (size_t g = 0; g < G; ++g) {
        size_t m = (M == 1) ? 0 : g;
        const geo_real_t *H = relationship + m * P * P;
        const geo_real_t *X = state + g * P * D;
        geo_real_t *Y = output + g * P * D;

        for (size_t p = 0; p < P; ++p) {
            for (size_t d = 0; d < D; ++d) {
                geo_real_t sum = (geo_real_t)0.0;
                for (size_t q = 0; q < P; ++q) {
                    sum += H[p * P + q] * X[q * D + d];
                }
                Y[p * D + d] = sum;
            }
        }
    }
    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_mix_vjp(
    const geo_real_t *state,
    const geo_real_t *relationship,
    const geo_real_t *output_cotangent,
    geo_real_t *state_cotangent,
    geo_real_t *relationship_cotangent,
    const geo_relational_shape *shape
) {
    if (!state || !relationship || !output_cotangent || !state_cotangent || !relationship_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    size_t M = shape->matrix_count;
    if (M != 1 && M != G) return GEO_RELATIONAL_INVALID_ARGUMENT;

    memset(relationship_cotangent, 0, M * P * P * sizeof(geo_real_t));

    for (size_t g = 0; g < G; ++g) {
        size_t m = (M == 1) ? 0 : g;
        const geo_real_t *H = relationship + m * P * P;
        const geo_real_t *X = state + g * P * D;
        const geo_real_t *cot_Y = output_cotangent + g * P * D;
        geo_real_t *cot_X = state_cotangent + g * P * D;
        geo_real_t *cot_H = relationship_cotangent + m * P * P;

        // state_cotangent[g, q, d] = sum_p H[m, p, q] * cot_Y[g, p, d]
        for (size_t q = 0; q < P; ++q) {
            for (size_t d = 0; d < D; ++d) {
                geo_real_t sum = (geo_real_t)0.0;
                for (size_t p = 0; p < P; ++p) {
                    sum += H[p * P + q] * cot_Y[p * D + d];
                }
                cot_X[q * D + d] = sum;
            }
        }

        // relationship_cotangent[m, p, q] += sum_d cot_Y[g, p, d] * X[g, q, d]
        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < P; ++q) {
                geo_real_t sum = (geo_real_t)0.0;
                for (size_t d = 0; d < D; ++d) {
                    sum += cot_Y[p * D + d] * X[q * D + d];
                }
                cot_H[p * P + q] += sum;
            }
        }
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_read_forward(
    const geo_real_t *state,
    const geo_real_t *read_weights,
    size_t weight_count,
    geo_real_t *read_state,
    const geo_relational_shape *shape
) {
    if (!state || !read_weights || !read_state || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;
    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    if (weight_count != 1 && weight_count != G) return GEO_RELATIONAL_INVALID_ARGUMENT;

    for (size_t g = 0; g < G; ++g) {
        size_t m = (weight_count == 1) ? 0 : g;
        const geo_real_t *r = read_weights + m * P;
        const geo_real_t *X = state + g * P * D;
        geo_real_t *z = read_state + g * D;

        for (size_t d = 0; d < D; ++d) {
            geo_real_t sum = (geo_real_t)0.0;
            for (size_t p = 0; p < P; ++p) {
                sum += r[p] * X[p * D + d];
            }
            z[d] = sum;
        }
    }
    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_read_vjp(
    const geo_real_t *state,
    const geo_real_t *read_weights,
    size_t weight_count,
    const geo_real_t *read_state_cotangent,
    geo_real_t *state_cotangent,
    geo_real_t *read_weights_cotangent,
    const geo_relational_shape *shape
) {
    if (!state || !read_weights || !read_state_cotangent || !state_cotangent || !read_weights_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    if (weight_count != 1 && weight_count != G) return GEO_RELATIONAL_INVALID_ARGUMENT;

    memset(read_weights_cotangent, 0, weight_count * P * sizeof(geo_real_t));

    for (size_t g = 0; g < G; ++g) {
        size_t m = (weight_count == 1) ? 0 : g;
        const geo_real_t *r = read_weights + m * P;
        const geo_real_t *X = state + g * P * D;
        const geo_real_t *cot_z = read_state_cotangent + g * D;
        geo_real_t *cot_X = state_cotangent + g * P * D;
        geo_real_t *cot_r = read_weights_cotangent + m * P;

        for (size_t p = 0; p < P; ++p) {
            geo_real_t sum_r = (geo_real_t)0.0;
            for (size_t d = 0; d < D; ++d) {
                cot_X[p * D + d] = r[p] * cot_z[d];
                sum_r += cot_z[d] * X[p * D + d];
            }
            cot_r[p] += sum_r;
        }
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_write_add_forward(
    const geo_real_t *transported_state,
    const geo_real_t *source,
    const geo_real_t *write_weights,
    size_t weight_count,
    const geo_real_t *source_scale,
    size_t scale_count,
    geo_real_t *output,
    const geo_relational_shape *shape
) {
    if (!transported_state || !source || !write_weights || !output || !shape) return GEO_RELATIONAL_INVALID_ARGUMENT;
    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    if (weight_count != 1 && weight_count != G) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (scale_count != 0 && scale_count != 1 && scale_count != G) return GEO_RELATIONAL_INVALID_ARGUMENT;

    for (size_t g = 0; g < G; ++g) {
        size_t mw = (weight_count == 1) ? 0 : g;
        const geo_real_t *w = write_weights + mw * P;
        const geo_real_t *Y = transported_state + g * P * D;
        const geo_real_t *u = source + g * D;
        geo_real_t *X_plus = output + g * P * D;

        geo_real_t beta = (geo_real_t)1.0;
        if (source_scale && scale_count > 0) {
            size_t ms = (scale_count == 1) ? 0 : g;
            beta = source_scale[ms];
        }

        for (size_t p = 0; p < P; ++p) {
            geo_real_t factor = beta * w[p];
            for (size_t d = 0; d < D; ++d) {
                X_plus[p * D + d] = Y[p * D + d] + factor * u[d];
            }
        }
    }
    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_write_add_vjp(
    const geo_real_t *source,
    const geo_real_t *write_weights,
    size_t weight_count,
    const geo_real_t *source_scale,
    size_t scale_count,
    const geo_real_t *output_cotangent,
    geo_real_t *transported_state_cotangent,
    geo_real_t *source_cotangent,
    geo_real_t *write_weights_cotangent,
    geo_real_t *source_scale_cotangent,
    const geo_relational_shape *shape
) {
    if (!source || !write_weights || !output_cotangent || !transported_state_cotangent ||
        !source_cotangent || !write_weights_cotangent || !shape)
        return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t G = shape->groups;
    size_t P = shape->streams;
    size_t D = shape->features;
    if (weight_count != 1 && weight_count != G) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (scale_count != 0 && scale_count != 1 && scale_count != G) return GEO_RELATIONAL_INVALID_ARGUMENT;

    memset(write_weights_cotangent, 0, weight_count * P * sizeof(geo_real_t));
    if (source_scale_cotangent && scale_count > 0) {
        memset(source_scale_cotangent, 0, scale_count * sizeof(geo_real_t));
    }

    for (size_t g = 0; g < G; ++g) {
        size_t mw = (weight_count == 1) ? 0 : g;
        const geo_real_t *w = write_weights + mw * P;
        const geo_real_t *u = source + g * D;
        const geo_real_t *cot_X_plus = output_cotangent + g * P * D;
        geo_real_t *cot_Y = transported_state_cotangent + g * P * D;
        geo_real_t *cot_u = source_cotangent + g * D;
        geo_real_t *cot_w = write_weights_cotangent + mw * P;

        geo_real_t beta = (geo_real_t)1.0;
        size_t ms = 0;
        if (source_scale && scale_count > 0) {
            ms = (scale_count == 1) ? 0 : g;
            beta = source_scale[ms];
        }

        // cot_Y = cot_X_plus
        memcpy(cot_Y, cot_X_plus, P * D * sizeof(geo_real_t));

        // cot_u[d] = sum_p beta * w[p] * cot_X_plus[p, d]
        for (size_t d = 0; d < D; ++d) {
            geo_real_t sum = (geo_real_t)0.0;
            for (size_t p = 0; p < P; ++p) {
                sum += beta * w[p] * cot_X_plus[p * D + d];
            }
            cot_u[d] = sum;
        }

        // cot_w[p] += sum_d beta * cot_X_plus[p, d] * u[d]
        for (size_t p = 0; p < P; ++p) {
            geo_real_t sum = (geo_real_t)0.0;
            for (size_t d = 0; d < D; ++d) {
                sum += beta * cot_X_plus[p * D + d] * u[d];
            }
            cot_w[p] += sum;
        }

        // cot_beta += sum_{p, d} w[p] * cot_X_plus[p, d] * u[d]
        if (source_scale_cotangent && scale_count > 0) {
            geo_real_t sum_beta = (geo_real_t)0.0;
            for (size_t p = 0; p < P; ++p) {
                for (size_t d = 0; d < D; ++d) {
                    sum_beta += w[p] * cot_X_plus[p * D + d] * u[d];
                }
            }
            source_scale_cotangent[ms] += sum_beta;
        }
    }

    return GEO_RELATIONAL_OK;
}

geo_relational_status geo_relational_compose(
    const geo_real_t *left,
    const geo_real_t *right,
    geo_real_t *product,
    size_t matrix_count,
    size_t streams
) {
    if (!left || !right || !product) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (streams == 0 || streams > GEO_RELATIONAL_MAX_STREAMS) return GEO_RELATIONAL_INVALID_ARGUMENT;
    if (matrix_count == 0) return GEO_RELATIONAL_INVALID_ARGUMENT;

    size_t P = streams;
    for (size_t m = 0; m < matrix_count; ++m) {
        const geo_real_t *L = left + m * P * P;
        const geo_real_t *R = right + m * P * P;
        geo_real_t *P_mat = product + m * P * P;

        for (size_t i = 0; i < P; ++i) {
            for (size_t j = 0; j < P; ++j) {
                geo_real_t sum = (geo_real_t)0.0;
                for (size_t k = 0; k < P; ++k) {
                    sum += L[i * P + k] * R[k * P + j];
                }
                P_mat[i * P + j] = sum;
            }
        }
    }
    return GEO_RELATIONAL_OK;
}
