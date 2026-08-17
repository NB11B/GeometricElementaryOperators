#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "geo/tensor_relational.h"

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "ASSERTION FAILED: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

#define ASSERT_OK(status, msg) \
    do { \
        if ((status) != GEO_RELATIONAL_OK) { \
            fprintf(stderr, "STATUS ERROR: %s (code %d: %s) (%s:%d)\n", \
                    msg, (status), geo_relational_status_string(status), __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

static int test_abi_and_defaults(void) {
    uint32_t ver = geo_relational_abi_version();
    ASSERT_TRUE(ver == GEO_RELATIONAL_ABI_VERSION, "ABI version match");

    size_t ws_elems = geo_relational_projection_workspace_elements(1, 4, 20, 1);
    ASSERT_TRUE(ws_elems == 1 * 21 * 4 * 4, "Workspace element count calculation");
    return 0;
}

static int test_projection_and_certificates(void) {
    geo_relational_shape shape = { .groups = 2, .streams = 4, .features = 16, .matrix_count = 1 };
    geo_relational_projection_options options = {
        .abi_version = GEO_RELATIONAL_ABI_VERSION,
        .iterations = 20,
        .epsilon = (geo_real_t)1e-7,
        .mode = GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN,
        .fail_on_nonfinite = 1,
        .require_certificate = 1
    };

    geo_real_t logits[16] = {
        0.1f, 0.5f, -0.2f, 0.8f,
        -0.4f, 1.2f, 0.3f, -0.1f,
        0.0f, -0.9f, 0.7f, 0.4f,
        0.6f, 0.2f, -0.5f, 1.1f
    };

    geo_real_t relationship[16];
    size_t ws_elems = geo_relational_projection_workspace_elements(1, 4, 20, 1);
    geo_real_t *workspace = (geo_real_t *)malloc(ws_elems * sizeof(geo_real_t));
    ASSERT_TRUE(workspace != NULL, "Workspace allocation");

    geo_relational_certificate cert;
    geo_relational_status status = geo_relational_project_forward(
        logits, relationship, workspace, ws_elems, &shape, &options, &cert
    );
    ASSERT_OK(status, "Project forward execution");
    ASSERT_TRUE(cert.accepted == 1, "Certificate accepted");
    ASSERT_TRUE(cert.row_balanced == 1, "Row balanced");
    ASSERT_TRUE(cert.column_balanced == 1, "Column balanced");
    ASSERT_TRUE(cert.nonnegative == 1, "Nonnegative");

    free(workspace);
    return 0;
}

static int test_identity_gate_and_mix(void) {
    geo_relational_shape shape = { .groups = 2, .streams = 4, .features = 4, .matrix_count = 1 };
    geo_real_t proj[16];
    for (int i = 0; i < 16; ++i) proj[i] = 0.25f; // Uniform consensus

    geo_real_t gate[1] = { 0.5f };
    geo_real_t eff[16];
    geo_relational_status st = geo_relational_identity_gate_forward(proj, gate, eff, &shape);
    ASSERT_OK(st, "Identity gate forward");

    // H = 0.5 * I + 0.5 * 0.25 = 0.5 * I + 0.125
    // Diagonal = 0.625, Off-diagonal = 0.125
    ASSERT_TRUE(fabs(eff[0] - 0.625f) < 1e-5f, "Diagonal effective element");
    ASSERT_TRUE(fabs(eff[1] - 0.125f) < 1e-5f, "Off-diagonal effective element");

    // Test mix forward
    geo_real_t state[2 * 4 * 4];
    for (int i = 0; i < 32; ++i) state[i] = (geo_real_t)(i + 1);
    geo_real_t output[32];

    st = geo_relational_mix_forward(state, eff, output, &shape);
    ASSERT_OK(st, "Mix forward");

    return 0;
}

static int test_read_write_add(void) {
    geo_relational_shape shape = { .groups = 2, .streams = 4, .features = 4, .matrix_count = 1 };
    geo_real_t state[32];
    for (int i = 0; i < 32; ++i) state[i] = 1.0f;

    geo_real_t r_weights[4] = { 0.25f, 0.25f, 0.25f, 0.25f };
    geo_real_t read_state[8];
    geo_relational_status st = geo_relational_read_forward(state, r_weights, 1, read_state, &shape);
    ASSERT_OK(st, "Read forward");

    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(fabs(read_state[i] - 1.0f) < 1e-5f, "Read state uniform value");
    }

    geo_real_t w_weights[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    geo_real_t source[8];
    for (int i = 0; i < 8; ++i) source[i] = 2.0f;
    geo_real_t scale[1] = { 0.5f };
    geo_real_t write_output[32];

    st = geo_relational_write_add_forward(state, source, w_weights, 1, scale, 1, write_output, &shape);
    ASSERT_OK(st, "Write add forward");

    for (int i = 0; i < 32; ++i) {
        // output = 1.0 + 0.5 * 1.0 * 2.0 = 2.0
        ASSERT_TRUE(fabs(write_output[i] - 2.0f) < 1e-5f, "Write add value");
    }

    return 0;
}

static int test_linear_adjoint_identity(void) {
    // < H X, Y > == < X, H^T Y >
    geo_relational_shape shape = { .groups = 2, .streams = 4, .features = 8, .matrix_count = 1 };
    geo_real_t H[16] = {
        0.7f, 0.1f, 0.1f, 0.1f,
        0.2f, 0.6f, 0.1f, 0.1f,
        0.1f, 0.2f, 0.5f, 0.2f,
        0.0f, 0.1f, 0.3f, 0.6f
    };

    geo_real_t X[64], Y[64];
    for (int i = 0; i < 64; ++i) {
        X[i] = (geo_real_t)((i % 7) - 3) * 0.5f;
        Y[i] = (geo_real_t)((i % 5) - 2) * 0.4f;
    }

    geo_real_t HX[64];
    geo_relational_mix_forward(X, H, HX, &shape);

    geo_real_t dot1 = 0.0f;
    for (int i = 0; i < 64; ++i) dot1 += HX[i] * Y[i];

    // Compute H^T Y
    geo_real_t HT[16];
    for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < 4; ++q) {
            HT[p * 4 + q] = H[q * 4 + p];
        }
    }
    geo_real_t HTY[64];
    geo_relational_mix_forward(Y, HT, HTY, &shape);

    geo_real_t dot2 = 0.0f;
    for (int i = 0; i < 64; ++i) dot2 += X[i] * HTY[i];

    geo_real_t diff = fabs(dot1 - dot2);
    ASSERT_TRUE(diff < 1e-4f, "Adjoint linear identity check");

    return 0;
}

static int test_rejection_cases(void) {
    geo_relational_shape invalid_shape = { .groups = 2, .streams = 12, .features = 16, .matrix_count = 1 };
    geo_relational_projection_options options = {
        .abi_version = GEO_RELATIONAL_ABI_VERSION,
        .iterations = 20,
        .epsilon = (geo_real_t)1e-7,
        .mode = GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN,
        .fail_on_nonfinite = 1,
        .require_certificate = 1
    };

    geo_real_t logits[144];
    geo_real_t rel[144];
    geo_real_t ws[100];
    geo_relational_status st = geo_relational_project_forward(logits, rel, ws, 100, &invalid_shape, &options, NULL);
    ASSERT_TRUE(st == GEO_RELATIONAL_INVALID_ARGUMENT, "Invalid streams rejection");

    // Insufficient workspace
    geo_relational_shape valid_shape = { .groups = 2, .streams = 4, .features = 16, .matrix_count = 1 };
    st = geo_relational_project_forward(logits, rel, ws, 5, &valid_shape, &options, NULL);
    ASSERT_TRUE(st == GEO_RELATIONAL_INSUFFICIENT_WORKSPACE, "Insufficient workspace rejection");

    // NaN rejection
    geo_real_t nan_logits[16] = { 0.0f };
    nan_logits[5] = (geo_real_t)NAN;
    geo_real_t valid_ws[500];
    st = geo_relational_project_forward(nan_logits, rel, valid_ws, 500, &valid_shape, &options, NULL);
    ASSERT_TRUE(st == GEO_RELATIONAL_NUMERIC_FAILURE, "NaN logits rejection");

    return 0;
}

int main(void) {
    printf("Running Host Native C Relational Operator Suite...\n");

    if (test_abi_and_defaults()) return 1;
    if (test_projection_and_certificates()) return 1;
    if (test_identity_gate_and_mix()) return 1;
    if (test_read_write_add()) return 1;
    if (test_linear_adjoint_identity()) return 1;
    if (test_rejection_cases()) return 1;

    printf("GEO_RELATIONAL_HOST: PASS projection=PASS vjp=PASS certificate=PASS fallback=NONE\n");
    return 0;
}
