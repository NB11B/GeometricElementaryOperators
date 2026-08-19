#include "geo/cl30_incidence.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static float rand_f(float min_val, float max_val) {
    return min_val + ((float)rand() / (float)RAND_MAX) * (max_val - min_val);
}

int main(void) {
    printf("=== Test Cl(3,0) Incidence Native Host Suite ===\n");
    srand(42);

    geo_cl30_incidence_shape shape = {
        .batch_count = 2,
        .node_count = 4,
        .edge_count = 6,
        .relation_count = 6,
        .channel_count = 32
    };

    size_t ws_bytes = geo_cl30_incidence_workspace_bytes(&shape);
    assert(ws_bytes > 0);
    void* workspace = malloc(ws_bytes);
    assert(workspace != NULL);

    // Allocate inputs
    size_t rel_bivec_count = (size_t)shape.relation_count * shape.channel_count * 3;
    float* rel_bivectors = (float*)malloc(rel_bivec_count * sizeof(float));
    for (size_t i = 0; i < rel_bivec_count; ++i) {
        rel_bivectors[i] = rand_f(-1.5f, 1.5f);
    }

    size_t node_count_total = (size_t)shape.batch_count * shape.node_count * shape.channel_count * 3;
    float* node_states = (float*)malloc(node_count_total * sizeof(float));
    for (size_t i = 0; i < node_count_total; ++i) {
        node_states[i] = rand_f(-2.0f, 2.0f);
    }

    int32_t edge_batch[6]       = {0, 0, 0, 1, 1, 1};
    int32_t edge_source[6]      = {0, 1, 2, 0, 2, 3};
    int32_t edge_relation[6]    = {0, 1, 2, 3, 4, 5};
    int32_t edge_destination[6] = {1, 2, 3, 1, 3, 0};

    float inverse_degrees[8] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };

    float* node_aggregates = (float*)malloc(node_count_total * sizeof(float));
    float max_residual = 0.0f;

    // 1. Forward Test
    geo_cl30_incidence_status status = geo_cl30_incidence_forward_f32(
        &shape,
        rel_bivectors,
        node_states,
        edge_batch,
        edge_source,
        edge_relation,
        edge_destination,
        inverse_degrees,
        node_aggregates,
        workspace,
        ws_bytes,
        &max_residual
    );
    assert(status == GEO_CL30_INCIDENCE_OK);
    printf("Forward status: OK, max rotor residual: %e\n", max_residual);

    // 2. VJP Test
    float* grad_node_aggregates = (float*)malloc(node_count_total * sizeof(float));
    for (size_t i = 0; i < node_count_total; ++i) {
        grad_node_aggregates[i] = rand_f(-1.0f, 1.0f);
    }

    float* grad_relation_bivectors = (float*)malloc(rel_bivec_count * sizeof(float));
    float* grad_node_states = (float*)malloc(node_count_total * sizeof(float));

    status = geo_cl30_incidence_vjp_f32(
        &shape,
        rel_bivectors,
        node_states,
        edge_batch,
        edge_source,
        edge_relation,
        edge_destination,
        inverse_degrees,
        grad_node_aggregates,
        grad_relation_bivectors,
        grad_node_states,
        workspace,
        ws_bytes
    );
    assert(status == GEO_CL30_INCIDENCE_OK);
    printf("VJP status: OK\n");

    // 3. Finite Difference Directional Gradient Check
    // Loss L = sum(grad_node_aggregates * node_aggregates)
    float eps = 1e-4f;
    float max_rel_err_b = 0.0f;
    float max_rel_err_x = 0.0f;

    // Check a sample of bivector gradients
    for (int k = 0; k < 10; ++k) {
        size_t idx = (size_t)(rand() % rel_bivec_count);
        float orig = rel_bivectors[idx];

        rel_bivectors[idx] = orig + eps;
        float* out_pos = (float*)malloc(node_count_total * sizeof(float));
        geo_cl30_incidence_forward_f32(&shape, rel_bivectors, node_states, edge_batch, edge_source, edge_relation, edge_destination, inverse_degrees, out_pos, workspace, ws_bytes, NULL);

        rel_bivectors[idx] = orig - eps;
        float* out_neg = (float*)malloc(node_count_total * sizeof(float));
        geo_cl30_incidence_forward_f32(&shape, rel_bivectors, node_states, edge_batch, edge_source, edge_relation, edge_destination, inverse_degrees, out_neg, workspace, ws_bytes, NULL);
        rel_bivectors[idx] = orig;

        float num_grad = 0.0f;
        for (size_t i = 0; i < node_count_total; ++i) {
            num_grad += grad_node_aggregates[i] * (out_pos[i] - out_neg[i]) / (2.0f * eps);
        }
        float ana_grad = grad_relation_bivectors[idx];
        float abs_err = (float)fabs((double)num_grad - (double)ana_grad);
        float rel_err = abs_err / (float)(fabs((double)num_grad) + fabs((double)ana_grad) + 1e-6);
        if (rel_err > max_rel_err_b) max_rel_err_b = rel_err;

        free(out_pos);
        free(out_neg);
    }
    printf("Max bivector relative gradient error: %e\n", max_rel_err_b);
    assert(max_rel_err_b < 1e-3f);

    // Check a sample of node state gradients
    for (int k = 0; k < 10; ++k) {
        size_t idx = (size_t)(rand() % node_count_total);
        float orig = node_states[idx];

        node_states[idx] = orig + eps;
        float* out_pos = (float*)malloc(node_count_total * sizeof(float));
        geo_cl30_incidence_forward_f32(&shape, rel_bivectors, node_states, edge_batch, edge_source, edge_relation, edge_destination, inverse_degrees, out_pos, workspace, ws_bytes, NULL);

        node_states[idx] = orig - eps;
        float* out_neg = (float*)malloc(node_count_total * sizeof(float));
        geo_cl30_incidence_forward_f32(&shape, rel_bivectors, node_states, edge_batch, edge_source, edge_relation, edge_destination, inverse_degrees, out_neg, workspace, ws_bytes, NULL);
        node_states[idx] = orig;

        float num_grad = 0.0f;
        for (size_t i = 0; i < node_count_total; ++i) {
            num_grad += grad_node_aggregates[i] * (out_pos[i] - out_neg[i]) / (2.0f * eps);
        }
        float ana_grad = grad_node_states[idx];
        float abs_err = (float)fabs((double)num_grad - (double)ana_grad);
        float rel_err = abs_err / (float)(fabs((double)num_grad) + fabs((double)ana_grad) + 1e-6);
        if (rel_err > max_rel_err_x) max_rel_err_x = rel_err;

        free(out_pos);
        free(out_neg);
    }
    printf("Max node state relative gradient error: %e\n", max_rel_err_x);
    assert(max_rel_err_x < 1e-3f);

    // Free buffers
    free(workspace);
    free(rel_bivectors);
    free(node_states);
    free(node_aggregates);
    free(grad_node_aggregates);
    free(grad_relation_bivectors);
    free(grad_node_states);

    printf("R4_NATIVE_C_HOST_TESTS: PASS\n");
    printf("R4_NATIVE_VJP_TESTS: PASS\n");
    return 0;
}
