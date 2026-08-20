#include "geo/cl30_morphism.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int run_test(void) {
    geo_cl30_morphism_shape shape = { 2, 4, 3, 6, 2 }; /* B=2, N=4, E=3, R=6, C=2 */

    size_t ws_bytes = geo_cl30_morphism_workspace_bytes(&shape);
    void* ws = malloc(ws_bytes);
    if (!ws) {
        printf("Failed to allocate workspace\n");
        return 1;
    }

    float bivectors[6 * 2 * 3];
    for (int i = 0; i < 36; ++i) bivectors[i] = ((float)rand() / (float)RAND_MAX) * 0.4f - 0.2f;

    float node_states[2 * 4 * 2 * 3];
    for (int i = 0; i < 48; ++i) node_states[i] = ((float)rand() / (float)RAND_MAX) * 1.0f - 0.5f;

    float goal_queries[2 * 2 * 3];
    for (int i = 0; i < 12; ++i) goal_queries[i] = ((float)rand() / (float)RAND_MAX) * 1.0f - 0.5f;

    float app_params[3] = { 1.0f, 0.5f, 0.1f };
    float comp_params[5] = { 1.0f, 0.5413f, 0.5f, 0.3f, 0.0f };

    int32_t edge_batch[3] = { 0, 0, 1 };
    int32_t edge_source[3] = { 0, 1, 2 };
    int32_t edge_relation[3] = { 1, 3, 0 };
    int32_t edge_destination[3] = { 1, 1, 3 };

    float node_aggregates[2 * 4 * 2 * 3];
    geo_cl30_morphism_telemetry tel;

    geo_cl30_morphism_status st = geo_cl30_morphism_forward_f32(
        &shape, bivectors, node_states, goal_queries, app_params, comp_params,
        edge_batch, edge_source, edge_relation, edge_destination, 0,
        node_aggregates, NULL, NULL, NULL, ws, ws_bytes, &tel
    );

    if (st != GEO_CL30_MORPHISM_OK) {
        printf("Forward failed with status %d\n", (int)st);
        free(ws);
        return 1;
    }

    printf("Forward PASS! Telemetry: alpha_mean=%.4f, beta_mean=%.4f, gate_prod_mean=%.4f\n",
           tel.alpha_mean, tel.beta_mean, tel.gate_product_mean);

    /* Test VJP with finite differences */
    float grad_agg[48];
    for (int i = 0; i < 48; ++i) grad_agg[i] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;

    float g_biv[36], g_nodes[48], g_goals[12], g_app[3], g_comp[5];
    st = geo_cl30_morphism_vjp_f32(
        &shape, bivectors, node_states, goal_queries, app_params, comp_params,
        edge_batch, edge_source, edge_relation, edge_destination,
        grad_agg, g_biv, g_nodes, g_goals, g_app, g_comp, ws, ws_bytes
    );

    if (st != GEO_CL30_MORPHISM_OK) {
        printf("VJP failed with status %d\n", (int)st);
        free(ws);
        return 1;
    }

    /* Finite difference check on app_params[0] */
    float eps = 1e-4f;
    float app_p[3], app_m[3];
    memcpy(app_p, app_params, sizeof(app_params));
    memcpy(app_m, app_params, sizeof(app_params));
    app_p[0] += eps;
    app_m[0] -= eps;

    float agg_p[48], agg_m[48];
    geo_cl30_morphism_forward_f32(&shape, bivectors, node_states, goal_queries, app_p, comp_params,
                                  edge_batch, edge_source, edge_relation, edge_destination, 0,
                                  agg_p, NULL, NULL, NULL, ws, ws_bytes, NULL);
    geo_cl30_morphism_forward_f32(&shape, bivectors, node_states, goal_queries, app_m, comp_params,
                                  edge_batch, edge_source, edge_relation, edge_destination, 0,
                                  agg_m, NULL, NULL, NULL, ws, ws_bytes, NULL);

    float num_grad_app0 = 0.0f;
    for (int i = 0; i < 48; ++i) {
        num_grad_app0 += grad_agg[i] * (agg_p[i] - agg_m[i]) / (2.0f * eps);
    }

    float diff_app0 = (float)fabs((double)(g_app[0] - num_grad_app0));
    printf("App[0] Grad: Analytic=%.6f, Numeric=%.6f, Diff=%.6e\n", g_app[0], num_grad_app0, diff_app0);

    /* Finite difference check on comp_params[0] */
    float comp_p[5], comp_m[5];
    memcpy(comp_p, comp_params, sizeof(comp_params));
    memcpy(comp_m, comp_params, sizeof(comp_params));
    comp_p[0] += eps;
    comp_m[0] -= eps;

    geo_cl30_morphism_forward_f32(&shape, bivectors, node_states, goal_queries, app_params, comp_p,
                                  edge_batch, edge_source, edge_relation, edge_destination, 0,
                                  agg_p, NULL, NULL, NULL, ws, ws_bytes, NULL);
    geo_cl30_morphism_forward_f32(&shape, bivectors, node_states, goal_queries, app_params, comp_m,
                                  edge_batch, edge_source, edge_relation, edge_destination, 0,
                                  agg_m, NULL, NULL, NULL, ws, ws_bytes, NULL);

    float num_grad_comp0 = 0.0f;
    for (int i = 0; i < 48; ++i) {
        num_grad_comp0 += grad_agg[i] * (agg_p[i] - agg_m[i]) / (2.0f * eps);
    }

    float diff_comp0 = (float)fabs((double)(g_comp[0] - num_grad_comp0));
    printf("Comp[0] Grad: Analytic=%.6f, Numeric=%.6f, Diff=%.6e\n", g_comp[0], num_grad_comp0, diff_comp0);

    free(ws);
    if (diff_app0 > 1e-3f || diff_comp0 > 1e-3f) {
        printf("Grad check FAIL!\n");
        return 1;
    }

    printf("Grad check PASS!\n");
    return 0;
}

int main(void) {
    return run_test();
}
