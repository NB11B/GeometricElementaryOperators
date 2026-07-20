#include "geo/batch_gp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 0; } } while (0)

static void init_mv(geo_operator_mv_f64_t *v, uint8_t n, const int8_t *sig) {
    memset(v, 0, sizeof(*v));
    v->dimension = n;
    memcpy(v->signature, sig, n);
}

static void fill(double *v, size_t count, unsigned seed) {
    size_t i;
    for (i = 0u; i < count; ++i) {
        v[i] = (double)((int)((i + 1u) * (seed * 13u + 7u) % 29u) - 14) / 9.0;
    }
}

static int near(double a, double b) {
    return fabs(a - b) <= 1e-11 * (1.0 + fabs(a) + fabs(b));
}

static int test_signature(uint8_t n, uint8_t q) {
    int8_t sig[GEO_OPERATOR_MAX_DIMENSION] = {0};
    geo_batch_gp_plan_t plan;
    double inputs[8u * GEO_OPERATOR_MAX_BLADES];
    double outputs[8u * GEO_OPERATOR_MAX_BLADES];
    double bars[8u * GEO_OPERATOR_MAX_BLADES];
    double parameter[GEO_OPERATOR_MAX_BLADES];
    double parameter_bar[GEO_OPERATOR_MAX_BLADES];
    double reference_bar[GEO_OPERATOR_MAX_BLADES] = {0};
    double targets[8u * GEO_OPERATOR_MAX_BLADES];
    double parameter_step[GEO_OPERATOR_MAX_BLADES];
    size_t sample;
    size_t blade;
    const size_t blades = (size_t)1u << n;
    const size_t batch = 8u;
    uint8_t axis;

    for (axis = 0u; axis < n; ++axis) sig[axis] = axis < (uint8_t)(n - q) ? 1 : -1;
    CHECK(geo_batch_gp_plan_init(&plan, n, sig) == GEO_BATCH_GP_OK);
    fill(inputs, batch * blades, (unsigned)n + q + 5u);
    fill(parameter, blades, (unsigned)n * 3u + q + 9u);
    fill(bars, batch * blades, (unsigned)n * 5u + q + 11u);

    CHECK(geo_batch_gp_right_forward_f64(&plan, inputs, batch, parameter, outputs) == GEO_BATCH_GP_OK);
    for (sample = 0u; sample < batch; ++sample) {
        geo_operator_mv_f64_t x;
        geo_operator_mv_f64_t w;
        geo_operator_mv_f64_t y;
        init_mv(&x, n, sig);
        init_mv(&w, n, sig);
        memcpy(x.coefficients, inputs + sample * blades, blades * sizeof(double));
        memcpy(w.coefficients, parameter, blades * sizeof(double));
        CHECK(geo_operator_gp_f64(&x, &w, &y) == GEO_OPERATOR_OK);
        for (blade = 0u; blade < blades; ++blade) {
            CHECK(near(outputs[sample * blades + blade], y.coefficients[blade]));
        }
    }

    CHECK(geo_batch_gp_right_vjp_parameter_f64(&plan, inputs, bars, batch, parameter_bar) == GEO_BATCH_GP_OK);
    for (sample = 0u; sample < batch; ++sample) {
        geo_operator_mv_f64_t x;
        geo_operator_mv_f64_t w;
        geo_operator_mv_f64_t bar_y;
        geo_operator_mv_f64_t bar_x;
        geo_operator_mv_f64_t bar_w;
        init_mv(&x, n, sig);
        init_mv(&w, n, sig);
        init_mv(&bar_y, n, sig);
        memcpy(x.coefficients, inputs + sample * blades, blades * sizeof(double));
        memcpy(w.coefficients, parameter, blades * sizeof(double));
        memcpy(bar_y.coefficients, bars + sample * blades, blades * sizeof(double));
        CHECK(geo_operator_gp_f64_vjp(&x, &w, &bar_y, &bar_x, &bar_w) == GEO_OPERATOR_OK);
        for (blade = 0u; blade < blades; ++blade) reference_bar[blade] += bar_w.coefficients[blade];
    }
    for (blade = 0u; blade < blades; ++blade) CHECK(near(parameter_bar[blade], reference_bar[blade]));

    memcpy(targets, outputs, batch * blades * sizeof(double));
    memcpy(parameter_step, parameter, blades * sizeof(double));
    {
        double loss = -1.0;
        CHECK(geo_batch_gp_right_mse_sgd_step_f64(
            &plan,
            inputs,
            targets,
            batch,
            0.01,
            parameter_step,
            &loss
        ) == GEO_BATCH_GP_OK);
        CHECK(near(loss, 0.0));
        for (blade = 0u; blade < blades; ++blade) CHECK(near(parameter_step[blade], parameter[blade]));
    }

    CHECK(geo_batch_gp_left_forward_f64(&plan, parameter, inputs, batch, outputs) == GEO_BATCH_GP_OK);
    CHECK(geo_batch_gp_left_vjp_parameter_f64(&plan, inputs, bars, batch, parameter_bar) == GEO_BATCH_GP_OK);
    return 1;
}

int main(void) {
    uint8_t n;
    unsigned signatures = 0u;
    CHECK(geo_batch_gp_abi_version() == GEO_BATCH_GP_ABI_VERSION);
    for (n = 1u; n <= GEO_OPERATOR_MAX_DIMENSION; ++n) {
        uint8_t q;
        for (q = 0u; q <= n; ++q) {
            if (!test_signature(n, q)) return EXIT_FAILURE;
            ++signatures;
        }
    }
    printf("GEO_BATCH_GP_TEST: PASS signatures=%u batch=8 forward=PASS vjp=PASS fused_sgd=PASS\n", signatures);
    return EXIT_SUCCESS;
}
