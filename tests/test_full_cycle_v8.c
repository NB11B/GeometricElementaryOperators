#include "geo/full_cycle_v8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 0; } } while (0)
#define CHECK_MAIN(c) do { if (!(c)) return EXIT_FAILURE; } while (0)

static void make_signature(uint8_t n, uint8_t q, int8_t *signature) {
    uint8_t i;
    for (i = 0u; i < GEO_OPERATOR_MAX_DIMENSION; ++i) {
        signature[i] = i < n ? (i < (uint8_t)(n - q) ? 1 : -1) : 0;
    }
}

static void mv_init(
    geo_operator_mv_f64_t *value,
    uint8_t n,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = n;
    memcpy(value->signature, signature, n);
}

static void mv_fill(
    geo_operator_mv_f64_t *value,
    uint8_t n,
    const int8_t *signature,
    unsigned seed
) {
    size_t blade;
    const size_t count = (size_t)1u << n;
    mv_init(value, n, signature);
    for (blade = 0u; blade < count; ++blade) {
        const unsigned mixed = (unsigned)(blade + 1u) * (seed * 31u + 17u) + seed * 7u;
        value->coefficients[blade] = (double)((int)(mixed % 29u) - 14) / 23.0;
    }
}

static double mv_dot(
    const geo_operator_mv_f64_t *a,
    const geo_operator_mv_f64_t *b
) {
    size_t blade;
    double value = 0.0;
    const size_t count = (size_t)1u << a->dimension;
    for (blade = 0u; blade < count; ++blade) {
        value += a->coefficients[blade] * b->coefficients[blade];
    }
    return value;
}

static int near(double a, double b, double tolerance) {
    return fabs(a - b) <= tolerance * (1.0 + fabs(a) + fabs(b));
}

static int add_loss_tail(
    geo_v8_program_t *program,
    geo_v8_node_id_t prediction,
    geo_v8_node_id_t target,
    geo_v8_node_id_t *loss
) {
    geo_v8_node_id_t negative_target;
    geo_v8_node_id_t residual;
    CHECK(geo_v8_add_unary(
        program,
        GEO_V8_NODE_SCALE,
        "negative_target",
        target,
        -1.0,
        0u,
        &negative_target
    ) == GEO_V8_OK);
    CHECK(geo_v8_add_binary(
        program,
        GEO_V8_NODE_ADD,
        "residual",
        prediction,
        negative_target,
        &residual
    ) == GEO_V8_OK);
    CHECK(geo_v8_add_unary(
        program,
        GEO_V8_NODE_SQUARED_NORM,
        "loss",
        residual,
        0.0,
        0u,
        loss
    ) == GEO_V8_OK);
    return 1;
}

static double loss_at(
    geo_v8_program_t *program,
    geo_v8_node_id_t parameter,
    const geo_operator_mv_f64_t *value
) {
    const geo_operator_mv_f64_t *loss;
    if (geo_v8_set_value(program, parameter, value) != GEO_V8_OK) return NAN;
    if (geo_v8_forward(program) != GEO_V8_OK) return NAN;
    loss = geo_v8_value(program, program->loss_node);
    return loss == NULL ? NAN : loss->coefficients[0];
}

static int test_all_signatures_composed_gradients(void) {
    uint8_t n;
    unsigned cases = 0u;
    for (n = 1u; n <= GEO_OPERATOR_MAX_DIMENSION; ++n) {
        uint8_t q;
        for (q = 0u; q <= n; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            geo_operator_mv_f64_t input;
            geo_operator_mv_f64_t target;
            geo_operator_mv_f64_t w1;
            geo_operator_mv_f64_t w2;
            geo_operator_mv_f64_t direction1;
            geo_operator_mv_f64_t direction2;
            geo_operator_mv_f64_t constant;
            geo_operator_mv_f64_t plus;
            geo_operator_mv_f64_t minus;
            geo_operator_mv_f64_t gradient1;
            geo_operator_mv_f64_t gradient2;
            geo_v8_program_t program;
            geo_v8_node_id_t x;
            geo_v8_node_id_t y;
            geo_v8_node_id_t p1;
            geo_v8_node_id_t p2;
            geo_v8_node_id_t c0;
            geo_v8_node_id_t gp;
            geo_v8_node_id_t activation;
            geo_v8_node_id_t gate;
            geo_v8_node_id_t gated;
            geo_v8_node_id_t reversed;
            geo_v8_node_id_t involuted;
            geo_v8_node_id_t conjugated;
            geo_v8_node_id_t projected;
            geo_v8_node_id_t shifted;
            geo_v8_node_id_t normalized;
            geo_v8_node_id_t loss;
            const double epsilon = 1e-6;
            double analytic;
            double numeric;
            size_t blade;
            const size_t count = (size_t)1u << n;

            make_signature(n, q, signature);
            mv_fill(&input, n, signature, 3u + n + q);
            mv_fill(&target, n, signature, 11u + n * 2u + q);
            mv_fill(&w1, n, signature, 19u + n + q);
            mv_fill(&w2, n, signature, 29u + n + q);
            mv_fill(&direction1, n, signature, 37u + n + q);
            mv_fill(&direction2, n, signature, 43u + n + q);
            mv_init(&constant, n, signature);
            constant.coefficients[0] = 1.25;

            CHECK(geo_v8_program_init(
                &program,
                n,
                signature,
                GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN,
                4u
            ) == GEO_V8_OK);
            CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_INPUT, "x", &input, 1, GEO_V8_CONSTRAINT_NONE, &x) == GEO_V8_OK);
            CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_TARGET, "target", &target, 0, GEO_V8_CONSTRAINT_NONE, &y) == GEO_V8_OK);
            CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_PARAMETER, "w1", &w1, 1, GEO_V8_CONSTRAINT_NONE, &p1) == GEO_V8_OK);
            CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_PARAMETER, "w2", &w2, 1, GEO_V8_CONSTRAINT_NONE, &p2) == GEO_V8_OK);
            CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_CONSTANT, "bias", &constant, 0, GEO_V8_CONSTRAINT_NONE, &c0) == GEO_V8_OK);
            CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_GEOMETRIC_PRODUCT, "gp", x, p1, &gp) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_TANH, "activation", gp, 0.0, 0u, &activation) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SIGMOID, "gate", p2, 0.0, 0u, &gate) == GEO_V8_OK);
            CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_HADAMARD, "gated", activation, gate, &gated) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_REVERSE, "reversed", gated, 0.0, 0u, &reversed) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_GRADE_INVOLUTION, "involuted", reversed, 0.0, 0u, &involuted) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_CLIFFORD_CONJUGATE, "conjugated", involuted, 0.0, 0u, &conjugated) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_GRADE_PROJECT, "projected", conjugated, 0.0, (uint8_t)(q % (n + 1u)), &projected) == GEO_V8_OK);
            CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_ADD, "shifted", projected, c0, &shifted) == GEO_V8_OK);
            CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_EUCLIDEAN_NORMALIZE, "normalized", shifted, 1e-12, 0u, &normalized) == GEO_V8_OK);
            CHECK(add_loss_tail(&program, normalized, y, &loss));
            CHECK(geo_v8_compile(&program, loss) == GEO_V8_OK);
            CHECK(geo_v8_forward(&program) == GEO_V8_OK);
            CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
            gradient1 = *geo_v8_gradient(&program, p1);
            gradient2 = *geo_v8_gradient(&program, p2);

            plus = w1;
            minus = w1;
            for (blade = 0u; blade < count; ++blade) {
                plus.coefficients[blade] += epsilon * direction1.coefficients[blade];
                minus.coefficients[blade] -= epsilon * direction1.coefficients[blade];
            }
            numeric = (loss_at(&program, p1, &plus) - loss_at(&program, p1, &minus)) / (2.0 * epsilon);
            analytic = mv_dot(&gradient1, &direction1);
            CHECK(isfinite(numeric));
            CHECK(near(numeric, analytic, 3e-5));
            CHECK(geo_v8_set_value(&program, p1, &w1) == GEO_V8_OK);

            plus = w2;
            minus = w2;
            for (blade = 0u; blade < count; ++blade) {
                plus.coefficients[blade] += epsilon * direction2.coefficients[blade];
                minus.coefficients[blade] -= epsilon * direction2.coefficients[blade];
            }
            numeric = (loss_at(&program, p2, &plus) - loss_at(&program, p2, &minus)) / (2.0 * epsilon);
            analytic = mv_dot(&gradient2, &direction2);
            CHECK(isfinite(numeric));
            CHECK(near(numeric, analytic, 3e-5));
            geo_v8_program_free(&program);
            ++cases;
        }
    }
    CHECK(cases == 27u);
    return 1;
}

static int build_simple_gp(
    geo_v8_program_t *program,
    const int8_t *signature,
    geo_v8_node_id_t *x,
    geo_v8_node_id_t *target,
    geo_v8_node_id_t *weight,
    geo_v8_node_id_t *loss
) {
    geo_operator_mv_f64_t zero;
    geo_v8_node_id_t prediction;
    mv_init(&zero, 2u, signature);
    CHECK(geo_v8_program_init(program, 2u, signature, GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN, 8u) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(program, GEO_V8_NODE_INPUT, "x", &zero, 0, GEO_V8_CONSTRAINT_NONE, x) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(program, GEO_V8_NODE_TARGET, "target", &zero, 0, GEO_V8_CONSTRAINT_NONE, target) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(program, GEO_V8_NODE_PARAMETER, "weight", &zero, 1, GEO_V8_CONSTRAINT_NONE, weight) == GEO_V8_OK);
    CHECK(geo_v8_add_binary(program, GEO_V8_NODE_GEOMETRIC_PRODUCT, "prediction", *x, *weight, &prediction) == GEO_V8_OK);
    CHECK(add_loss_tail(program, prediction, *target, loss));
    CHECK(geo_v8_compile(program, *loss) == GEO_V8_OK);
    return 1;
}

static int set_sample(
    geo_v8_program_t *program,
    geo_v8_node_id_t x,
    geo_v8_node_id_t target,
    const geo_operator_mv_f64_t *xv,
    const geo_operator_mv_f64_t *yv
) {
    CHECK(geo_v8_set_value(program, x, xv) == GEO_V8_OK);
    CHECK(geo_v8_set_value(program, target, yv) == GEO_V8_OK);
    CHECK(geo_v8_forward(program) == GEO_V8_OK);
    return 1;
}

static int test_batch_accumulation(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    geo_v8_program_t program;
    geo_v8_node_id_t x;
    geo_v8_node_id_t target;
    geo_v8_node_id_t weight;
    geo_v8_node_id_t loss;
    geo_operator_mv_f64_t x1;
    geo_operator_mv_f64_t x2;
    geo_operator_mv_f64_t y1;
    geo_operator_mv_f64_t y2;
    geo_operator_mv_f64_t g1;
    geo_operator_mv_f64_t g2;
    geo_operator_mv_f64_t sum;
    size_t blade;
    CHECK(build_simple_gp(&program, signature, &x, &target, &weight, &loss));
    mv_fill(&x1, 2u, signature, 3u);
    mv_fill(&x2, 2u, signature, 5u);
    mv_fill(&y1, 2u, signature, 7u);
    mv_fill(&y2, 2u, signature, 11u);

    CHECK(set_sample(&program, x, target, &x1, &y1));
    CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
    g1 = *geo_v8_gradient(&program, weight);
    CHECK(set_sample(&program, x, target, &x2, &y2));
    CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
    g2 = *geo_v8_gradient(&program, weight);

    CHECK(set_sample(&program, x, target, &x1, &y1));
    CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
    CHECK(set_sample(&program, x, target, &x2, &y2));
    CHECK(geo_v8_backward(&program, 1) == GEO_V8_OK);
    sum = *geo_v8_gradient(&program, weight);
    for (blade = 0u; blade < 4u; ++blade) {
        CHECK(near(sum.coefficients[blade], g1.coefficients[blade] + g2.coefficients[blade], 1e-12));
    }
    CHECK(program.gradient_samples == 2u);
    CHECK(geo_v8_optimizer_step(&program, GEO_V8_OPTIMIZER_SGD, 0.01, 0.9, 0.999, 1e-8, 0.5) == GEO_V8_OK);
    CHECK(program.gradient_samples == 0u);
    geo_v8_program_free(&program);
    return 1;
}

static int test_dynamic_graph_scale(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 0, 0, 0, 0, 0};
    geo_operator_mv_f64_t value;
    geo_v8_program_t program;
    geo_v8_node_id_t current;
    geo_v8_node_id_t next;
    geo_v8_node_id_t loss;
    unsigned i;
    mv_init(&value, 1u, signature);
    value.coefficients[0] = 0.25;
    value.coefficients[1] = -0.5;
    CHECK(geo_v8_program_init(&program, 1u, signature, GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN, 1u) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_INPUT, "root", &value, 1, GEO_V8_CONSTRAINT_NONE, &current) == GEO_V8_OK);
    for (i = 0u; i < 600u; ++i) {
        char name[32];
        CHECK(snprintf(name, sizeof(name), "scale_%u", i) > 0);
        CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SCALE, name, current, 0.999, 0u, &next) == GEO_V8_OK);
        current = next;
    }
    CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SQUARED_NORM, "large_loss", current, 0.0, 0u, &loss) == GEO_V8_OK);
    CHECK(program.node_count == 602u);
    CHECK(program.node_capacity >= 602u);
    CHECK(geo_v8_compile(&program, loss) == GEO_V8_OK);
    CHECK(geo_v8_forward(&program) == GEO_V8_OK);
    CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
    CHECK(geo_v8_gradient(&program, 0u) != NULL);
    geo_v8_program_free(&program);
    return 1;
}

static int test_stateful_recurrence(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    geo_operator_mv_f64_t zero;
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t target;
    geo_operator_mv_f64_t weight_value;
    geo_v8_program_t program;
    geo_v8_node_id_t x;
    geo_v8_node_id_t y;
    geo_v8_node_id_t state;
    geo_v8_node_id_t weight;
    geo_v8_node_id_t transformed;
    geo_v8_node_id_t preactivation;
    geo_v8_node_id_t next_state;
    geo_v8_node_id_t loss;
    geo_operator_mv_f64_t before;
    geo_operator_mv_f64_t after;
    mv_init(&zero, 2u, signature);
    mv_fill(&input, 2u, signature, 3u);
    mv_fill(&target, 2u, signature, 5u);
    mv_fill(&weight_value, 2u, signature, 7u);
    CHECK(geo_v8_program_init(&program, 2u, signature, GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN, 16u) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_INPUT, "x", &input, 0, GEO_V8_CONSTRAINT_NONE, &x) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_TARGET, "target", &target, 0, GEO_V8_CONSTRAINT_NONE, &y) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_STATE, "memory", &zero, 0, GEO_V8_CONSTRAINT_NONE, &state) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_PARAMETER, "weight", &weight_value, 1, GEO_V8_CONSTRAINT_NONE, &weight) == GEO_V8_OK);
    CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_GEOMETRIC_PRODUCT, "transformed", x, weight, &transformed) == GEO_V8_OK);
    CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_ADD, "preactivation", transformed, state, &preactivation) == GEO_V8_OK);
    CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_TANH, "next_memory", preactivation, 0.0, 0u, &next_state) == GEO_V8_OK);
    CHECK(add_loss_tail(&program, next_state, y, &loss));
    CHECK(geo_v8_bind_state_update(&program, state, next_state) == GEO_V8_OK);
    CHECK(geo_v8_compile(&program, loss) == GEO_V8_OK);
    before = *geo_v8_value(&program, state);
    CHECK(geo_v8_forward(&program) == GEO_V8_OK);
    CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
    CHECK(geo_v8_commit_states(&program) == GEO_V8_OK);
    after = *geo_v8_value(&program, state);
    CHECK(!near(mv_dot(&before, &before), mv_dot(&after, &after), 1e-12));
    CHECK(geo_v8_optimizer_step(&program, GEO_V8_OPTIMIZER_ADAM, 0.01, 0.9, 0.999, 1e-8, 1.0) == GEO_V8_OK);
    geo_v8_program_free(&program);
    return 1;
}

static int test_constraints_and_transactionality(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    geo_operator_mv_f64_t zero;
    geo_operator_mv_f64_t euclidean;
    geo_operator_mv_f64_t vector;
    geo_operator_mv_f64_t versor;
    geo_v8_program_t program;
    geo_v8_node_id_t p1;
    geo_v8_node_id_t p2;
    geo_v8_node_id_t p3;
    geo_v8_node_id_t z1;
    geo_v8_node_id_t z2;
    geo_v8_node_id_t z3;
    geo_v8_node_id_t a1;
    geo_v8_node_id_t a2;
    geo_v8_node_id_t loss;
    double norm2;
    mv_init(&zero, 2u, signature);
    mv_init(&euclidean, 2u, signature);
    mv_init(&vector, 2u, signature);
    mv_init(&versor, 2u, signature);
    euclidean.coefficients[0] = 3.0;
    euclidean.coefficients[1] = 4.0;
    vector.coefficients[1] = 2.0;
    vector.coefficients[2] = 1.0;
    versor.coefficients[0] = 2.0;
    CHECK(geo_v8_program_init(&program, 2u, signature, GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN, 16u) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_PARAMETER, "unit", &euclidean, 1, GEO_V8_CONSTRAINT_UNIT_EUCLIDEAN, &p1) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_PARAMETER, "vector", &vector, 1, GEO_V8_CONSTRAINT_UNIT_VECTOR_METRIC, &p2) == GEO_V8_OK);
    CHECK(geo_v8_add_leaf(&program, GEO_V8_NODE_PARAMETER, "versor", &versor, 1, GEO_V8_CONSTRAINT_EVEN_VERSOR, &p3) == GEO_V8_OK);
    CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SCALE, "zero1", p1, 0.0, 0u, &z1) == GEO_V8_OK);
    CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SCALE, "zero2", p2, 0.0, 0u, &z2) == GEO_V8_OK);
    CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SCALE, "zero3", p3, 0.0, 0u, &z3) == GEO_V8_OK);
    CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_ADD, "sum1", z1, z2, &a1) == GEO_V8_OK);
    CHECK(geo_v8_add_binary(&program, GEO_V8_NODE_ADD, "sum2", a1, z3, &a2) == GEO_V8_OK);
    CHECK(geo_v8_add_unary(&program, GEO_V8_NODE_SQUARED_NORM, "constraint_loss", a2, 0.0, 0u, &loss) == GEO_V8_OK);
    CHECK(geo_v8_compile(&program, loss) == GEO_V8_OK);
    CHECK(geo_v8_forward(&program) == GEO_V8_OK);
    CHECK(geo_v8_backward(&program, 0) == GEO_V8_OK);
    CHECK(geo_v8_optimizer_step(&program, GEO_V8_OPTIMIZER_SGD, 0.1, 0.9, 0.999, 1e-8, 1.0) == GEO_V8_OK);
    norm2 = mv_dot(geo_v8_value(&program, p1), geo_v8_value(&program, p1));
    CHECK(near(norm2, 1.0, 1e-12));
    CHECK(near(
        geo_v8_value(&program, p2)->coefficients[1] * geo_v8_value(&program, p2)->coefficients[1] +
        geo_v8_value(&program, p2)->coefficients[2] * geo_v8_value(&program, p2)->coefficients[2],
        1.0,
        1e-12
    ));
    CHECK(near(fabs(geo_v8_value(&program, p3)->coefficients[0]), 1.0, 1e-12));
    geo_v8_program_free(&program);

    {
        geo_v8_program_t bad;
        geo_v8_node_id_t parameter;
        geo_v8_node_id_t scaled;
        geo_v8_node_id_t bad_loss;
        geo_operator_mv_f64_t preserved;
        CHECK(geo_v8_program_init(&bad, 2u, signature, GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN, 4u) == GEO_V8_OK);
        CHECK(geo_v8_add_leaf(&bad, GEO_V8_NODE_PARAMETER, "null_vector", &zero, 1, GEO_V8_CONSTRAINT_UNIT_VECTOR_METRIC, &parameter) == GEO_V8_OK);
        CHECK(geo_v8_add_unary(&bad, GEO_V8_NODE_SCALE, "zero", parameter, 0.0, 0u, &scaled) == GEO_V8_OK);
        CHECK(geo_v8_add_unary(&bad, GEO_V8_NODE_SQUARED_NORM, "loss", scaled, 0.0, 0u, &bad_loss) == GEO_V8_OK);
        CHECK(geo_v8_compile(&bad, bad_loss) == GEO_V8_OK);
        CHECK(geo_v8_forward(&bad) == GEO_V8_OK);
        CHECK(geo_v8_backward(&bad, 0) == GEO_V8_OK);
        preserved = *geo_v8_value(&bad, parameter);
        CHECK(geo_v8_optimizer_step(&bad, GEO_V8_OPTIMIZER_SGD, 0.1, 0.9, 0.999, 1e-8, 1.0) == GEO_V8_CONSTRAINT_FAILURE);
        CHECK(memcmp(&preserved, geo_v8_value(&bad, parameter), sizeof(preserved)) == 0);
        geo_v8_program_free(&bad);
    }
    return 1;
}

int main(void) {
    CHECK_MAIN(geo_v8_abi_version() == GEO_V8_ABI_VERSION);
    CHECK_MAIN(test_all_signatures_composed_gradients());
    CHECK_MAIN(test_batch_accumulation());
    CHECK_MAIN(test_dynamic_graph_scale());
    CHECK_MAIN(test_stateful_recurrence());
    CHECK_MAIN(test_constraints_and_transactionality());
    puts(
        "GEO_FULL_CYCLE_V8_TEST: PASS signatures=27 arbitrary_graph=PASS "
        "multiple_parameters=PASS nonlinear_adjoints=PASS batching=PASS "
        "stateful_recurrence=PASS dynamic_nodes=602 constraints=PASS "
        "external_autograd=NONE cuda=DEFERRED"
    );
    return EXIT_SUCCESS;
}
