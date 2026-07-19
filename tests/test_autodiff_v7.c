#include "geo/autodiff_v7.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); \
            abort();                                                            \
        }                                                                       \
    } while (0)

static geo_v7_program_t program;

static void make_signature(uint8_t dimension, uint8_t q, int8_t *signature) {
    uint8_t index;
    for (index = 0u; index < GEO_OPERATOR_MAX_DIMENSION; ++index) {
        if (index >= dimension) {
            signature[index] = 0;
        } else {
            signature[index] = index < (uint8_t)(dimension - q) ? 1 : -1;
        }
    }
}

static void init_mv(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static void fill_mv(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature,
    unsigned seed
) {
    size_t blade;
    const size_t blade_count = (size_t)1u << dimension;
    init_mv(value, dimension, signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        const unsigned mixed =
            (unsigned)(blade + 1u) * (seed * 17u + 29u) + seed * 13u;
        value->coefficients[blade] =
            (double)((int)(mixed % 19u) - 9) / 4.0;
    }
}

static double mv_dot(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right
) {
    size_t blade;
    double result = 0.0;
    const size_t blade_count = (size_t)1u << left->dimension;
    for (blade = 0u; blade < blade_count; ++blade) {
        result += left->coefficients[blade] * right->coefficients[blade];
    }
    return result;
}

static void assert_near(double left, double right, double tolerance) {
    CHECK(
        fabs(left - right) <=
        tolerance * (1.0 + fabs(left) + fabs(right))
    );
}

static void assert_mv_near(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    double tolerance
) {
    size_t blade;
    const size_t blade_count = (size_t)1u << left->dimension;
    for (blade = 0u; blade < blade_count; ++blade) {
        assert_near(
            left->coefficients[blade],
            right->coefficients[blade],
            tolerance
        );
    }
}

static unsigned popcount_unsigned(unsigned value) {
    unsigned count = 0u;
    while (value != 0u) {
        value &= value - 1u;
        ++count;
    }
    return count;
}

static void reverse_reference(
    const geo_operator_mv_f64_t *input,
    geo_operator_mv_f64_t *output
) {
    size_t blade;
    const size_t blade_count = (size_t)1u << input->dimension;
    init_mv(output, input->dimension, input->signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        const unsigned grade = popcount_unsigned((unsigned)blade);
        const int sign = (((grade * (grade - 1u) / 2u) & 1u) != 0u) ? -1 : 1;
        output->coefficients[blade] =
            input->coefficients[blade] * (double)sign;
    }
}

static void project_reference(
    const geo_operator_mv_f64_t *input,
    uint8_t grade,
    geo_operator_mv_f64_t *output
) {
    size_t blade;
    const size_t blade_count = (size_t)1u << input->dimension;
    init_mv(output, input->dimension, input->signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        if (popcount_unsigned((unsigned)blade) == (unsigned)grade) {
            output->coefficients[blade] = input->coefficients[blade];
        }
    }
}

static void test_composed_adjoint_all_signatures(void) {
    uint8_t dimension;
    size_t checked = 0u;

    for (dimension = 1u; dimension <= GEO_OPERATOR_MAX_DIMENSION; ++dimension) {
        uint8_t q;
        for (q = 0u; q <= dimension; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            const uint8_t grade = dimension >= 2u ? 2u : 1u;
            geo_operator_mv_f64_t input;
            geo_operator_mv_f64_t weight;
            geo_operator_mv_f64_t input_tangent;
            geo_operator_mv_f64_t weight_tangent;
            geo_operator_mv_f64_t product_tangent;
            geo_operator_mv_f64_t reversed_tangent;
            geo_operator_mv_f64_t projected_tangent;
            geo_v7_node_id_t input_node;
            geo_v7_node_id_t weight_node;
            geo_v7_node_id_t product_node;
            geo_v7_node_id_t reverse_node;
            geo_v7_node_id_t projection_node;
            geo_v7_node_id_t loss_node;
            const geo_operator_mv_f64_t *projection;
            const geo_operator_mv_f64_t *input_gradient;
            const geo_operator_mv_f64_t *weight_gradient;

            make_signature(dimension, q, signature);
            fill_mv(&input, dimension, signature, (unsigned)dimension + q + 3u);
            fill_mv(
                &weight,
                dimension,
                signature,
                (unsigned)dimension * 3u + q + 7u
            );
            fill_mv(
                &input_tangent,
                dimension,
                signature,
                (unsigned)dimension + q + 11u
            );
            fill_mv(
                &weight_tangent,
                dimension,
                signature,
                (unsigned)dimension * 5u + q + 13u
            );

            CHECK(
                geo_v7_program_init(
                    &program,
                    dimension,
                    signature,
                    GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN
                ) == GEO_V7_OK
            );
            CHECK(
                geo_v7_add_input(&program, &input, 1, &input_node) ==
                GEO_V7_OK
            );
            CHECK(
                geo_v7_add_parameter(&program, &weight, &weight_node) ==
                GEO_V7_OK
            );
            CHECK(
                geo_v7_add_geometric_product(
                    &program,
                    input_node,
                    weight_node,
                    &product_node
                ) == GEO_V7_OK
            );
            CHECK(
                geo_v7_add_reverse(&program, product_node, &reverse_node) ==
                GEO_V7_OK
            );
            CHECK(
                geo_v7_add_grade_project(
                    &program,
                    reverse_node,
                    grade,
                    &projection_node
                ) == GEO_V7_OK
            );
            CHECK(
                geo_v7_add_squared_norm(
                    &program,
                    projection_node,
                    &loss_node
                ) == GEO_V7_OK
            );
            CHECK(geo_v7_compile(&program, loss_node) == GEO_V7_OK);
            CHECK(geo_v7_forward(&program) == GEO_V7_OK);
            CHECK(geo_v7_backward(&program) == GEO_V7_OK);

            projection = geo_v7_value(&program, projection_node);
            input_gradient = geo_v7_gradient(&program, input_node);
            weight_gradient = geo_v7_gradient(&program, weight_node);
            CHECK(projection != NULL);
            CHECK(input_gradient != NULL);
            CHECK(weight_gradient != NULL);

            CHECK(
                geo_operator_gp_f64_jvp(
                    &input,
                    &weight,
                    &input_tangent,
                    &weight_tangent,
                    &product_tangent
                ) == GEO_OPERATOR_OK
            );
            reverse_reference(&product_tangent, &reversed_tangent);
            project_reference(
                &reversed_tangent,
                grade,
                &projected_tangent
            );

            assert_near(
                mv_dot(projection, &projected_tangent),
                mv_dot(input_gradient, &input_tangent) +
                    mv_dot(weight_gradient, &weight_tangent),
                1e-10
            );
            ++checked;
        }
    }

    CHECK(checked == 27u);
}

static void test_branch_accumulation(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, 0, 0, 0};
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t weight;
    geo_operator_mv_f64_t product;
    geo_operator_mv_f64_t product_seed;
    geo_operator_mv_f64_t expected_input_gradient;
    geo_operator_mv_f64_t expected_weight_gradient;
    geo_v7_node_id_t input_node;
    geo_v7_node_id_t weight_node;
    geo_v7_node_id_t product_node;
    geo_v7_node_id_t doubled_node;
    geo_v7_node_id_t loss_node;
    const geo_operator_mv_f64_t *weight_gradient;
    size_t blade;

    fill_mv(&input, 3u, signature, 23u);
    fill_mv(&weight, 3u, signature, 29u);

    CHECK(
        geo_v7_program_init(
            &program,
            3u,
            signature,
            GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN
        ) == GEO_V7_OK
    );
    CHECK(geo_v7_add_input(&program, &input, 1, &input_node) == GEO_V7_OK);
    CHECK(
        geo_v7_add_parameter(&program, &weight, &weight_node) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_geometric_product(
            &program,
            input_node,
            weight_node,
            &product_node
        ) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_add(
            &program,
            product_node,
            product_node,
            &doubled_node
        ) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_squared_norm(&program, doubled_node, &loss_node) ==
        GEO_V7_OK
    );
    CHECK(geo_v7_compile(&program, loss_node) == GEO_V7_OK);
    CHECK(geo_v7_forward(&program) == GEO_V7_OK);
    CHECK(geo_v7_backward(&program) == GEO_V7_OK);

    CHECK(geo_operator_gp_f64(&input, &weight, &product) == GEO_OPERATOR_OK);
    product_seed = product;
    for (blade = 0u; blade < 8u; ++blade) {
        product_seed.coefficients[blade] *= 4.0;
    }
    CHECK(
        geo_operator_gp_f64_vjp(
            &input,
            &weight,
            &product_seed,
            &expected_input_gradient,
            &expected_weight_gradient
        ) == GEO_OPERATOR_OK
    );

    weight_gradient = geo_v7_gradient(&program, weight_node);
    CHECK(weight_gradient != NULL);
    assert_mv_near(weight_gradient, &expected_weight_gradient, 1e-12);
}

static void test_end_to_end_sgd_training(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, 0, 0, 0};
    const size_t blade_count = 8u;
    geo_operator_mv_f64_t truth;
    geo_operator_mv_f64_t zero;
    geo_v7_node_id_t parameter_node;
    geo_v7_node_id_t loss_sum_node;
    size_t source_blade;
    double initial_loss;
    double final_loss;

    fill_mv(&truth, 3u, signature, 41u);
    init_mv(&zero, 3u, signature);

    CHECK(
        geo_v7_program_init(
            &program,
            3u,
            signature,
            GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN
        ) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_parameter(&program, &zero, &parameter_node) ==
        GEO_V7_OK
    );
    CHECK(
        geo_v7_add_constant(&program, &zero, &loss_sum_node) ==
        GEO_V7_OK
    );

    for (source_blade = 0u; source_blade < blade_count; ++source_blade) {
        geo_operator_mv_f64_t input;
        geo_operator_mv_f64_t target;
        geo_v7_node_id_t input_node;
        geo_v7_node_id_t target_node;
        geo_v7_node_id_t prediction_node;
        geo_v7_node_id_t negative_target_node;
        geo_v7_node_id_t residual_node;
        geo_v7_node_id_t sample_loss_node;
        geo_v7_node_id_t next_loss_sum_node;

        init_mv(&input, 3u, signature);
        input.coefficients[source_blade] = 1.0;
        CHECK(
            geo_operator_gp_f64(&input, &truth, &target) == GEO_OPERATOR_OK
        );
        CHECK(
            geo_v7_add_constant(&program, &input, &input_node) == GEO_V7_OK
        );
        CHECK(
            geo_v7_add_constant(&program, &target, &target_node) ==
            GEO_V7_OK
        );
        CHECK(
            geo_v7_add_geometric_product(
                &program,
                input_node,
                parameter_node,
                &prediction_node
            ) == GEO_V7_OK
        );
        CHECK(
            geo_v7_add_scale(
                &program,
                target_node,
                -1.0,
                &negative_target_node
            ) == GEO_V7_OK
        );
        CHECK(
            geo_v7_add_add(
                &program,
                prediction_node,
                negative_target_node,
                &residual_node
            ) == GEO_V7_OK
        );
        CHECK(
            geo_v7_add_squared_norm(
                &program,
                residual_node,
                &sample_loss_node
            ) == GEO_V7_OK
        );
        CHECK(
            geo_v7_add_add(
                &program,
                loss_sum_node,
                sample_loss_node,
                &next_loss_sum_node
            ) == GEO_V7_OK
        );
        loss_sum_node = next_loss_sum_node;
    }

    CHECK(geo_v7_compile(&program, loss_sum_node) == GEO_V7_OK);
    CHECK(geo_v7_forward(&program) == GEO_V7_OK);
    initial_loss = geo_v7_value(&program, loss_sum_node)->coefficients[0];
    CHECK(initial_loss > 0.0);
    CHECK(geo_v7_backward(&program) == GEO_V7_OK);
    CHECK(
        geo_v7_sgd_step(&program, 1.0 / (double)blade_count) ==
        GEO_V7_OK
    );
    CHECK(geo_v7_forward(&program) == GEO_V7_OK);

    final_loss = geo_v7_value(&program, loss_sum_node)->coefficients[0];
    assert_near(final_loss, 0.0, 1e-13);
    assert_mv_near(
        geo_v7_value(&program, parameter_node),
        &truth,
        1e-13
    );
}

static void test_adam_training(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 0, 0, 0, 0, 0};
    geo_operator_mv_f64_t zero;
    geo_operator_mv_f64_t target;
    geo_v7_node_id_t parameter_node;
    geo_v7_node_id_t target_node;
    geo_v7_node_id_t negative_target_node;
    geo_v7_node_id_t residual_node;
    geo_v7_node_id_t loss_node;
    double initial_loss;
    double final_loss;
    int iteration;

    init_mv(&zero, 1u, signature);
    init_mv(&target, 1u, signature);
    target.coefficients[0] = 2.0;
    target.coefficients[1] = -1.0;

    CHECK(
        geo_v7_program_init(
            &program,
            1u,
            signature,
            GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN
        ) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_parameter(&program, &zero, &parameter_node) ==
        GEO_V7_OK
    );
    CHECK(
        geo_v7_add_constant(&program, &target, &target_node) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_scale(
            &program,
            target_node,
            -1.0,
            &negative_target_node
        ) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_add(
            &program,
            parameter_node,
            negative_target_node,
            &residual_node
        ) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_squared_norm(&program, residual_node, &loss_node) ==
        GEO_V7_OK
    );
    CHECK(geo_v7_compile(&program, loss_node) == GEO_V7_OK);
    CHECK(geo_v7_forward(&program) == GEO_V7_OK);
    initial_loss = geo_v7_value(&program, loss_node)->coefficients[0];

    for (iteration = 0; iteration < 150; ++iteration) {
        CHECK(geo_v7_backward(&program) == GEO_V7_OK);
        CHECK(
            geo_v7_adam_step(
                &program,
                0.05,
                0.9,
                0.999,
                1e-8
            ) == GEO_V7_OK
        );
        CHECK(geo_v7_forward(&program) == GEO_V7_OK);
    }

    final_loss = geo_v7_value(&program, loss_node)->coefficients[0];
    CHECK(final_loss < initial_loss * 1e-5);
}

static void test_rejections_and_transactionality(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    geo_operator_mv_f64_t value;
    geo_operator_mv_f64_t invalid_value;
    geo_operator_mv_f64_t preserved_value;
    geo_v7_node_id_t input_node;
    geo_v7_node_id_t unused_node;

    init_mv(&value, 2u, signature);
    value.coefficients[1] = 1.0;

    CHECK(
        geo_v7_program_init(
            &program,
            2u,
            signature,
            GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN
        ) == GEO_V7_OK
    );
    CHECK(geo_v7_forward(&program) == GEO_V7_NOT_COMPILED);
    CHECK(
        geo_v7_add_input(&program, &value, 1, &input_node) == GEO_V7_OK
    );
    CHECK(
        geo_v7_add_grade_project(
            &program,
            input_node,
            3u,
            &unused_node
        ) == GEO_V7_INVALID_ARGUMENT
    );
    CHECK(geo_v7_compile(&program, input_node) == GEO_V7_OK);
    CHECK(geo_v7_backward(&program) == GEO_V7_FORWARD_REQUIRED);
    CHECK(geo_v7_forward(&program) == GEO_V7_OK);
    CHECK(geo_v7_backward(&program) == GEO_V7_NON_SCALAR_LOSS);
    CHECK(geo_v7_sgd_step(&program, 0.1) == GEO_V7_BACKWARD_REQUIRED);

    preserved_value = *geo_v7_value(&program, input_node);
    invalid_value = value;
    invalid_value.coefficients[0] = NAN;
    CHECK(
        geo_v7_set_value(&program, input_node, &invalid_value) ==
        GEO_V7_INVALID_ARGUMENT
    );
    CHECK(
        memcmp(
            &preserved_value,
            geo_v7_value(&program, input_node),
            sizeof(preserved_value)
        ) == 0
    );
    CHECK(
        geo_v7_add_squared_norm(&program, input_node, &unused_node) ==
        GEO_V7_BAD_PROGRAM
    );
}

int main(void) {
    CHECK(geo_v7_abi_version() == GEO_V7_ABI_VERSION);
    CHECK(geo_v7_program_bytes() == sizeof(geo_v7_program_t));

    test_composed_adjoint_all_signatures();
    test_branch_accumulation();
    test_end_to_end_sgd_training();
    test_adam_training();
    test_rejections_and_transactionality();

    puts(
        "GEO_V7_NATIVE_AUTODIFF_TEST: PASS "
        "signatures=27 branch_accumulation=PASS "
        "sgd_training=PASS adam_training=PASS "
        "no_external_autograd=TRUE"
    );
    return 0;
}
