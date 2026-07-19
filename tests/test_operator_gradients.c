#include "geo/operator_kernel.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        abort(); \
    } \
} while (0)

static void make_signature(uint8_t dimension, uint8_t q, int8_t *signature) {
    uint8_t index;
    for (index = 0u; index < GEO_OPERATOR_MAX_DIMENSION; ++index) {
        if (index >= dimension) signature[index] = 0;
        else signature[index] = index < (uint8_t)(dimension - q) ? 1 : -1;
    }
}

static void init_f64(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static void fill_f64(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature,
    uint32_t seed
) {
    size_t blade;
    const size_t blade_count = geo_operator_blade_count(dimension);
    init_f64(value, dimension, signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        const uint32_t mixed =
            (uint32_t)(blade + 1u) * (seed * UINT32_C(17) + UINT32_C(29)) +
            seed * UINT32_C(13);
        value->coefficients[blade] = (double)((int)(mixed % UINT32_C(19)) - 9) / 4.0;
    }
}

static double mv_dot(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right
) {
    size_t blade;
    double sum = 0.0;
    const size_t blade_count = geo_operator_blade_count(left->dimension);
    CHECK(left->dimension == right->dimension);
    for (blade = 0u; blade < blade_count; ++blade) {
        sum += left->coefficients[blade] * right->coefficients[blade];
    }
    return sum;
}

static void mv_add(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    geo_operator_mv_f64_t *output
) {
    size_t blade;
    const size_t blade_count = geo_operator_blade_count(left->dimension);
    init_f64(output, left->dimension, left->signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        output->coefficients[blade] = left->coefficients[blade] + right->coefficients[blade];
    }
}

static void assert_scalar_near(double left, double right, double tolerance) {
    const double scale = 1.0 + fabs(left) + fabs(right);
    CHECK(fabs(left - right) <= tolerance * scale);
}

static void assert_mv_near(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    double tolerance
) {
    size_t blade;
    const size_t blade_count = geo_operator_blade_count(left->dimension);
    CHECK(left->dimension == right->dimension);
    for (blade = 0u; blade < blade_count; ++blade) {
        assert_scalar_near(left->coefficients[blade], right->coefficients[blade], tolerance);
    }
}

static void test_fixed_plan_vjp_all_dimensions(void) {
    uint8_t dimension;
    size_t checked = 0u;
    for (dimension = 2u; dimension <= 6u; ++dimension) {
        uint8_t q;
        const size_t blade_count = geo_operator_blade_count(dimension);
        for (q = 0u; q <= dimension; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            size_t fixed_blade;
            make_signature(dimension, q, signature);
            for (fixed_blade = 0u; fixed_blade < blade_count; ++fixed_blade) {
                uint8_t side;
                for (side = GEO_OPERATOR_SIDE_RIGHT; side <= GEO_OPERATOR_SIDE_LEFT; ++side) {
                    const int32_t coefficient =
                        (int32_t)(((fixed_blade % 3u) + 1u) * (side == GEO_OPERATOR_SIDE_RIGHT ? 1u : 2u));
                    geo_operator_plan_i32_t plan;
                    geo_operator_mv_f64_t input;
                    geo_operator_mv_f64_t output;
                    geo_operator_mv_f64_t output_cotangent;
                    geo_operator_mv_f64_t input_cotangent;
                    fill_f64(&input, dimension, signature, (uint32_t)(fixed_blade + 3u));
                    fill_f64(&output_cotangent, dimension, signature, (uint32_t)(q + side + 11u));
                    CHECK(
                        geo_operator_plan_fixed_blade_i32(
                            &plan,
                            dimension,
                            signature,
                            (geo_operator_side_t)side,
                            (uint8_t)fixed_blade,
                            coefficient
                        ) == GEO_OPERATOR_OK
                    );
                    CHECK(geo_operator_apply_f64(&plan, &input, &output) == GEO_OPERATOR_OK);
                    CHECK(
                        geo_operator_apply_f64_vjp(&plan, &output_cotangent, &input_cotangent) ==
                        GEO_OPERATOR_OK
                    );
                    assert_scalar_near(
                        mv_dot(&output, &output_cotangent),
                        mv_dot(&input, &input_cotangent),
                        1e-12
                    );
                    ++checked;
                }
            }
        }
    }
    CHECK(checked == 1528u);
}

static void test_sparse_plan_vjp_all_signatures(void) {
    uint8_t dimension;
    size_t checked = 0u;
    for (dimension = 2u; dimension <= 6u; ++dimension) {
        uint8_t q;
        const uint8_t pseudoscalar = (uint8_t)(geo_operator_blade_count(dimension) - 1u);
        for (q = 0u; q <= dimension; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            const geo_operator_term_i32_t terms[] = {
                {0u, 1},
                {1u, -2},
                {pseudoscalar, 3}
            };
            uint8_t side;
            make_signature(dimension, q, signature);
            for (side = GEO_OPERATOR_SIDE_RIGHT; side <= GEO_OPERATOR_SIDE_LEFT; ++side) {
                geo_operator_plan_i32_t plan;
                geo_operator_mv_f64_t input;
                geo_operator_mv_f64_t output;
                geo_operator_mv_f64_t output_cotangent;
                geo_operator_mv_f64_t input_cotangent;
                fill_f64(&input, dimension, signature, (uint32_t)(dimension + q + 5u));
                fill_f64(&output_cotangent, dimension, signature, (uint32_t)(dimension + side + 17u));
                CHECK(
                    geo_operator_plan_sparse_i32(
                        &plan,
                        dimension,
                        signature,
                        (geo_operator_side_t)side,
                        terms,
                        sizeof(terms) / sizeof(terms[0])
                    ) == GEO_OPERATOR_OK
                );
                CHECK(geo_operator_apply_f64(&plan, &input, &output) == GEO_OPERATOR_OK);
                CHECK(
                    geo_operator_apply_f64_vjp(&plan, &output_cotangent, &input_cotangent) ==
                    GEO_OPERATOR_OK
                );
                assert_scalar_near(
                    mv_dot(&output, &output_cotangent),
                    mv_dot(&input, &input_cotangent),
                    1e-12
                );
                ++checked;
            }
        }
    }
    CHECK(checked == 50u);
}

static void test_parametric_vjp_all_signatures(void) {
    uint8_t dimension;
    size_t checked = 0u;
    for (dimension = 2u; dimension <= 6u; ++dimension) {
        uint8_t q;
        const uint8_t pseudoscalar = (uint8_t)(geo_operator_blade_count(dimension) - 1u);
        for (q = 0u; q <= dimension; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            const geo_operator_term_i32_t terms[] = {
                {0u, 1},
                {1u, -2},
                {pseudoscalar, 3}
            };
            const double parameters[] = {0.75, -1.5, 2.25};
            const double parameter_tangent[] = {-0.5, 0.25, 1.125};
            uint8_t side;
            make_signature(dimension, q, signature);
            for (side = GEO_OPERATOR_SIDE_RIGHT; side <= GEO_OPERATOR_SIDE_LEFT; ++side) {
                geo_operator_plan_i32_t plan;
                geo_operator_mv_f64_t input;
                geo_operator_mv_f64_t input_tangent;
                geo_operator_mv_f64_t output_cotangent;
                geo_operator_mv_f64_t input_cotangent;
                geo_operator_mv_f64_t input_jvp;
                geo_operator_mv_f64_t parameter_jvp;
                geo_operator_mv_f64_t total_jvp;
                double parameter_cotangent[3] = {0.0, 0.0, 0.0};
                double lhs;
                double rhs;
                size_t term_index;
                fill_f64(&input, dimension, signature, (uint32_t)(dimension + q + 23u));
                fill_f64(&input_tangent, dimension, signature, (uint32_t)(dimension + side + 31u));
                fill_f64(&output_cotangent, dimension, signature, (uint32_t)(q + side + 41u));
                CHECK(
                    geo_operator_plan_sparse_i32(
                        &plan,
                        dimension,
                        signature,
                        (geo_operator_side_t)side,
                        terms,
                        sizeof(terms) / sizeof(terms[0])
                    ) == GEO_OPERATOR_OK
                );
                CHECK(
                    geo_operator_apply_parametric_f64_vjp(
                        &plan,
                        parameters,
                        3u,
                        &input,
                        &output_cotangent,
                        &input_cotangent,
                        parameter_cotangent,
                        3u
                    ) == GEO_OPERATOR_OK
                );
                CHECK(
                    geo_operator_apply_parametric_f64(
                        &plan,
                        parameters,
                        3u,
                        &input_tangent,
                        &input_jvp
                    ) == GEO_OPERATOR_OK
                );
                CHECK(
                    geo_operator_apply_parametric_f64(
                        &plan,
                        parameter_tangent,
                        3u,
                        &input,
                        &parameter_jvp
                    ) == GEO_OPERATOR_OK
                );
                mv_add(&input_jvp, &parameter_jvp, &total_jvp);
                lhs = mv_dot(&output_cotangent, &total_jvp);
                rhs = mv_dot(&input_cotangent, &input_tangent);
                for (term_index = 0u; term_index < 3u; ++term_index) {
                    rhs += parameter_cotangent[term_index] * parameter_tangent[term_index];
                }
                assert_scalar_near(lhs, rhs, 1e-12);
                ++checked;
            }
        }
    }
    CHECK(checked == 50u);
}

static void test_parametric_initialization_matches_integer_plan(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, -1, 0, 0};
    const geo_operator_term_i32_t terms[] = {{0u, 1}, {1u, -2}, {15u, 3}};
    const double parameters[] = {1.0, -2.0, 3.0};
    geo_operator_plan_i32_t plan;
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t integer_output;
    geo_operator_mv_f64_t parametric_output;
    fill_f64(&input, 4u, signature, 53u);
    CHECK(
        geo_operator_plan_sparse_i32(
            &plan,
            4u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            terms,
            3u
        ) == GEO_OPERATOR_OK
    );
    CHECK(geo_operator_apply_f64(&plan, &input, &integer_output) == GEO_OPERATOR_OK);
    CHECK(
        geo_operator_apply_parametric_f64(&plan, parameters, 3u, &input, &parametric_output) ==
        GEO_OPERATOR_OK
    );
    assert_mv_near(&integer_output, &parametric_output, 0.0);
}

static void test_gp_jvp_vjp_all_signatures(void) {
    uint8_t dimension;
    size_t checked = 0u;
    for (dimension = 1u; dimension <= 6u; ++dimension) {
        uint8_t q;
        for (q = 0u; q <= dimension; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            geo_operator_mv_f64_t left;
            geo_operator_mv_f64_t right;
            geo_operator_mv_f64_t left_tangent;
            geo_operator_mv_f64_t right_tangent;
            geo_operator_mv_f64_t output_cotangent;
            geo_operator_mv_f64_t output_tangent;
            geo_operator_mv_f64_t left_cotangent;
            geo_operator_mv_f64_t right_cotangent;
            make_signature(dimension, q, signature);
            fill_f64(&left, dimension, signature, (uint32_t)(dimension + q + 61u));
            fill_f64(&right, dimension, signature, (uint32_t)(dimension * 3u + q + 67u));
            fill_f64(&left_tangent, dimension, signature, (uint32_t)(dimension + q + 71u));
            fill_f64(&right_tangent, dimension, signature, (uint32_t)(dimension * 5u + q + 73u));
            fill_f64(&output_cotangent, dimension, signature, (uint32_t)(dimension + q + 79u));
            CHECK(
                geo_operator_gp_f64_jvp(
                    &left,
                    &right,
                    &left_tangent,
                    &right_tangent,
                    &output_tangent
                ) == GEO_OPERATOR_OK
            );
            CHECK(
                geo_operator_gp_f64_vjp(
                    &left,
                    &right,
                    &output_cotangent,
                    &left_cotangent,
                    &right_cotangent
                ) == GEO_OPERATOR_OK
            );
            assert_scalar_near(
                mv_dot(&output_cotangent, &output_tangent),
                mv_dot(&left_cotangent, &left_tangent) +
                    mv_dot(&right_cotangent, &right_tangent),
                1e-11
            );
            ++checked;
        }
    }
    CHECK(checked == 27u);
}

static void test_native_training_step(void) {
    const uint8_t dimension = 3u;
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, 0, 0, 0};
    const size_t blade_count = geo_operator_blade_count(dimension);
    geo_operator_mv_f64_t truth;
    geo_operator_mv_f64_t weight;
    geo_operator_mv_f64_t accumulated_gradient;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    size_t sample;
    size_t blade;
    fill_f64(&truth, dimension, signature, 101u);
    init_f64(&weight, dimension, signature);
    init_f64(&accumulated_gradient, dimension, signature);

    for (sample = 0u; sample < blade_count; ++sample) {
        geo_operator_mv_f64_t input;
        geo_operator_mv_f64_t target;
        geo_operator_mv_f64_t output;
        geo_operator_mv_f64_t residual;
        geo_operator_mv_f64_t input_gradient;
        geo_operator_mv_f64_t weight_gradient;
        init_f64(&input, dimension, signature);
        input.coefficients[sample] = 1.0;
        CHECK(geo_operator_gp_f64(&input, &truth, &target) == GEO_OPERATOR_OK);
        CHECK(geo_operator_gp_f64(&input, &weight, &output) == GEO_OPERATOR_OK);
        init_f64(&residual, dimension, signature);
        for (blade = 0u; blade < blade_count; ++blade) {
            residual.coefficients[blade] = output.coefficients[blade] - target.coefficients[blade];
        }
        initial_loss += 0.5 * mv_dot(&residual, &residual);
        CHECK(
            geo_operator_gp_f64_vjp(
                &input,
                &weight,
                &residual,
                &input_gradient,
                &weight_gradient
            ) == GEO_OPERATOR_OK
        );
        for (blade = 0u; blade < blade_count; ++blade) {
            accumulated_gradient.coefficients[blade] += weight_gradient.coefficients[blade];
        }
    }

    for (blade = 0u; blade < blade_count; ++blade) {
        weight.coefficients[blade] -= accumulated_gradient.coefficients[blade] / (double)blade_count;
    }

    for (sample = 0u; sample < blade_count; ++sample) {
        geo_operator_mv_f64_t input;
        geo_operator_mv_f64_t target;
        geo_operator_mv_f64_t output;
        geo_operator_mv_f64_t residual;
        init_f64(&input, dimension, signature);
        input.coefficients[sample] = 1.0;
        CHECK(geo_operator_gp_f64(&input, &truth, &target) == GEO_OPERATOR_OK);
        CHECK(geo_operator_gp_f64(&input, &weight, &output) == GEO_OPERATOR_OK);
        init_f64(&residual, dimension, signature);
        for (blade = 0u; blade < blade_count; ++blade) {
            residual.coefficients[blade] = output.coefficients[blade] - target.coefficients[blade];
        }
        final_loss += 0.5 * mv_dot(&residual, &residual);
    }

    CHECK(initial_loss > 0.0);
    CHECK(final_loss <= 1e-24);
    assert_mv_near(&weight, &truth, 1e-13);
}

static void test_gradient_rejections_are_transactional(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    const int8_t wrong_signature[GEO_OPERATOR_MAX_DIMENSION] = {1, -1, 0, 0, 0, 0};
    const geo_operator_term_i32_t terms[] = {{0u, 1}, {1u, -1}};
    const double parameters[] = {1.0, -1.0};
    geo_operator_plan_i32_t plan;
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t wrong_cotangent;
    geo_operator_mv_f64_t preserved;
    geo_operator_mv_f64_t before;
    double parameter_cotangents[2] = {123.0, 456.0};
    double parameter_before[2];
    CHECK(
        geo_operator_plan_sparse_i32(
            &plan,
            2u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            terms,
            2u
        ) == GEO_OPERATOR_OK
    );
    fill_f64(&input, 2u, signature, 109u);
    fill_f64(&wrong_cotangent, 2u, wrong_signature, 113u);
    fill_f64(&preserved, 2u, signature, 127u);
    before = preserved;
    memcpy(parameter_before, parameter_cotangents, sizeof(parameter_before));
    CHECK(
        geo_operator_apply_f64_vjp(&plan, &wrong_cotangent, &preserved) ==
        GEO_OPERATOR_INVALID_ARGUMENT
    );
    CHECK(memcmp(&preserved, &before, sizeof(preserved)) == 0);
    CHECK(
        geo_operator_apply_parametric_f64_vjp(
            &plan,
            parameters,
            2u,
            &input,
            &wrong_cotangent,
            &preserved,
            parameter_cotangents,
            2u
        ) == GEO_OPERATOR_INVALID_ARGUMENT
    );
    CHECK(memcmp(&preserved, &before, sizeof(preserved)) == 0);
    CHECK(memcmp(parameter_cotangents, parameter_before, sizeof(parameter_before)) == 0);
    CHECK(
        geo_operator_gp_f64_vjp(&input, &input, &input, &preserved, &preserved) ==
        GEO_OPERATOR_INVALID_ARGUMENT
    );
    CHECK(memcmp(&preserved, &before, sizeof(preserved)) == 0);
}

int main(void) {
    CHECK(geo_operator_abi_version() == GEO_OPERATOR_ABI_VERSION);
    CHECK(geo_operator_gradient_abi_version() == GEO_OPERATOR_GRADIENT_ABI_VERSION);
    test_fixed_plan_vjp_all_dimensions();
    test_sparse_plan_vjp_all_signatures();
    test_parametric_vjp_all_signatures();
    test_parametric_initialization_matches_integer_plan();
    test_gp_jvp_vjp_all_signatures();
    test_native_training_step();
    test_gradient_rejections_are_transactional();
    puts(
        "GEO_OPERATOR_GRADIENT_TEST: PASS "
        "fixed_vjp=1528 sparse_vjp=50 parametric_vjp=50 gp_signatures=27 native_training=PASS"
    );
    return 0;
}
