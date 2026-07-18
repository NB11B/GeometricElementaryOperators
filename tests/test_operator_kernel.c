#include "geo/operator_embedded.h"
#include "geo/operator_kernel.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

static void init_i32(
    geo_operator_mv_i32_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static void assert_near(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    double tolerance
) {
    size_t blade;
    const size_t blade_count = geo_operator_blade_count(left->dimension);
    assert(left->dimension == right->dimension);
    for (blade = 0u; blade < blade_count; ++blade) {
        assert(fabs(left->coefficients[blade] - right->coefficients[blade]) <= tolerance);
    }
}

static void test_fixed_blades_all_dimensions(void) {
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
                    geo_operator_mv_f64_t input;
                    geo_operator_mv_f64_t constant;
                    geo_operator_mv_f64_t generic;
                    geo_operator_mv_f64_t specialized;
                    geo_operator_plan_i32_t plan;
                    size_t blade;
                    init_f64(&input, dimension, signature);
                    init_f64(&constant, dimension, signature);
                    for (blade = 0u; blade < blade_count; ++blade) {
                        input.coefficients[blade] = (double)((int)(blade % 11u) - 5);
                    }
                    constant.coefficients[fixed_blade] = 1.0;
                    assert(
                        geo_operator_plan_fixed_blade_i32(
                            &plan,
                            dimension,
                            signature,
                            (geo_operator_side_t)side,
                            (uint8_t)fixed_blade,
                            1
                        ) == GEO_OPERATOR_OK
                    );
                    assert(geo_operator_apply_f64(&plan, &input, &specialized) == GEO_OPERATOR_OK);
                    if (side == GEO_OPERATOR_SIDE_RIGHT) {
                        assert(geo_operator_gp_f64(&input, &constant, &generic) == GEO_OPERATOR_OK);
                    }
                    else {
                        assert(geo_operator_gp_f64(&constant, &input, &generic) == GEO_OPERATOR_OK);
                    }
                    assert_near(&generic, &specialized, 1e-12);
                    ++checked;
                }
            }
        }
    }
    assert(checked == 1528u);
}

static void test_sparse_all_signatures(void) {
    uint8_t dimension;
    size_t checked = 0u;
    for (dimension = 2u; dimension <= 6u; ++dimension) {
        uint8_t q;
        const size_t blade_count = geo_operator_blade_count(dimension);
        const uint8_t pseudoscalar = (uint8_t)(blade_count - 1u);
        for (q = 0u; q <= dimension; ++q) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
            const geo_operator_term_i32_t terms[] = {
                {0u, 1},
                {1u, -2},
                {pseudoscalar, 1}
            };
            uint8_t side;
            make_signature(dimension, q, signature);
            for (side = GEO_OPERATOR_SIDE_RIGHT; side <= GEO_OPERATOR_SIDE_LEFT; ++side) {
                geo_operator_plan_i32_t plan;
                geo_operator_mv_f64_t input;
                geo_operator_mv_f64_t constant;
                geo_operator_mv_f64_t generic;
                geo_operator_mv_f64_t specialized;
                size_t blade;
                init_f64(&input, dimension, signature);
                init_f64(&constant, dimension, signature);
                for (blade = 0u; blade < blade_count; ++blade) {
                    input.coefficients[blade] = (double)((int)(blade % 7u) - 3);
                }
                constant.coefficients[0] = 1.0;
                constant.coefficients[1] = -2.0;
                constant.coefficients[pseudoscalar] = 1.0;
                assert(
                    geo_operator_plan_sparse_i32(
                        &plan,
                        dimension,
                        signature,
                        (geo_operator_side_t)side,
                        terms,
                        sizeof(terms) / sizeof(terms[0])
                    ) == GEO_OPERATOR_OK
                );
                assert(geo_operator_apply_f64(&plan, &input, &specialized) == GEO_OPERATOR_OK);
                if (side == GEO_OPERATOR_SIDE_RIGHT) {
                    assert(geo_operator_gp_f64(&input, &constant, &generic) == GEO_OPERATOR_OK);
                }
                else {
                    assert(geo_operator_gp_f64(&constant, &input, &generic) == GEO_OPERATOR_OK);
                }
                assert_near(&generic, &specialized, 1e-12);
                ++checked;
            }
        }
    }
    assert(checked == 50u);
}

static void test_modular_path(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, -1, 0, 0};
    const geo_operator_term_i32_t terms[] = {{0u, 1}, {15u, -1}};
    geo_operator_plan_i32_t plan;
    geo_operator_mv_i32_t input;
    geo_operator_mv_i32_t output;
    size_t blade;
    init_i32(&input, 4u, signature);
    for (blade = 0u; blade < 16u; ++blade) {
        input.coefficients[blade] = (int32_t)((int)blade - 10);
    }
    assert(
        geo_operator_plan_sparse_i32(
            &plan,
            4u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            terms,
            2u
        ) == GEO_OPERATOR_OK
    );
    assert(geo_operator_apply_mod_i32(&plan, &input, 65521u, &output) == GEO_OPERATOR_OK);
    for (blade = 0u; blade < 16u; ++blade) {
        assert(output.coefficients[blade] >= 0);
        assert(output.coefficients[blade] < 65521);
    }
}

static void test_q_path(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    const geo_operator_term_i32_t term = {3u, 65536};
    geo_operator_plan_i32_t plan;
    geo_operator_mv_i32_t input;
    geo_operator_mv_i32_t output;
    init_i32(&input, 2u, signature);
    input.coefficients[1] = 32768;
    assert(
        geo_operator_plan_sparse_i32(
            &plan,
            2u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            &term,
            1u
        ) == GEO_OPERATOR_OK
    );
    assert(geo_operator_apply_q_i32(&plan, &input, 16u, &output) == GEO_OPERATOR_OK);
    assert(output.coefficients[2] == 32768 || output.coefficients[2] == -32768);
}

static void test_embedded_contract(void) {
    const geo_operator_embedded_limits_t limits = geo_operator_embedded_limits();
    assert(limits.maximum_dimension == 6u);
    assert(limits.maximum_blades == 64u);
    assert(limits.maximum_terms == 64u);
    assert(limits.maximum_plan_bytes == sizeof(geo_operator_plan_i32_t));
    assert(limits.i32_scratch_bytes == sizeof(geo_operator_mv_i32_t));
    assert(limits.q_i32_scratch_bytes == sizeof(geo_operator_mv_i32_t));
    assert(limits.f64_scratch_bytes == sizeof(geo_operator_mv_f64_t));
    assert(limits.maximum_contributions_per_output == 64u);
    assert(!limits.heap_required);
    assert(!limits.runtime_parser_required);
    assert(GEO_OPERATOR_EMBEDDED_NO_HEAP == 1);
    assert(GEO_OPERATOR_EMBEDDED_NO_RUNTIME_PARSER == 1);
}

static void test_rejections(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, -1, 0, 0};
    const geo_operator_term_i32_t duplicate[] = {{1u, 1}, {1u, 2}};
    const geo_operator_term_i32_t zero[] = {{1u, 0}};
    geo_operator_plan_i32_t plan;
    assert(
        geo_operator_plan_sparse_i32(
            &plan,
            4u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            duplicate,
            2u
        ) == GEO_OPERATOR_INVALID_ARGUMENT
    );
    assert(
        geo_operator_plan_sparse_i32(
            &plan,
            4u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            zero,
            1u
        ) == GEO_OPERATOR_INVALID_ARGUMENT
    );
    assert(
        geo_operator_plan_fixed_blade_i32(
            &plan,
            7u,
            signature,
            GEO_OPERATOR_SIDE_RIGHT,
            0u,
            1
        ) == GEO_OPERATOR_UNSUPPORTED_DIMENSION
    );
    assert(geo_operator_abi_version() == GEO_OPERATOR_ABI_VERSION);
}

int main(void) {
    test_fixed_blades_all_dimensions();
    test_sparse_all_signatures();
    test_modular_path();
    test_q_path();
    test_embedded_contract();
    test_rejections();
    puts("GEO_OPERATOR_KERNEL_TEST: PASS dimensions=2-6 signatures=25 fixed_cases=1528 sparse_cases=50");
    return 0;
}
