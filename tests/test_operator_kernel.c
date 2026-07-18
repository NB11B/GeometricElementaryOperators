#include "geo/operator_kernel.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void init_f64(geo_operator_mv_f64_t *value, uint8_t dimension, const int8_t *signature) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static void init_i32(geo_operator_mv_i32_t *value, uint8_t dimension, const int8_t *signature) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static void assert_near(const geo_operator_mv_f64_t *a, const geo_operator_mv_f64_t *b, double tolerance) {
    size_t blade;
    const size_t blade_count = geo_operator_blade_count(a->dimension);
    assert(a->dimension == b->dimension);
    for (blade = 0u; blade < blade_count; ++blade) {
        assert(fabs(a->coefficients[blade] - b->coefficients[blade]) <= tolerance);
    }
}

static void test_fixed_blade_matches_generic(void) {
    const int8_t signatures[][GEO_OPERATOR_MAX_DIMENSION] = {
        {1, 1, 1, 1, 0, 0},
        {1, 1, 1, -1, 0, 0},
        {1, 1, -1, -1, 0, 0},
        {1, -1, -1, -1, 0, 0},
        {-1, -1, -1, -1, 0, 0}
    };
    size_t signature_index;
    for (signature_index = 0u; signature_index < 5u; ++signature_index) {
        geo_operator_mv_f64_t input;
        geo_operator_mv_f64_t constant;
        geo_operator_mv_f64_t generic;
        geo_operator_mv_f64_t specialized;
        geo_operator_plan_i32_t plan;
        size_t blade;
        init_f64(&input, 4u, signatures[signature_index]);
        init_f64(&constant, 4u, signatures[signature_index]);
        for (blade = 0u; blade < 16u; ++blade) input.coefficients[blade] = (double)((int)blade - 7) / 3.0;
        constant.coefficients[15] = 1.0;
        assert(geo_operator_plan_fixed_blade_i32(
            &plan, 4u, signatures[signature_index], GEO_OPERATOR_SIDE_RIGHT, 15u, 1
        ) == GEO_OPERATOR_OK);
        assert(geo_operator_apply_f64(&plan, &input, &specialized) == GEO_OPERATOR_OK);
        assert(geo_operator_gp_f64(&input, &constant, &generic) == GEO_OPERATOR_OK);
        assert_near(&generic, &specialized, 1e-12);
    }
}

static void test_sparse_matches_generic(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, -1, 0, 0};
    const geo_operator_term_i32_t terms[] = {
        {0u, 1}, {1u, -2}, {3u, 3}, {15u, 1}
    };
    geo_operator_plan_i32_t plan;
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t constant;
    geo_operator_mv_f64_t generic;
    geo_operator_mv_f64_t specialized;
    size_t blade;
    init_f64(&input, 4u, signature);
    init_f64(&constant, 4u, signature);
    for (blade = 0u; blade < 16u; ++blade) input.coefficients[blade] = (double)((int)(blade % 5u) - 2);
    for (blade = 0u; blade < sizeof(terms) / sizeof(terms[0]); ++blade) {
        constant.coefficients[terms[blade].blade] = (double)terms[blade].coefficient;
    }
    assert(geo_operator_plan_sparse_i32(
        &plan, 4u, signature, GEO_OPERATOR_SIDE_RIGHT, terms, sizeof(terms) / sizeof(terms[0])
    ) == GEO_OPERATOR_OK);
    assert(geo_operator_apply_f64(&plan, &input, &specialized) == GEO_OPERATOR_OK);
    assert(geo_operator_gp_f64(&input, &constant, &generic) == GEO_OPERATOR_OK);
    assert_near(&generic, &specialized, 1e-12);
}

static void test_left_sparse_matches_generic(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 1, -1, 0, 0};
    const geo_operator_term_i32_t terms[] = {{0u, 2}, {5u, -1}, {15u, 1}};
    geo_operator_plan_i32_t plan;
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t constant;
    geo_operator_mv_f64_t generic;
    geo_operator_mv_f64_t specialized;
    size_t blade;
    init_f64(&input, 4u, signature);
    init_f64(&constant, 4u, signature);
    for (blade = 0u; blade < 16u; ++blade) input.coefficients[blade] = (double)((int)blade - 8);
    for (blade = 0u; blade < sizeof(terms) / sizeof(terms[0]); ++blade) {
        constant.coefficients[terms[blade].blade] = (double)terms[blade].coefficient;
    }
    assert(geo_operator_plan_sparse_i32(
        &plan, 4u, signature, GEO_OPERATOR_SIDE_LEFT, terms, sizeof(terms) / sizeof(terms[0])
    ) == GEO_OPERATOR_OK);
    assert(geo_operator_apply_f64(&plan, &input, &specialized) == GEO_OPERATOR_OK);
    assert(geo_operator_gp_f64(&constant, &input, &generic) == GEO_OPERATOR_OK);
    assert_near(&generic, &specialized, 1e-12);
}

static void test_modular_path(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, -1, 0, 0};
    const geo_operator_term_i32_t terms[] = {{0u, 1}, {15u, -1}};
    geo_operator_plan_i32_t plan;
    geo_operator_mv_i32_t input;
    geo_operator_mv_i32_t output;
    size_t blade;
    init_i32(&input, 4u, signature);
    for (blade = 0u; blade < 16u; ++blade) input.coefficients[blade] = (int32_t)((int)blade - 10);
    assert(geo_operator_plan_sparse_i32(&plan, 4u, signature, GEO_OPERATOR_SIDE_RIGHT, terms, 2u) == GEO_OPERATOR_OK);
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
    assert(geo_operator_plan_sparse_i32(&plan, 2u, signature, GEO_OPERATOR_SIDE_RIGHT, &term, 1u) == GEO_OPERATOR_OK);
    assert(geo_operator_apply_q_i32(&plan, &input, 16u, &output) == GEO_OPERATOR_OK);
    assert(output.coefficients[2] == 32768 || output.coefficients[2] == -32768);
}

static void test_rejections(void) {
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, -1, -1, 0, 0};
    const geo_operator_term_i32_t duplicate[] = {{1u, 1}, {1u, 2}};
    const geo_operator_term_i32_t zero[] = {{1u, 0}};
    geo_operator_plan_i32_t plan;
    assert(geo_operator_plan_sparse_i32(&plan, 4u, signature, GEO_OPERATOR_SIDE_RIGHT, duplicate, 2u) == GEO_OPERATOR_INVALID_ARGUMENT);
    assert(geo_operator_plan_sparse_i32(&plan, 4u, signature, GEO_OPERATOR_SIDE_RIGHT, zero, 1u) == GEO_OPERATOR_INVALID_ARGUMENT);
    assert(geo_operator_plan_fixed_blade_i32(&plan, 7u, signature, GEO_OPERATOR_SIDE_RIGHT, 0u, 1) == GEO_OPERATOR_UNSUPPORTED_DIMENSION);
    assert(geo_operator_abi_version() == GEO_OPERATOR_ABI_VERSION);
}

int main(void) {
    test_fixed_blade_matches_generic();
    test_sparse_matches_generic();
    test_left_sparse_matches_generic();
    test_modular_path();
    test_q_path();
    test_rejections();
    return 0;
}
