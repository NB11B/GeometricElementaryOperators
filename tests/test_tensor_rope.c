#include "geo/tensor_rope.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOL ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOL ((geo_real_t)2e-5f)
#endif

static void assert_close(geo_real_t actual, geo_real_t expected) {
    const geo_real_t error = (geo_real_t)fabs((double)(actual - expected));
    assert(error <= GEO_TEST_TOL);
}

static void test_build_tables(void) {
    geo_real_t cos_table[6];
    geo_real_t sin_table[6];
    const geo_tensor_rope_table_shape shape = {3u, 4u};
    assert(geo_tensor_rope_build((geo_real_t)10000, cos_table, sin_table, shape) == GEO_TENSOR_OK);

    assert_close(cos_table[0], (geo_real_t)1);
    assert_close(cos_table[1], (geo_real_t)1);
    assert_close(sin_table[0], (geo_real_t)0);
    assert_close(sin_table[1], (geo_real_t)0);

    assert_close(cos_table[2], (geo_real_t)cos(1.0));
    assert_close(sin_table[2], (geo_real_t)sin(1.0));
    assert_close(cos_table[3], (geo_real_t)cos(0.01));
    assert_close(sin_table[3], (geo_real_t)sin(0.01));

    assert_close(cos_table[4], (geo_real_t)cos(2.0));
    assert_close(sin_table[4], (geo_real_t)sin(2.0));
    assert_close(cos_table[5], (geo_real_t)cos(0.02));
    assert_close(sin_table[5], (geo_real_t)sin(0.02));
}

static void test_apply_and_vjp(void) {
    geo_real_t cos_table[4];
    geo_real_t sin_table[4];
    const geo_tensor_rope_table_shape table_shape = {2u, 4u};
    assert(geo_tensor_rope_build((geo_real_t)10000, cos_table, sin_table, table_shape) == GEO_TENSOR_OK);

    const geo_real_t x[8] = {
        (geo_real_t)1, (geo_real_t)2, (geo_real_t)3, (geo_real_t)4,
        (geo_real_t)1, (geo_real_t)2, (geo_real_t)3, (geo_real_t)4
    };
    geo_real_t out[8];
    const geo_tensor_rope_apply_shape shape = {1u, 2u, 4u, 2u};
    assert(geo_tensor_rope_apply_forward(x, cos_table, sin_table, out, shape) == GEO_TENSOR_OK);

    assert_close(out[0], x[0]);
    assert_close(out[1], x[1]);
    assert_close(out[2], x[2]);
    assert_close(out[3], x[3]);

    const geo_real_t c0 = (geo_real_t)cos(1.0);
    const geo_real_t s0 = (geo_real_t)sin(1.0);
    const geo_real_t c1 = (geo_real_t)cos(0.01);
    const geo_real_t s1 = (geo_real_t)sin(0.01);
    assert_close(out[4], x[4] * c0 - x[6] * s0);
    assert_close(out[5], x[5] * c1 - x[7] * s1);
    assert_close(out[6], x[6] * c0 + x[4] * s0);
    assert_close(out[7], x[7] * c1 + x[5] * s1);

    const geo_real_t grad_out[8] = {
        (geo_real_t)1, (geo_real_t)-2, (geo_real_t)3, (geo_real_t)4,
        (geo_real_t)0.5, (geo_real_t)-1, (geo_real_t)2, (geo_real_t)3
    };
    geo_real_t grad_x[8];
    assert(geo_tensor_rope_apply_vjp(grad_out, cos_table, sin_table, grad_x, shape) == GEO_TENSOR_OK);

    assert_close(grad_x[0], grad_out[0]);
    assert_close(grad_x[1], grad_out[1]);
    assert_close(grad_x[2], grad_out[2]);
    assert_close(grad_x[3], grad_out[3]);
    assert_close(grad_x[4], grad_out[4] * c0 + grad_out[6] * s0);
    assert_close(grad_x[5], grad_out[5] * c1 + grad_out[7] * s1);
    assert_close(grad_x[6], -grad_out[4] * s0 + grad_out[6] * c0);
    assert_close(grad_x[7], -grad_out[5] * s1 + grad_out[7] * c1);
}

static void test_invalid_shapes(void) {
    geo_real_t cos_table[4];
    geo_real_t sin_table[4];
    const geo_tensor_rope_table_shape odd = {2u, 3u};
    assert(geo_tensor_rope_build((geo_real_t)10000, cos_table, sin_table, odd) == GEO_TENSOR_INVALID_ARGUMENT);

    const geo_real_t x[8] = {0};
    geo_real_t out[8];
    const geo_tensor_rope_apply_shape short_table = {1u, 2u, 4u, 1u};
    assert(geo_tensor_rope_apply_forward(x, cos_table, sin_table, out, short_table) == GEO_TENSOR_INVALID_ARGUMENT);
}

int main(void) {
    test_build_tables();
    test_apply_and_vjp();
    test_invalid_shapes();
    puts("tensor_rope: ok");
    return 0;
}
