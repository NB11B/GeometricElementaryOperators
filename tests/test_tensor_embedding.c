#include "geo/tensor_embedding.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_TEST_TOL ((geo_real_t)1e-12)
#else
#define GEO_TEST_TOL ((geo_real_t)1e-6f)
#endif

static void assert_close(geo_real_t actual, geo_real_t expected) {
    const geo_real_t error = (geo_real_t)fabs((double)(actual - expected));
    assert(error <= GEO_TEST_TOL);
}

static void test_forward(void) {
    const geo_tensor_embedding_shape shape = {4u, 3u, 2u};
    const int64_t indices[4] = {2, 0, 2, 1};
    const geo_real_t weight[6] = {
        (geo_real_t)1, (geo_real_t)2,
        (geo_real_t)3, (geo_real_t)4,
        (geo_real_t)5, (geo_real_t)6
    };
    geo_real_t out[8];
    assert(geo_tensor_embedding_forward(indices, weight, out, shape) == GEO_TENSOR_OK);
    const geo_real_t expected[8] = {
        (geo_real_t)5, (geo_real_t)6,
        (geo_real_t)1, (geo_real_t)2,
        (geo_real_t)5, (geo_real_t)6,
        (geo_real_t)3, (geo_real_t)4
    };
    for (size_t index = 0u; index < 8u; ++index) {
        assert_close(out[index], expected[index]);
    }
}

static void test_vjp_accumulates_repeated_tokens(void) {
    const geo_tensor_embedding_shape shape = {4u, 3u, 2u};
    const int64_t indices[4] = {2, 0, 2, 1};
    const geo_real_t grad_out[8] = {
        (geo_real_t)1, (geo_real_t)2,
        (geo_real_t)3, (geo_real_t)4,
        (geo_real_t)5, (geo_real_t)6,
        (geo_real_t)7, (geo_real_t)8
    };
    geo_real_t grad_weight[6];
    assert(geo_tensor_embedding_vjp(
        indices, grad_out, grad_weight, shape
    ) == GEO_TENSOR_OK);
    const geo_real_t expected[6] = {
        (geo_real_t)3, (geo_real_t)4,
        (geo_real_t)7, (geo_real_t)8,
        (geo_real_t)6, (geo_real_t)8
    };
    for (size_t index = 0u; index < 6u; ++index) {
        assert_close(grad_weight[index], expected[index]);
    }
}

static void test_invalid_index(void) {
    const geo_tensor_embedding_shape shape = {1u, 2u, 2u};
    const int64_t indices[1] = {2};
    const geo_real_t weight[4] = {0};
    geo_real_t out[2];
    assert(geo_tensor_embedding_forward(
        indices, weight, out, shape
    ) == GEO_TENSOR_INVALID_ARGUMENT);
}

int main(void) {
    test_forward();
    test_vjp_accumulates_repeated_tokens();
    test_invalid_index();
    puts("tensor_embedding: ok");
    return 0;
}
