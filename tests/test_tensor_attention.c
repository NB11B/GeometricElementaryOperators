#include "geo/tensor_attention.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_FD_EPS ((geo_real_t)1e-6)
#define GEO_FD_TOL ((geo_real_t)2e-6)
#define GEO_VALUE_TOL ((geo_real_t)1e-12)
#else
#define GEO_FD_EPS ((geo_real_t)1e-3f)
#define GEO_FD_TOL ((geo_real_t)4e-3f)
#define GEO_VALUE_TOL ((geo_real_t)3e-5f)
#endif

static void assert_close(geo_real_t actual, geo_real_t expected, geo_real_t tolerance) {
    const geo_real_t error = (geo_real_t)fabs((double)(actual - expected));
    assert(error <= tolerance);
}

static geo_real_t attention_loss(
    const geo_real_t *q,
    const geo_real_t *k,
    const geo_real_t *v,
    const geo_real_t *upstream,
    geo_tensor_attention_shape shape
) {
    geo_real_t out[6];
    geo_real_t probabilities[9];
    assert(geo_tensor_causal_attention_forward(
        q, k, v, out, probabilities, shape
    ) == GEO_TENSOR_OK);
    geo_real_t loss = (geo_real_t)0;
    for (size_t index = 0u; index < 6u; ++index) {
        loss += out[index] * upstream[index];
    }
    return loss;
}

static void test_uniform_causal_forward(void) {
    const geo_tensor_attention_shape shape = {1u, 3u, 2u};
    const geo_real_t q[6] = {0};
    const geo_real_t k[6] = {0};
    const geo_real_t v[6] = {
        (geo_real_t)1, (geo_real_t)2,
        (geo_real_t)3, (geo_real_t)4,
        (geo_real_t)5, (geo_real_t)8
    };
    geo_real_t out[6];
    geo_real_t probabilities[9];

    assert(geo_tensor_causal_attention_forward(
        q, k, v, out, probabilities, shape
    ) == GEO_TENSOR_OK);

    assert_close(probabilities[0], (geo_real_t)1, GEO_VALUE_TOL);
    assert_close(probabilities[1], (geo_real_t)0, GEO_VALUE_TOL);
    assert_close(probabilities[2], (geo_real_t)0, GEO_VALUE_TOL);
    assert_close(probabilities[3], (geo_real_t)0.5, GEO_VALUE_TOL);
    assert_close(probabilities[4], (geo_real_t)0.5, GEO_VALUE_TOL);
    assert_close(probabilities[5], (geo_real_t)0, GEO_VALUE_TOL);
    assert_close(probabilities[6], (geo_real_t)(1.0 / 3.0), GEO_VALUE_TOL);
    assert_close(probabilities[7], (geo_real_t)(1.0 / 3.0), GEO_VALUE_TOL);
    assert_close(probabilities[8], (geo_real_t)(1.0 / 3.0), GEO_VALUE_TOL);

    assert_close(out[0], (geo_real_t)1, GEO_VALUE_TOL);
    assert_close(out[1], (geo_real_t)2, GEO_VALUE_TOL);
    assert_close(out[2], (geo_real_t)2, GEO_VALUE_TOL);
    assert_close(out[3], (geo_real_t)3, GEO_VALUE_TOL);
    assert_close(out[4], (geo_real_t)3, GEO_VALUE_TOL);
    assert_close(out[5], (geo_real_t)(14.0 / 3.0), GEO_VALUE_TOL);
}

static void test_vjp_finite_difference(void) {
    const geo_tensor_attention_shape shape = {1u, 3u, 2u};
    geo_real_t q[6] = {
        (geo_real_t)0.2, (geo_real_t)-0.4,
        (geo_real_t)0.7, (geo_real_t)0.1,
        (geo_real_t)-0.3, (geo_real_t)0.8
    };
    geo_real_t k[6] = {
        (geo_real_t)-0.5, (geo_real_t)0.6,
        (geo_real_t)0.9, (geo_real_t)-0.2,
        (geo_real_t)0.4, (geo_real_t)0.3
    };
    geo_real_t v[6] = {
        (geo_real_t)0.3, (geo_real_t)-0.7,
        (geo_real_t)1.1, (geo_real_t)0.2,
        (geo_real_t)-0.4, (geo_real_t)0.9
    };
    const geo_real_t upstream[6] = {
        (geo_real_t)0.5, (geo_real_t)-0.1,
        (geo_real_t)-0.3, (geo_real_t)0.8,
        (geo_real_t)0.7, (geo_real_t)-0.6
    };

    geo_real_t out[6];
    geo_real_t probabilities[9];
    geo_real_t grad_q[6];
    geo_real_t grad_k[6];
    geo_real_t grad_v[6];
    assert(geo_tensor_causal_attention_forward(
        q, k, v, out, probabilities, shape
    ) == GEO_TENSOR_OK);
    assert(geo_tensor_causal_attention_vjp(
        q, k, v, probabilities, upstream,
        grad_q, grad_k, grad_v, shape
    ) == GEO_TENSOR_OK);

    for (size_t index = 0u; index < 6u; ++index) {
        const geo_real_t original = q[index];
        q[index] = original + GEO_FD_EPS;
        const geo_real_t plus = attention_loss(q, k, v, upstream, shape);
        q[index] = original - GEO_FD_EPS;
        const geo_real_t minus = attention_loss(q, k, v, upstream, shape);
        q[index] = original;
        assert_close(
            grad_q[index],
            (plus - minus) / ((geo_real_t)2 * GEO_FD_EPS),
            GEO_FD_TOL
        );
    }

    for (size_t index = 0u; index < 6u; ++index) {
        const geo_real_t original = k[index];
        k[index] = original + GEO_FD_EPS;
        const geo_real_t plus = attention_loss(q, k, v, upstream, shape);
        k[index] = original - GEO_FD_EPS;
        const geo_real_t minus = attention_loss(q, k, v, upstream, shape);
        k[index] = original;
        assert_close(
            grad_k[index],
            (plus - minus) / ((geo_real_t)2 * GEO_FD_EPS),
            GEO_FD_TOL
        );
    }

    for (size_t index = 0u; index < 6u; ++index) {
        const geo_real_t original = v[index];
        v[index] = original + GEO_FD_EPS;
        const geo_real_t plus = attention_loss(q, k, v, upstream, shape);
        v[index] = original - GEO_FD_EPS;
        const geo_real_t minus = attention_loss(q, k, v, upstream, shape);
        v[index] = original;
        assert_close(
            grad_v[index],
            (plus - minus) / ((geo_real_t)2 * GEO_FD_EPS),
            GEO_FD_TOL
        );
    }
}

static void test_invalid_shape(void) {
    const geo_real_t values[1] = {(geo_real_t)0};
    geo_real_t out[1];
    geo_real_t probabilities[1];
    const geo_tensor_attention_shape invalid = {0u, 1u, 1u};
    assert(geo_tensor_causal_attention_forward(
        values, values, values, out, probabilities, invalid
    ) == GEO_TENSOR_INVALID_ARGUMENT);
}

int main(void) {
    test_uniform_causal_forward();
    test_vjp_finite_difference();
    test_invalid_shape();
    puts("tensor_attention: ok");
    return 0;
}
