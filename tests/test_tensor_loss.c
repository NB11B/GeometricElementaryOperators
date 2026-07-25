#include "geo/tensor_loss.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_FD_EPS ((geo_real_t)1e-6)
#define GEO_FD_TOL ((geo_real_t)2e-7)
#define GEO_VALUE_TOL ((geo_real_t)1e-12)
#else
#define GEO_FD_EPS ((geo_real_t)1e-3f)
#define GEO_FD_TOL ((geo_real_t)4e-4f)
#define GEO_VALUE_TOL ((geo_real_t)3e-6f)
#endif

static void assert_close(geo_real_t actual, geo_real_t expected, geo_real_t tolerance) {
    const geo_real_t error = (geo_real_t)fabs((double)(actual - expected));
    assert(error <= tolerance);
}

static geo_real_t reference_row_loss(const geo_real_t *row, size_t classes, size_t target) {
    geo_real_t max_logit = row[0];
    for (size_t class_index = 1u; class_index < classes; ++class_index) {
        if (row[class_index] > max_logit) {
            max_logit = row[class_index];
        }
    }
    geo_real_t sum = (geo_real_t)0;
    for (size_t class_index = 0u; class_index < classes; ++class_index) {
        sum += (geo_real_t)exp((double)(row[class_index] - max_logit));
    }
    return max_logit + (geo_real_t)log((double)sum) - row[target];
}

static geo_real_t compute_loss(
    const geo_real_t *logits,
    const int64_t *targets,
    int64_t ignore_index,
    geo_tensor_cross_entropy_shape shape
) {
    geo_real_t probabilities[12];
    geo_real_t loss;
    geo_real_t normalizer;
    assert(geo_tensor_cross_entropy_forward(
        logits, targets, ignore_index, &loss, probabilities, &normalizer, shape
    ) == GEO_TENSOR_OK);
    return loss;
}

static void test_forward_and_ignore_index(void) {
    const geo_tensor_cross_entropy_shape shape = {4u, 3u};
    const geo_real_t logits[12] = {
        (geo_real_t)2.0, (geo_real_t)0.5, (geo_real_t)-1.0,
        (geo_real_t)-0.2, (geo_real_t)0.3, (geo_real_t)1.4,
        (geo_real_t)0.7, (geo_real_t)-0.5, (geo_real_t)0.1,
        (geo_real_t)-1.0, (geo_real_t)1.2, (geo_real_t)0.4
    };
    const int64_t targets[4] = {0, 2, -1, 1};
    geo_real_t probabilities[12];
    geo_real_t loss;
    geo_real_t normalizer;

    assert(geo_tensor_cross_entropy_forward(
        logits, targets, -1, &loss, probabilities, &normalizer, shape
    ) == GEO_TENSOR_OK);

    const geo_real_t expected = (
        reference_row_loss(&logits[0], 3u, 0u) +
        reference_row_loss(&logits[3], 3u, 2u) +
        reference_row_loss(&logits[9], 3u, 1u)
    ) / (geo_real_t)3;
    assert_close(loss, expected, GEO_VALUE_TOL);
    assert_close(normalizer, (geo_real_t)(1.0 / 3.0), GEO_VALUE_TOL);

    assert_close(probabilities[6], (geo_real_t)0, GEO_VALUE_TOL);
    assert_close(probabilities[7], (geo_real_t)0, GEO_VALUE_TOL);
    assert_close(probabilities[8], (geo_real_t)0, GEO_VALUE_TOL);
    for (size_t row = 0u; row < shape.rows; ++row) {
        if (targets[row] == -1) {
            continue;
        }
        geo_real_t sum = (geo_real_t)0;
        for (size_t class_index = 0u; class_index < shape.classes; ++class_index) {
            sum += probabilities[row * shape.classes + class_index];
        }
        assert_close(sum, (geo_real_t)1, GEO_VALUE_TOL);
    }
}

static void test_vjp_finite_difference(void) {
    const geo_tensor_cross_entropy_shape shape = {3u, 4u};
    geo_real_t logits[12] = {
        (geo_real_t)0.2, (geo_real_t)-0.1, (geo_real_t)0.7, (geo_real_t)1.1,
        (geo_real_t)-0.8, (geo_real_t)0.4, (geo_real_t)0.9, (geo_real_t)-0.3,
        (geo_real_t)0.5, (geo_real_t)0.2, (geo_real_t)-0.6, (geo_real_t)0.1
    };
    const int64_t targets[3] = {3, -1, 0};
    geo_real_t probabilities[12];
    geo_real_t loss;
    geo_real_t normalizer;
    geo_real_t gradients[12];
    const geo_real_t upstream = (geo_real_t)0.7;

    assert(geo_tensor_cross_entropy_forward(
        logits, targets, -1, &loss, probabilities, &normalizer, shape
    ) == GEO_TENSOR_OK);
    assert(geo_tensor_cross_entropy_vjp(
        probabilities, targets, -1, normalizer, upstream, gradients, shape
    ) == GEO_TENSOR_OK);

    for (size_t index = 0u; index < 12u; ++index) {
        const geo_real_t original = logits[index];
        logits[index] = original + GEO_FD_EPS;
        const geo_real_t plus = compute_loss(logits, targets, -1, shape) * upstream;
        logits[index] = original - GEO_FD_EPS;
        const geo_real_t minus = compute_loss(logits, targets, -1, shape) * upstream;
        logits[index] = original;
        assert_close(
            gradients[index],
            (plus - minus) / ((geo_real_t)2 * GEO_FD_EPS),
            GEO_FD_TOL
        );
    }
}

static void test_all_ignored(void) {
    const geo_tensor_cross_entropy_shape shape = {2u, 2u};
    const geo_real_t logits[4] = {
        (geo_real_t)1, (geo_real_t)2,
        (geo_real_t)3, (geo_real_t)4
    };
    const int64_t targets[2] = {-1, -1};
    geo_real_t probabilities[4];
    geo_real_t loss;
    geo_real_t normalizer;
    geo_real_t gradients[4];

    assert(geo_tensor_cross_entropy_forward(
        logits, targets, -1, &loss, probabilities, &normalizer, shape
    ) == GEO_TENSOR_OK);
    assert(isnan((double)loss));
    assert_close(normalizer, (geo_real_t)0, GEO_VALUE_TOL);
    assert(geo_tensor_cross_entropy_vjp(
        probabilities, targets, -1, normalizer, (geo_real_t)1, gradients, shape
    ) == GEO_TENSOR_OK);
    for (size_t index = 0u; index < 4u; ++index) {
        assert_close(gradients[index], (geo_real_t)0, GEO_VALUE_TOL);
    }
}

static void test_invalid_target(void) {
    const geo_tensor_cross_entropy_shape shape = {1u, 2u};
    const geo_real_t logits[2] = {(geo_real_t)1, (geo_real_t)2};
    const int64_t targets[1] = {2};
    geo_real_t probabilities[2];
    geo_real_t loss;
    geo_real_t normalizer;
    assert(geo_tensor_cross_entropy_forward(
        logits, targets, -1, &loss, probabilities, &normalizer, shape
    ) == GEO_TENSOR_INVALID_ARGUMENT);
}

int main(void) {
    test_forward_and_ignore_index();
    test_vjp_finite_difference();
    test_all_ignored();
    test_invalid_target();
    puts("tensor_loss: ok");
    return 0;
}
