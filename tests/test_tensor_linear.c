#include "geo/tensor_linear.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

static int close_enough(geo_real_t a, geo_real_t b) {
    const geo_real_t tolerance = (geo_real_t)1e-5;
    return fabs((double)(a - b)) <= (double)tolerance;
}

static int check_array(const char *name, const geo_real_t *actual, const geo_real_t *expected, size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        if (!close_enough(actual[index], expected[index])) {
            fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n", name, index,
                    (double)actual[index], (double)expected[index]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const geo_tensor_linear_shape shape = {2u, 3u, 2u};
    const geo_real_t x[6] = {
        (geo_real_t)1, (geo_real_t)2, (geo_real_t)3,
        (geo_real_t)4, (geo_real_t)5, (geo_real_t)6
    };
    const geo_real_t weight[6] = {
        (geo_real_t)1, (geo_real_t)0, (geo_real_t)-1,
        (geo_real_t)2, (geo_real_t)3, (geo_real_t)4
    };
    const geo_real_t grad_y[4] = {
        (geo_real_t)1, (geo_real_t)2,
        (geo_real_t)3, (geo_real_t)4
    };
    const geo_real_t expected_y[4] = {
        (geo_real_t)-2, (geo_real_t)20,
        (geo_real_t)-2, (geo_real_t)47
    };
    const geo_real_t expected_grad_x[6] = {
        (geo_real_t)5, (geo_real_t)6, (geo_real_t)7,
        (geo_real_t)11, (geo_real_t)12, (geo_real_t)13
    };
    const geo_real_t expected_grad_weight[6] = {
        (geo_real_t)13, (geo_real_t)17, (geo_real_t)21,
        (geo_real_t)18, (geo_real_t)24, (geo_real_t)30
    };
    geo_real_t y[4];
    geo_real_t grad_x[6];
    geo_real_t grad_weight[6];

    if (geo_tensor_linear_forward(x, weight, y, shape) != GEO_TENSOR_OK) {
        fprintf(stderr, "forward returned failure\n");
        return 1;
    }
    if (!check_array("y", y, expected_y, 4u)) {
        return 1;
    }

    if (geo_tensor_linear_vjp(x, weight, grad_y, grad_x, grad_weight, shape) != GEO_TENSOR_OK) {
        fprintf(stderr, "vjp returned failure\n");
        return 1;
    }
    if (!check_array("grad_x", grad_x, expected_grad_x, 6u) ||
        !check_array("grad_weight", grad_weight, expected_grad_weight, 6u)) {
        return 1;
    }

    if (geo_tensor_linear_forward(NULL, weight, y, shape) != GEO_TENSOR_INVALID_ARGUMENT) {
        fprintf(stderr, "null argument was not rejected\n");
        return 1;
    }

    puts("GEO_TENSOR_LINEAR: PASS");
    return 0;
}
