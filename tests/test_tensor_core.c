#include "geo/tensor_core.h"

#include <math.h>
#include <stdio.h>

static int close_enough(geo_real_t a, geo_real_t b) {
    return fabs((double)(a - b)) < 1e-5;
}

int main(void) {
    const geo_real_t a[4] = {(geo_real_t)1, (geo_real_t)2, (geo_real_t)3, (geo_real_t)4};
    const geo_real_t b[4] = {(geo_real_t)5, (geo_real_t)6, (geo_real_t)7, (geo_real_t)8};
    geo_real_t out[4];
    geo_real_t grad_a[4];
    geo_real_t grad_b[4];
    const geo_real_t grad_out[4] = {(geo_real_t)1, (geo_real_t)1, (geo_real_t)1, (geo_real_t)1};
    const geo_real_t weight[2] = {(geo_real_t)1.5, (geo_real_t)0.5};
    const geo_real_t x[4] = {(geo_real_t)3, (geo_real_t)4, (geo_real_t)0, (geo_real_t)5};
    geo_real_t norm_out[4];
    geo_real_t inv_rms[2];
    geo_real_t grad_x[4];
    geo_real_t grad_weight[2];
    geo_tensor_norm_shape norm_shape = {2u, 2u};

    if (geo_tensor_add_forward(a, b, out, 4u) != GEO_TENSOR_OK) return 1;
    if (!close_enough(out[0], (geo_real_t)6) || !close_enough(out[3], (geo_real_t)12)) return 2;

    if (geo_tensor_mul_forward(a, b, out, 4u) != GEO_TENSOR_OK) return 3;
    if (!close_enough(out[1], (geo_real_t)12) || !close_enough(out[2], (geo_real_t)21)) return 4;

    if (geo_tensor_mul_vjp(a, b, grad_out, grad_a, grad_b, 4u) != GEO_TENSOR_OK) return 5;
    if (!close_enough(grad_a[2], (geo_real_t)7) || !close_enough(grad_b[2], (geo_real_t)3)) return 6;

    if (geo_tensor_scale_forward(a, (geo_real_t)0.5, out, 4u) != GEO_TENSOR_OK) return 7;
    if (!close_enough(out[3], (geo_real_t)2)) return 8;

    if (geo_tensor_rms_norm_forward(x, weight, (geo_real_t)1e-6, norm_out, inv_rms, norm_shape) != GEO_TENSOR_OK) return 9;
    if (!(inv_rms[0] > (geo_real_t)0) || !(inv_rms[1] > (geo_real_t)0)) return 10;

    if (geo_tensor_rms_norm_vjp(x, weight, grad_out, inv_rms, grad_x, grad_weight, norm_shape) != GEO_TENSOR_OK) return 11;
    if (!isfinite((double)grad_x[0]) || !isfinite((double)grad_weight[0])) return 12;

    puts("GEO_TENSOR_CORE: PASS");
    return 0;
}
