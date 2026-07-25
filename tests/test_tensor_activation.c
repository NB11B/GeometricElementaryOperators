#include "geo/tensor_activation.h"

#include <math.h>
#include <stdio.h>

static int close_enough(geo_real_t a, geo_real_t b, geo_real_t tolerance) {
    return fabs((double)(a - b)) <= (double)tolerance;
}

int main(void) {
    const geo_real_t x[4] = {(geo_real_t)-1.0, (geo_real_t)-0.25, (geo_real_t)0.5, (geo_real_t)2.0};
    const geo_real_t up[4] = {(geo_real_t)0.5, (geo_real_t)-2.0, (geo_real_t)1.5, (geo_real_t)0.25};
    const geo_real_t grad_out[4] = {(geo_real_t)1.0, (geo_real_t)-0.5, (geo_real_t)2.0, (geo_real_t)0.75};
    geo_real_t out[4];
    geo_real_t grad_x[4];
    geo_real_t grad_gate[4];
    geo_real_t grad_up[4];

    if (geo_tensor_gelu_forward(x, out, 4u) != GEO_TENSOR_OK) return 1;
    if (geo_tensor_gelu_vjp(x, grad_out, grad_x, 4u) != GEO_TENSOR_OK) return 2;

    for (size_t i = 0; i < 4u; ++i) {
        const geo_real_t h = (geo_real_t)1e-3;
        geo_real_t xp[4] = {x[0], x[1], x[2], x[3]};
        geo_real_t xm[4] = {x[0], x[1], x[2], x[3]};
        geo_real_t yp[4];
        geo_real_t ym[4];
        xp[i] += h;
        xm[i] -= h;
        geo_tensor_gelu_forward(xp, yp, 4u);
        geo_tensor_gelu_forward(xm, ym, 4u);
        const geo_real_t numerical = grad_out[i] * (yp[i] - ym[i]) / ((geo_real_t)2 * h);
        if (!close_enough(grad_x[i], numerical, (geo_real_t)2e-3)) return 3;
    }

    if (geo_tensor_silu_mul_forward(x, up, out, 4u) != GEO_TENSOR_OK) return 4;
    if (geo_tensor_silu_mul_vjp(x, up, grad_out, grad_gate, grad_up, 4u) != GEO_TENSOR_OK) return 5;

    for (size_t i = 0; i < 4u; ++i) {
        const geo_real_t sigmoid = (geo_real_t)1 / ((geo_real_t)1 + (geo_real_t)exp((double)-x[i]));
        const geo_real_t expected_up = grad_out[i] * x[i] * sigmoid;
        if (!close_enough(grad_up[i], expected_up, (geo_real_t)1e-5)) return 6;
    }

    if (geo_tensor_gelu_forward(NULL, out, 4u) != GEO_TENSOR_INVALID_ARGUMENT) return 7;
    if (geo_tensor_silu_mul_forward(x, up, out, 0u) != GEO_TENSOR_INVALID_ARGUMENT) return 8;

    puts("GEO_TENSOR_ACTIVATION: PASS");
    return 0;
}
