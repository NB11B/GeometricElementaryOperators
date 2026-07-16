#include "geo/control.h"

#include <math.h>

geo_mat2_t geo_mat2_make(
    geo_real_t m00,
    geo_real_t m01,
    geo_real_t m10,
    geo_real_t m11
) {
    geo_mat2_t result;
    result.m00 = m00;
    result.m01 = m01;
    result.m10 = m10;
    result.m11 = m11;
    return result;
}

geo_mat2_t geo_mat2_zero(void) {
    return geo_mat2_make(
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)0
    );
}

geo_mat2_t geo_mat2_identity(void) {
    return geo_mat2_make(
        (geo_real_t)1,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)1
    );
}

geo_mat2_t geo_mat2_neg_identity(void) {
    return geo_mat2_make(
        (geo_real_t)-1,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)-1
    );
}

geo_mat2_t geo_mat2_e11(void) {
    return geo_mat2_make(
        (geo_real_t)1,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)0
    );
}

geo_mat2_t geo_mat2_e12(void) {
    return geo_mat2_make(
        (geo_real_t)0,
        (geo_real_t)1,
        (geo_real_t)0,
        (geo_real_t)0
    );
}

geo_mat2_t geo_mat2_e21(void) {
    return geo_mat2_make(
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)1,
        (geo_real_t)0
    );
}

geo_mat2_t geo_mat2_e22(void) {
    return geo_mat2_make(
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)1
    );
}

geo_mat2_t geo_mat2_exchange(void) {
    return geo_mat2_make(
        (geo_real_t)0,
        (geo_real_t)1,
        (geo_real_t)1,
        (geo_real_t)0
    );
}

geo_mat2_t geo_mat2_add(geo_mat2_t a, geo_mat2_t b) {
    return geo_mat2_make(
        a.m00 + b.m00,
        a.m01 + b.m01,
        a.m10 + b.m10,
        a.m11 + b.m11
    );
}

geo_mat2_t geo_mat2_sub(geo_mat2_t a, geo_mat2_t b) {
    return geo_mat2_make(
        a.m00 - b.m00,
        a.m01 - b.m01,
        a.m10 - b.m10,
        a.m11 - b.m11
    );
}

geo_mat2_t geo_mat2_mul(geo_mat2_t a, geo_mat2_t b) {
    return geo_mat2_make(
        a.m00 * b.m00 + a.m01 * b.m10,
        a.m00 * b.m01 + a.m01 * b.m11,
        a.m10 * b.m00 + a.m11 * b.m10,
        a.m10 * b.m01 + a.m11 * b.m11
    );
}

geo_mat2_t geo_control_gc(geo_mat2_t x, geo_mat2_t y) {
    return geo_mat2_sub(geo_mat2_mul(x, y), x);
}

bool geo_mat2_near(geo_mat2_t a, geo_mat2_t b, geo_real_t tolerance) {
    return
        fabs((double)(a.m00 - b.m00)) <= (double)tolerance &&
        fabs((double)(a.m01 - b.m01)) <= (double)tolerance &&
        fabs((double)(a.m10 - b.m10)) <= (double)tolerance &&
        fabs((double)(a.m11 - b.m11)) <= (double)tolerance;
}
