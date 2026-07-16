#include "geo/control.h"

#include <math.h>

geo_mat2_t geo_mat2_make(geo_real_t m00, geo_real_t m01, geo_real_t m10, geo_real_t m11) {
    geo_mat2_t result;
    result.m00 = m00; result.m01 = m01; result.m10 = m10; result.m11 = m11;
    return result;
}

geo_mat2_t geo_mat2_zero(void) {
    return geo_mat2_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, (geo_real_t)0);
}
geo_mat2_t geo_mat2_identity(void) {
    return geo_mat2_make((geo_real_t)1, (geo_real_t)0, (geo_real_t)0, (geo_real_t)1);
}
geo_mat2_t geo_mat2_neg_identity(void) {
    return geo_mat2_make((geo_real_t)-1, (geo_real_t)0, (geo_real_t)0, (geo_real_t)-1);
}
geo_mat2_t geo_mat2_e11(void) {
    return geo_mat2_make((geo_real_t)1, (geo_real_t)0, (geo_real_t)0, (geo_real_t)0);
}
geo_mat2_t geo_mat2_e12(void) {
    return geo_mat2_make((geo_real_t)0, (geo_real_t)1, (geo_real_t)0, (geo_real_t)0);
}
geo_mat2_t geo_mat2_e21(void) {
    return geo_mat2_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)1, (geo_real_t)0);
}
geo_mat2_t geo_mat2_e22(void) {
    return geo_mat2_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, (geo_real_t)1);
}
geo_mat2_t geo_mat2_exchange(void) {
    return geo_mat2_make((geo_real_t)0, (geo_real_t)1, (geo_real_t)1, (geo_real_t)0);
}
geo_mat2_t geo_mat2_add(geo_mat2_t a, geo_mat2_t b) {
    return geo_mat2_make(a.m00 + b.m00, a.m01 + b.m01, a.m10 + b.m10, a.m11 + b.m11);
}
geo_mat2_t geo_mat2_sub(geo_mat2_t a, geo_mat2_t b) {
    return geo_mat2_make(a.m00 - b.m00, a.m01 - b.m01, a.m10 - b.m10, a.m11 - b.m11);
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
    const double t = (double)tolerance;
    const double d00 = (double)a.m00 - (double)b.m00;
    const double d01 = (double)a.m01 - (double)b.m01;
    const double d10 = (double)a.m10 - (double)b.m10;
    const double d11 = (double)a.m11 - (double)b.m11;
    if (!isfinite(t) || t < 0.0 || !isfinite(d00) || !isfinite(d01) ||
        !isfinite(d10) || !isfinite(d11)) return false;
    return fabs(d00) <= t && fabs(d01) <= t && fabs(d10) <= t && fabs(d11) <= t;
}
