#include "geo/fixed_control.h"

#include <limits.h>
#include <stddef.h>

static geo_fixed_status_t geo_fixed_control_add(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
) {
    const int64_t value = (int64_t)left + (int64_t)right;
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) {
        return GEO_FIXED_OVERFLOW;
    }
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t geo_fixed_control_sub(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
) {
    const int64_t value = (int64_t)left - (int64_t)right;
    if (output == NULL || value < INT32_MIN || value > INT32_MAX) {
        return GEO_FIXED_OVERFLOW;
    }
    *output = (geo_fixed_t)value;
    return GEO_FIXED_OK;
}

static geo_fixed_status_t geo_fixed_control_dot2(
    geo_fixed_t left0,
    geo_fixed_t right0,
    geo_fixed_t left1,
    geo_fixed_t right1,
    geo_fixed_t *output
) {
    geo_fixed_t product0;
    geo_fixed_t product1;
    geo_fixed_status_t status;

    if (output == NULL) return GEO_FIXED_OVERFLOW;
    status = geo_fixed_mul(left0, right0, &product0);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_mul(left1, right1, &product1);
    if (status != GEO_FIXED_OK) return status;
    return geo_fixed_control_add(product0, product1, output);
}

geo_fixed_status_t geo_fixed_control_apply(
    geo_fixed_m2_t left,
    geo_fixed_m2_t right,
    geo_fixed_m2_t *output
) {
    geo_fixed_m2_t product;
    geo_fixed_m2_t result;
    geo_fixed_status_t status;

    if (output == NULL) return GEO_FIXED_OVERFLOW;

    status = geo_fixed_control_dot2(
        left.m00,
        right.m00,
        left.m01,
        right.m10,
        &product.m00
    );
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_control_dot2(
        left.m00,
        right.m01,
        left.m01,
        right.m11,
        &product.m01
    );
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_control_dot2(
        left.m10,
        right.m00,
        left.m11,
        right.m10,
        &product.m10
    );
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_control_dot2(
        left.m10,
        right.m01,
        left.m11,
        right.m11,
        &product.m11
    );
    if (status != GEO_FIXED_OK) return status;

    status = geo_fixed_control_sub(product.m00, left.m00, &result.m00);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_control_sub(product.m01, left.m01, &result.m01);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_control_sub(product.m10, left.m10, &result.m10);
    if (status != GEO_FIXED_OK) return status;
    status = geo_fixed_control_sub(product.m11, left.m11, &result.m11);
    if (status != GEO_FIXED_OK) return status;

    *output = result;
    return GEO_FIXED_OK;
}
