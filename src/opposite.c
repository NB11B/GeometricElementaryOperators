#include "geo/opposite.h"

geo_opposite_t geo_opposite_zero(void) {
    geo_opposite_t result;
    result.forward = geo_cl20_zero();
    result.reverse = geo_cl20_zero();
    return result;
}

geo_opposite_t geo_opposite_identity(void) {
    geo_opposite_t result;
    result.forward = geo_cl20_one();
    result.reverse = geo_cl20_one();
    return result;
}

geo_opposite_t geo_opposite_from_cl20(geo_cl20_t value) {
    geo_opposite_t result;
    result.forward = value;
    result.reverse = geo_cl20_reverse(value);
    return result;
}

geo_opposite_t geo_opposite_mul(geo_opposite_t a, geo_opposite_t b) {
    geo_opposite_t result;
    result.forward = geo_cl20_mul(a.forward, b.forward);
    result.reverse = geo_cl20_mul(b.reverse, a.reverse);
    return result;
}

bool geo_opposite_is_consistent(
    geo_opposite_t value,
    geo_real_t tolerance
) {
    return geo_cl20_near(
        value.reverse,
        geo_cl20_reverse(value.forward),
        tolerance
    );
}
