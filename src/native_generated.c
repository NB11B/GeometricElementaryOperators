#include "geo/native_generated.h"

geo_cl20_t geo_native_add(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_make(
        a.scalar + b.scalar,
        a.e1 + b.e1,
        a.e2 + b.e2,
        a.e12 + b.e12
    );
}

geo_real_t geo_native_vector_dot(geo_cl20_t a, geo_cl20_t b) {
    return a.e1 * b.e1 + a.e2 * b.e2;
}

geo_cl20_t geo_native_cl20_product(geo_cl20_t a, geo_cl20_t b) {
    return geo_cl20_make(
        a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 - a.e12 * b.e12,
        a.scalar * b.e1 + a.e1 * b.scalar - a.e2 * b.e12 + a.e12 * b.e2,
        a.scalar * b.e2 + a.e2 * b.scalar + a.e1 * b.e12 - a.e12 * b.e1,
        a.scalar * b.e12 + a.e12 * b.scalar + a.e1 * b.e2 - a.e2 * b.e1
    );
}

geo_cl20_t geo_native_rotor_action(geo_cl20_t rotor, geo_cl20_t value) {
    geo_cl20_t reverse = rotor;
    reverse.e12 = -reverse.e12;
    return geo_native_cl20_product(geo_native_cl20_product(rotor, value), reverse);
}
