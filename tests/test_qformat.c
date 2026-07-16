#include "geo/fixed.h"
#include "geo/fixed_geb36.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    geo_fixed_t one;
    geo_fixed_t two;
    geo_fixed_t half;
    geo_fixed_cl20_t a = {0, 0, 0, 0};
    geo_fixed_cl20_t b = {0, 0, 0, 0};
    geo_fixed_geb_result_t result;

    if (geo_fixed_from_double(1.0, &one) != GEO_FIXED_OK ||
        geo_fixed_from_double(2.0, &two) != GEO_FIXED_OK ||
        geo_fixed_div(one, two, &half) != GEO_FIXED_OK) {
        return EXIT_FAILURE;
    }
    if (geo_fixed_to_double(half) < 0.49 || geo_fixed_to_double(half) > 0.51) {
        return EXIT_FAILURE;
    }

    a.e1 = half;
    b.e2 = half;
    if (geo_fixed_geb36_execute(GEO_GEB_GEOMETRIC_PRODUCT, a, b, a, &result) !=
        GEO_FIXED_OK) {
        return EXIT_FAILURE;
    }
    if (result.kind != GEO_FIXED_RESULT_CL20) return EXIT_FAILURE;
    puts("Q-format safety test passed.");
    return EXIT_SUCCESS;
}
