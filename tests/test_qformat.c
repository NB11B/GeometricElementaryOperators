#include "geo/fixed.h"
#include "geo/fixed_geb36.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    geo_fixed_t one;
    geo_fixed_t half;
    geo_fixed_t quotient;
    geo_fixed_cl20_t a = {0, 0, 0, 0};
    geo_fixed_cl20_t b = {0, 0, 0, 0};
    geo_fixed_geb_result_t result;

    /*
     * Do not require 2.0 to be representable: in Q30 the positive range is
     * [0, 2), while 1.0 and 0.5 remain exactly representable.
     */
    if (geo_fixed_from_double(1.0, &one) != GEO_FIXED_OK ||
        geo_fixed_from_double(0.5, &half) != GEO_FIXED_OK ||
        geo_fixed_div(half, one, &quotient) != GEO_FIXED_OK) {
        return EXIT_FAILURE;
    }
    if (!isfinite(geo_fixed_to_double(quotient)) ||
        fabs(geo_fixed_to_double(quotient) - 0.5) > 1e-9) {
        return EXIT_FAILURE;
    }

    a.e1 = half;
    b.e2 = half;
    if (geo_fixed_geb36_execute(GEO_GEB_GEOMETRIC_PRODUCT, a, b, a, &result) !=
        GEO_FIXED_OK) {
        return EXIT_FAILURE;
    }
    if (result.kind != GEO_FIXED_RESULT_CL20) return EXIT_FAILURE;
    if (fabs(geo_fixed_to_double(result.as.cl20.e12) - 0.25) > 1e-9) {
        return EXIT_FAILURE;
    }

    puts("Q-format safety test passed.");
    return EXIT_SUCCESS;
}
