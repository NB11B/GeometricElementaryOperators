#include "geo/geb36.h"
#include "geo/native_generated.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define TOL ((geo_real_t)1e-12)
#else
#define TOL ((geo_real_t)1e-5f)
#endif

int main(void) {
    const geo_cl20_t a = geo_cl20_make(1, 2, -3, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make(-2, 4, 1, (geo_real_t)-0.25);
    const geo_cl20_t rotor = geo_cl20_make((geo_real_t)0.9238795, 0, 0, (geo_real_t)-0.3826834);
    const geo_cl20_t vector = geo_cl20_make(0, 1, 2, 0);
    int failures = 0;

    if (!geo_cl20_near(geo_native_add(a, b), geo_geb_addition(a, b), TOL)) ++failures;
    if (!geo_cl20_near(geo_native_cl20_product(a, b), geo_geb_geometric_product(a, b), TOL)) ++failures;
    if (fabs((double)(geo_native_vector_dot(a, b) - geo_geb_vector_dot(a, b))) > (double)TOL) ++failures;
    if (!geo_cl20_near(geo_native_rotor_action(rotor, vector), geo_geb_rotor_action(rotor, vector), TOL)) ++failures;

    if (failures != 0) {
        fprintf(stderr, "%d generated-native assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("Generated native kernels match the GEB reference API.");
    return EXIT_SUCCESS;
}
