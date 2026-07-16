#include "geo/geb36.h"
#include "geo/structured.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_RANDOM_TOLERANCE ((geo_real_t)1e-10)
#else
#define GEO_RANDOM_TOLERANCE ((geo_real_t)2e-4f)
#endif

#define GEO_RANDOM_CASES 10000u

static uint32_t rng_state = UINT32_C(0x6d2b79f5);
static int failures = 0;

static uint32_t rng_next(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static geo_real_t random_real(void) {
    const double unit = (double)(rng_next() & UINT32_C(0x00ffffff)) / 16777215.0;
    return (geo_real_t)(-2.0 + 4.0 * unit);
}

static geo_cl20_t random_multivector(void) {
    return geo_cl20_make(random_real(), random_real(), random_real(), random_real());
}

static geo_cl20_t random_vector(void) {
    return geo_cl20_make((geo_real_t)0, random_real(), random_real(), (geo_real_t)0);
}

static void fail(const char *name, size_t index) {
    if (failures < 16) {
        fprintf(stderr, "FAIL: %s at case %zu\n", name, index);
    }
    ++failures;
}

static void test_reverse_antihomomorphism(void) {
    size_t i;
    for (i = 0u; i < GEO_RANDOM_CASES; ++i) {
        const geo_cl20_t a = random_multivector();
        const geo_cl20_t b = random_multivector();
        const geo_cl20_t lhs = geo_cl20_reverse(geo_cl20_mul(a, b));
        const geo_cl20_t rhs = geo_cl20_mul(geo_cl20_reverse(b), geo_cl20_reverse(a));
        if (!geo_cl20_near(lhs, rhs, GEO_RANDOM_TOLERANCE)) {
            fail("reverse(AB)=reverse(B)reverse(A)", i);
        }
    }
}

static void test_vector_decomposition(void) {
    size_t i;
    for (i = 0u; i < GEO_RANDOM_CASES; ++i) {
        const geo_cl20_t a = random_vector();
        const geo_cl20_t b = random_vector();
        const geo_cl20_t product = geo_cl20_mul(a, b);
        const geo_cl20_t reconstructed = geo_cl20_add(
            geo_cl20_make(geo_cl20_vector_dot(a, b), (geo_real_t)0, (geo_real_t)0, (geo_real_t)0),
            geo_cl20_make((geo_real_t)0, (geo_real_t)0, (geo_real_t)0, geo_cl20_vector_wedge(a, b))
        );
        if (!geo_cl20_near(product, reconstructed, GEO_RANDOM_TOLERANCE)) {
            fail("ab=dot+wedge", i);
        }
    }
}

static void test_unipotent_addition(void) {
    size_t i;
    for (i = 0u; i < GEO_RANDOM_CASES; ++i) {
        const geo_cl20_t a = random_multivector();
        const geo_cl20_t b = random_multivector();
        const geo_unipotent_t composed = geo_unipotent_mul(
            geo_unipotent_from_cl20(a),
            geo_unipotent_from_cl20(b)
        );
        if (!geo_cl20_near(
                geo_unipotent_extract(composed),
                geo_cl20_add(a, b),
                GEO_RANDOM_TOLERANCE)) {
            fail("U(A)U(B)=U(A+B)", i);
        }
    }
}

static void test_hadamard_paths(void) {
    size_t i;
    for (i = 0u; i < GEO_RANDOM_CASES; ++i) {
        const geo_cl20_t a = random_multivector();
        const geo_cl20_t b = random_multivector();
        const geo_ordered_pair_t ordered = geo_ordered_products(a, b);
        const geo_hadamard_pair_t exact = geo_hadamard_mix_exact(ordered);
        const geo_hadamard_pair_t projective = geo_hadamard_mix_projective(ordered);
        const geo_cl20_t commutator = geo_geb_commutator(a, b);
        const geo_cl20_t anticommutator = geo_geb_anticommutator(a, b);

        if (!geo_cl20_near(exact.antisymmetric.represented, commutator, GEO_RANDOM_TOLERANCE)) {
            fail("exact antisymmetric mixer", i);
        }
        if (!geo_cl20_near(exact.symmetric.represented, anticommutator, GEO_RANDOM_TOLERANCE)) {
            fail("exact symmetric mixer", i);
        }
        if (!geo_cl20_near(
                projective.antisymmetric.represented,
                geo_cl20_scale(commutator, (geo_real_t)2),
                GEO_RANDOM_TOLERANCE)) {
            fail("projective antisymmetric mixer", i);
        }
        if (!geo_cl20_near(
                projective.symmetric.represented,
                geo_cl20_scale(anticommutator, (geo_real_t)2),
                GEO_RANDOM_TOLERANCE)) {
            fail("projective symmetric mixer", i);
        }
    }
}

static void test_rotor_vector_grade(void) {
    size_t i;
    for (i = 0u; i < GEO_RANDOM_CASES; ++i) {
        const geo_real_t angle = random_real();
        const geo_real_t half = (geo_real_t)0.5 * angle;
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
        const geo_real_t c = cos(half);
        const geo_real_t s = sin(half);
#else
        const geo_real_t c = cosf(half);
        const geo_real_t s = sinf(half);
#endif
        const geo_cl20_t rotor = geo_cl20_make(c, (geo_real_t)0, (geo_real_t)0, -s);
        const geo_cl20_t vector = random_vector();
        const geo_cl20_t rotated = geo_geb_rotor_action(rotor, vector);
        if (fabs((double)rotated.scalar) > (double)GEO_RANDOM_TOLERANCE ||
            fabs((double)rotated.e12) > (double)GEO_RANDOM_TOLERANCE) {
            fail("rotor action preserves vector grade", i);
        }
    }
}

int main(void) {
    test_reverse_antihomomorphism();
    test_vector_decomposition();
    test_unipotent_addition();
    test_hadamard_paths();
    test_rotor_vector_grade();

    if (failures != 0) {
        fprintf(stderr, "%d randomized assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    printf("Randomized validation passed: %u cases per property.\n", GEO_RANDOM_CASES);
    return EXIT_SUCCESS;
}
