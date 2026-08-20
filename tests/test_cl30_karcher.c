/**
 * @file test_cl30_karcher.c
 * @brief Unit tests for Cl(3,0) / Spin(3) Rotor and Riemannian Karcher Mean Engine.
 */

#include "geo/cl30_karcher.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5f

static void test_rotor_from_bivector_and_multiply() {
    printf("Running test_rotor_from_bivector_and_multiply...\n");
    // Identity check
    geo_rotor3_t r_id = geo_rotor3_from_bivector(0.0f, 0.0f, 0.0f);
    assert(fabsf(r_id.s - 1.0f) < EPSILON);
    assert(fabsf(r_id.b0) < EPSILON);
    assert(fabsf(r_id.b1) < EPSILON);
    assert(fabsf(r_id.b2) < EPSILON);

    // 90 deg rotation around z-axis (e12 bivector)
    // B = (pi/2, 0, 0) -> R = cos(pi/4) - e12*sin(pi/4)
    geo_rotor3_t r_z90 = geo_rotor3_from_bivector(1.57079632679f, 0.0f, 0.0f);
    assert(fabsf(r_z90.s - 0.70710678f) < 1e-4f);
    assert(fabsf(r_z90.b0 - (-0.70710678f)) < 1e-4f);

    // Inverse check: R * R~ = Identity
    geo_rotor3_t r_inv = geo_rotor3_reverse(r_z90);
    geo_rotor3_t prod = geo_rotor3_multiply(r_z90, r_inv);
    assert(fabsf(prod.s - 1.0f) < 1e-4f);
    assert(fabsf(prod.b0) < 1e-4f);
    assert(fabsf(prod.b1) < 1e-4f);
    assert(fabsf(prod.b2) < 1e-4f);

    printf("  PASS: test_rotor_from_bivector_and_multiply\n");
}

static void test_log_exp_round_trip() {
    printf("Running test_log_exp_round_trip...\n");
    geo_vec3_t v = {0.3f, -0.5f, 0.2f};
    geo_rotor3_t r = geo_rotor3_exp(v);
    geo_vec3_t v_rec = geo_rotor3_log(r);

    assert(fabsf(v.x - v_rec.x) < 1e-4f);
    assert(fabsf(v.y - v_rec.y) < 1e-4f);
    assert(fabsf(v.z - v_rec.z) < 1e-4f);

    printf("  PASS: test_log_exp_round_trip\n");
}

static void test_slerp_properties() {
    printf("Running test_slerp_properties...\n");
    geo_rotor3_t r0 = geo_rotor3_from_bivector(0.1f, 0.2f, 0.3f);
    geo_rotor3_t r1 = geo_rotor3_from_bivector(0.4f, -0.1f, 0.5f);

    // Endpoint t=0 -> r0
    geo_rotor3_t s0 = geo_rotor3_slerp(r0, r1, 0.0f);
    assert(fabsf(s0.s - r0.s) < 1e-4f);
    assert(fabsf(s0.b0 - r0.b0) < 1e-4f);

    // Endpoint t=1 -> r1 (up to antipodal sign)
    geo_rotor3_t s1 = geo_rotor3_slerp(r0, r1, 1.0f);
    float dot = fabsf(geo_rotor3_dot(s1, r1));
    assert(fabsf(dot - 1.0f) < 1e-4f);

    printf("  PASS: test_slerp_properties\n");
}

static void test_karcher_mean_edge_order_invariance() {
    printf("Running test_karcher_mean_edge_order_invariance...\n");
    geo_rotor3_t r1 = geo_rotor3_from_bivector(0.2f, 0.1f, 0.0f);
    geo_rotor3_t r2 = geo_rotor3_from_bivector(-0.3f, 0.4f, 0.1f);
    geo_rotor3_t r3 = geo_rotor3_from_bivector(0.1f, -0.2f, 0.5f);
    geo_rotor3_t init = {1.0f, 0.0f, 0.0f, 0.0f};

    geo_rotor3_t rotors_a[3] = {r1, r2, r3};
    float weights_a[3] = {0.33333333f, 0.33333333f, 0.33333333f};

    geo_rotor3_t rotors_b[3] = {r3, r1, r2};
    float weights_b[3] = {0.33333333f, 0.33333333f, 0.33333333f};

    geo_rotor3_t mean_a = geo_karcher_mean_spin3(rotors_a, weights_a, 3, init);
    geo_rotor3_t mean_b = geo_karcher_mean_spin3(rotors_b, weights_b, 3, init);

    float diff_s = fabsf(mean_a.s - mean_b.s);
    float diff_b0 = fabsf(mean_a.b0 - mean_b.b0);
    float diff_b1 = fabsf(mean_a.b1 - mean_b.b1);
    float diff_b2 = fabsf(mean_a.b2 - mean_b.b2);

    assert(diff_s < 1e-5f);
    assert(diff_b0 < 1e-5f);
    assert(diff_b1 < 1e-5f);
    assert(diff_b2 < 1e-5f);

    printf("  PASS: test_karcher_mean_edge_order_invariance\n");
}

int main(void) {
    printf("=== RUNNING CL30 KARCHER TESTS ===\n");
    test_rotor_from_bivector_and_multiply();
    test_log_exp_round_trip();
    test_slerp_properties();
    test_karcher_mean_edge_order_invariance();
    printf("ALL CL30 KARCHER TESTS PASSED!\n");
    return 0;
}
