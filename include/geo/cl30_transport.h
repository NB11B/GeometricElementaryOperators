#ifndef GEO_CL30_TRANSPORT_H
#define GEO_CL30_TRANSPORT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
typedef double geo_real_t;
#else
typedef float geo_real_t;
#endif

// 4-component pure-even unit rotor in Cl+(3,0): (scalar, e12, e23, e31)
typedef struct {
    geo_real_t s;    // scalar component
    geo_real_t b12;  // e12 bivector component
    geo_real_t b23;  // e23 bivector component
    geo_real_t b31;  // e31 bivector component
} geo_cl30_rotor_t;

// 3-component spatial vector in Cl1(3,0): (e1, e2, e3)
typedef struct {
    geo_real_t x;  // e1 component
    geo_real_t y;  // e2 component
    geo_real_t z;  // e3 component
} geo_cl30_vec3_t;

// 3x3 orthogonal matrix in SO(3)
typedef struct {
    geo_real_t m[3][3];
} geo_so3_matrix_t;

// Rotor construction from unconstrained 3D bivector generator
geo_cl30_rotor_t geo_cl30_rotor_exp(geo_real_t b12, geo_real_t b23, geo_real_t b31);

// Certified normalization and reversal
geo_cl30_rotor_t geo_cl30_rotor_normalize(geo_cl30_rotor_t r);
geo_cl30_rotor_t geo_cl30_rotor_reverse(geo_cl30_rotor_t r);

// Noncommutative rotor-rotor composition: R_out = R1 * R2
geo_cl30_rotor_t geo_cl30_rotor_mul(geo_cl30_rotor_t r1, geo_cl30_rotor_t r2);

// Two-sided rotor sandwich vector rotation: v_out = R * v * R~
geo_cl30_vec3_t geo_cl30_rotor_apply_vec3(geo_cl30_rotor_t r, geo_cl30_vec3_t v);

// Convert rotor to equivalent 3x3 SO(3) matrix (for C4 equivalence verification)
geo_so3_matrix_t geo_cl30_rotor_to_so3(geo_cl30_rotor_t r);

// Apply SO(3) matrix multiplication: v_out = M * v
geo_cl30_vec3_t geo_so3_apply_vec3(geo_so3_matrix_t m, geo_cl30_vec3_t v);

// Compute normalization residual: |s^2 + b12^2 + b23^2 + b31^2 - 1.0|
geo_real_t geo_cl30_rotor_norm_residual(geo_cl30_rotor_t r);

// Array-level batch ABI functions
void cl30_rotor_normalize_batch(const float* bivectors, float* rotors, int64_t count);
void cl30_rotor_compose_batch(const float* r1, const float* r2, float* out, int64_t count);
void cl30_rotor_reverse_batch(const float* r, float* out, int64_t count);
void cl30_rotor_apply_batch(const float* rotors, const float* vectors, float* out, int64_t count);
void cl30_so3_matrix_apply_batch(const float* matrices, const float* vectors, float* out, int64_t count);

#ifdef __cplusplus
}
#endif

#endif // GEO_CL30_TRANSPORT_H
