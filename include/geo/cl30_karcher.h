/**
 * @file cl30_karcher.h
 * @brief Clifford Algebra Cl(3,0) / Spin(3) Rotor and Riemannian Karcher Mean Engine.
 *
 * Implements:
 * - Rotor representation in Spin(3): R = s + B_12 e_12 + B_23 e_23 + B_31 e_31
 * - Rotor multiplication, inverse, and conjugation
 * - Spin(3) Logarithm and Exponential maps
 * - Spherical Linear Interpolation (SLERP) on Spin(3) and S^2
 * - Fixed 4-iteration Riemannian Karcher mean on Spin(3) with shortest-arc antipodal alignment
 */

#ifndef GEO_CL30_KARCHER_H
#define GEO_CL30_KARCHER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 4-component Spin(3) Rotor: R = s + b0*e12 + b1*e23 + b2*e31
 * Satisfies s^2 + b0^2 + b1^2 + b2^2 = 1.
 */
typedef struct {
    float s;    /**< Scalar component (grade 0) */
    float b0;   /**< e12 bivector component */
    float b1;   /**< e23 bivector component */
    float b2;   /**< e31 bivector component */
} geo_rotor3_t;

/**
 * @brief 3D vector (grade 1)
 */
typedef struct {
    float x;
    float y;
    float z;
} geo_vec3_t;

/**
 * @brief Construct rotor from bivector coordinate B = (b0, b1, b2).
 * R = exp(-0.5 * B) = cos(||B||/2) - (B / ||B||) * sin(||B||/2)
 */
geo_rotor3_t geo_rotor3_from_bivector(float b0, float b1, float b2);

/**
 * @brief Rotor product: R = R1 * R2 in Cl(3,0) / Spin(3).
 */
geo_rotor3_t geo_rotor3_multiply(geo_rotor3_t r1, geo_rotor3_t r2);

/**
 * @brief Rotor reverse / conjugate / inverse: R~ = s - b0*e12 - b1*e23 - b2*e31.
 */
geo_rotor3_t geo_rotor3_reverse(geo_rotor3_t r);

/**
 * @brief Normalize rotor to unit norm: ||R|| = 1.
 */
geo_rotor3_t geo_rotor3_normalize(geo_rotor3_t r);

/**
 * @brief Rotor dot product <R1, R2> = s1*s2 + b0_1*b0_2 + b1_1*b1_2 + b2_1*b2_2.
 */
float geo_rotor3_dot(geo_rotor3_t r1, geo_rotor3_t r2);

/**
 * @brief Spin(3) Logarithm map: Log(R) returns 3D tangent vector (bivector).
 * For R = cos(theta) + u * sin(theta), Log(R) = theta * u.
 */
geo_vec3_t geo_rotor3_log(geo_rotor3_t r);

/**
 * @brief Spin(3) Exponential map: Exp(v) for tangent vector v = theta * u.
 * Exp(v) = cos(theta) + u * sin(theta).
 */
geo_rotor3_t geo_rotor3_exp(geo_vec3_t v);

/**
 * @brief Spherical linear interpolation (SLERP) on S^2 for unit 3D vectors.
 */
geo_vec3_t geo_vec3_slerp(geo_vec3_t v0, geo_vec3_t v1, float t);

/**
 * @brief Spherical linear interpolation (SLERP) on Spin(3) (unit quaternions).
 */
geo_rotor3_t geo_rotor3_slerp(geo_rotor3_t r0, geo_rotor3_t r1, float t);

/**
 * @brief Rotate 3D vector by rotor: v' = R * v * R~
 */
geo_vec3_t geo_rotor3_rotate_vec(geo_rotor3_t r, geo_vec3_t v);

/**
 * @brief Exact 4-iteration Riemannian Karcher mean on Spin(3).
 *
 * Given N candidate rotors G_e and non-negative normalized weights w_e (sum w_e = 1):
 * 1. G^[0] = G_init
 * 2. For k = 0, 1, 2, 3:
 *    Align G_e: G_e_aligned = (<G^[k], G_e> >= 0) ? G_e : -G_e
 *    Delta^[k] = sum_e w_e * Log( (G^[k])~ * G_e_aligned )
 *    G^[k+1] = G^[k] * Exp(Delta^[k])
 * 3. Return normalized G^[4].
 *
 * @param rotors Array of N candidate rotors.
 * @param weights Array of N non-negative normalized weights (sum = 1.0).
 * @param count Number of candidate rotors N.
 * @param init_rotor Initial rotor G_init.
 * @return geo_rotor3_t Computed Karcher mean on Spin(3).
 */
geo_rotor3_t geo_karcher_mean_spin3(
    const geo_rotor3_t* rotors,
    const float* weights,
    size_t count,
    geo_rotor3_t init_rotor
);

#ifdef __cplusplus
}
#endif

#endif /* GEO_CL30_KARCHER_H */
