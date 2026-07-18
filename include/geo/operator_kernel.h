#ifndef GEO_OPERATOR_KERNEL_H
#define GEO_OPERATOR_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_OPERATOR_MAX_DIMENSION 6u
#define GEO_OPERATOR_MAX_BLADES 64u
#define GEO_OPERATOR_MAX_TERMS 64u
#define GEO_OPERATOR_ABI_VERSION 0x00050100u

typedef enum {
    GEO_OPERATOR_OK = 0,
    GEO_OPERATOR_INVALID_ARGUMENT = 1,
    GEO_OPERATOR_UNSUPPORTED_DIMENSION = 2,
    GEO_OPERATOR_TOO_MANY_TERMS = 3,
    GEO_OPERATOR_ZERO_MODULUS = 4,
    GEO_OPERATOR_OVERFLOW = 5,
    GEO_OPERATOR_BAD_PLAN = 6
} geo_operator_status_t;

typedef enum {
    GEO_OPERATOR_SIDE_RIGHT = 0,
    GEO_OPERATOR_SIDE_LEFT = 1
} geo_operator_side_t;

typedef enum {
    GEO_OPERATOR_PLAN_FIXED_BLADE = 1,
    GEO_OPERATOR_PLAN_SPARSE_FIXED_MULTIVECTOR = 2
} geo_operator_plan_kind_t;

typedef struct {
    uint8_t blade;
    int32_t coefficient;
} geo_operator_term_i32_t;

typedef struct {
    uint32_t abi_version;
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    uint8_t kind;
    uint8_t side;
    uint8_t term_count;
    uint8_t reserved;
    geo_operator_term_i32_t terms[GEO_OPERATOR_MAX_TERMS];
} geo_operator_plan_i32_t;

typedef struct {
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    double coefficients[GEO_OPERATOR_MAX_BLADES];
} geo_operator_mv_f64_t;

typedef struct {
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    int32_t coefficients[GEO_OPERATOR_MAX_BLADES];
} geo_operator_mv_i32_t;

uint32_t geo_operator_abi_version(void);
size_t geo_operator_blade_count(uint8_t dimension);
int geo_operator_gp_sign(uint8_t left_blade, uint8_t right_blade, const int8_t *signature, uint8_t dimension);

geo_operator_status_t geo_operator_plan_fixed_blade_i32(
    geo_operator_plan_i32_t *plan,
    uint8_t dimension,
    const int8_t *signature,
    geo_operator_side_t side,
    uint8_t blade,
    int32_t coefficient
);

geo_operator_status_t geo_operator_plan_sparse_i32(
    geo_operator_plan_i32_t *plan,
    uint8_t dimension,
    const int8_t *signature,
    geo_operator_side_t side,
    const geo_operator_term_i32_t *terms,
    size_t term_count
);

geo_operator_status_t geo_operator_apply_f64(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_f64_t *input,
    geo_operator_mv_f64_t *output
);

geo_operator_status_t geo_operator_apply_mod_i32(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_i32_t *input,
    uint32_t modulus,
    geo_operator_mv_i32_t *output
);

geo_operator_status_t geo_operator_apply_q_i32(
    const geo_operator_plan_i32_t *plan,
    const geo_operator_mv_i32_t *input,
    uint8_t fraction_bits,
    geo_operator_mv_i32_t *output
);

geo_operator_status_t geo_operator_gp_f64(
    const geo_operator_mv_f64_t *left,
    const geo_operator_mv_f64_t *right,
    geo_operator_mv_f64_t *output
);

#ifdef __cplusplus
}
#endif

#endif
