#ifndef GEO_FIXED_OMEGA_H
#define GEO_FIXED_OMEGA_H

#include <stddef.h>
#include <stdint.h>

#include "geo/fixed.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_FIXED_OMEGA_OK = 0,
    GEO_FIXED_OMEGA_OVERFLOW = 1,
    GEO_FIXED_OMEGA_DIVIDE_BY_ZERO = 2,
    GEO_FIXED_OMEGA_LOG_DOMAIN = 3,
    GEO_FIXED_OMEGA_NULL_ARGUMENT = 4,
    GEO_FIXED_OMEGA_REGISTER_RANGE = 5,
    GEO_FIXED_OMEGA_BAD_OPCODE = 6,
    GEO_FIXED_OMEGA_BAD_LANES = 7,
    GEO_FIXED_OMEGA_INVALID_SCALE = 8
} geo_fixed_omega_status_t;

typedef enum {
    GEO_FIXED_LANE_NONE = 0u,
    GEO_FIXED_LANE_SCALAR = 1u << 0,
    GEO_FIXED_LANE_GEOMETRIC = 1u << 1,
    GEO_FIXED_LANE_ALL = GEO_FIXED_LANE_SCALAR | GEO_FIXED_LANE_GEOMETRIC
} geo_fixed_lane_mask_t;

typedef struct {
    geo_fixed_cl20_t forward;
    geo_fixed_cl20_t reverse;
} geo_fixed_opposite_t;

typedef struct {
    int32_t numerator;
    int32_t denominator;
} geo_fixed_scale_t;

typedef struct {
    geo_fixed_t scalar;
    geo_fixed_opposite_t geometric;
    geo_fixed_scale_t scale;
    uint8_t active_lanes;
} geo_fixed_state_t;

typedef enum {
    GEO_FIXED_OMEGA_OPCODE_COPY = 0,
    GEO_FIXED_OMEGA_OPCODE_APPLY = 1
} geo_fixed_omega_opcode_t;

typedef struct {
    uint8_t opcode;
    uint8_t destination;
    uint8_t left;
    uint8_t right;
    uint8_t requested_lanes;
} geo_fixed_omega_instruction_t;

typedef struct {
    const geo_fixed_omega_instruction_t *instructions;
    size_t instruction_count;
    size_t register_count;
} geo_fixed_omega_program_t;

geo_fixed_omega_status_t geo_fixed_eml_exp(
    geo_fixed_t input,
    geo_fixed_t *output
);

geo_fixed_omega_status_t geo_fixed_eml_log(
    geo_fixed_t input,
    geo_fixed_t *output
);

geo_fixed_omega_status_t geo_fixed_eml_apply(
    geo_fixed_t left,
    geo_fixed_t right,
    geo_fixed_t *output
);

geo_fixed_omega_status_t geo_fixed_opposite_from_cl20(
    geo_fixed_cl20_t value,
    geo_fixed_opposite_t *output
);

geo_fixed_omega_status_t geo_fixed_opposite_mul(
    geo_fixed_opposite_t left,
    geo_fixed_opposite_t right,
    geo_fixed_opposite_t *output
);

geo_fixed_state_t geo_fixed_state_zero(void);
geo_fixed_state_t geo_fixed_state_from_scalar(geo_fixed_t scalar);
geo_fixed_omega_status_t geo_fixed_state_from_cl20(
    geo_fixed_cl20_t value,
    geo_fixed_state_t *output
);

geo_fixed_omega_status_t geo_fixed_omega_apply(
    const geo_fixed_state_t *left,
    const geo_fixed_state_t *right,
    uint8_t requested_lanes,
    geo_fixed_state_t *output
);

geo_fixed_omega_status_t geo_fixed_omega_program_execute(
    const geo_fixed_omega_program_t *program,
    geo_fixed_state_t *registers,
    size_t register_capacity
);

#ifdef __cplusplus
}
#endif

#endif
