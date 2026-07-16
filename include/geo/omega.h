#ifndef GEO_OMEGA_H
#define GEO_OMEGA_H

#include <stddef.h>
#include <stdint.h>

#include "geo/opposite.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_STATUS_OK = 0,
    GEO_STATUS_NULL_ARGUMENT = 1,
    GEO_STATUS_LOG_DOMAIN = 2,
    GEO_STATUS_SCALE_OVERFLOW = 3,
    GEO_STATUS_REGISTER_RANGE = 4,
    GEO_STATUS_BAD_OPCODE = 5,
    GEO_STATUS_ZERO_NORM = 6
} geo_status_t;

typedef enum {
    GEO_LANE_NONE = 0u,
    GEO_LANE_SCALAR = 1u << 0,
    GEO_LANE_GEOMETRIC = 1u << 1,
    GEO_LANE_ALL = GEO_LANE_SCALAR | GEO_LANE_GEOMETRIC
} geo_lane_mask_t;

typedef struct {
    int32_t numerator;
    int32_t denominator;
} geo_scale_t;

typedef struct {
    geo_real_t scalar;
    geo_opposite_t geometric;
    geo_scale_t scale;
    uint8_t active_lanes;
} geo_state_t;

typedef enum {
    GEO_OPCODE_COPY = 0,
    GEO_OPCODE_OMEGA = 1
} geo_opcode_t;

typedef struct {
    uint8_t opcode;
    uint8_t destination;
    uint8_t left;
    uint8_t right;
    uint8_t requested_lanes;
} geo_instruction_t;

typedef struct {
    const geo_instruction_t *instructions;
    size_t instruction_count;
    size_t register_count;
} geo_program_t;

geo_scale_t geo_scale_one(void);
geo_state_t geo_state_zero(void);
geo_state_t geo_state_from_scalar(geo_real_t scalar);
geo_state_t geo_state_from_cl20(geo_cl20_t value);

geo_status_t geo_omega_apply(
    const geo_state_t *left,
    const geo_state_t *right,
    uint8_t requested_lanes,
    geo_state_t *output
);

geo_status_t geo_program_execute(
    const geo_program_t *program,
    geo_state_t *registers,
    size_t register_capacity
);

#ifdef __cplusplus
}
#endif

#endif
