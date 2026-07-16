#ifndef GEO_FIXED_PROGRAM_H
#define GEO_FIXED_PROGRAM_H

#include <stddef.h>
#include <stdint.h>

#include "geo/fixed_geb36.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_FIXED_PROGRAM_OK = 0,
    GEO_FIXED_PROGRAM_OVERFLOW = 1,
    GEO_FIXED_PROGRAM_DIVIDE_BY_ZERO = 2,
    GEO_FIXED_PROGRAM_NULL_ARGUMENT = 3,
    GEO_FIXED_PROGRAM_REGISTER_RANGE = 4,
    GEO_FIXED_PROGRAM_TYPE_MISMATCH = 5,
    GEO_FIXED_PROGRAM_BAD_TARGET = 6,
    GEO_FIXED_PROGRAM_INVALID_SCALE = 7
} geo_fixed_program_status_t;

typedef struct {
    uint8_t target_id;
    uint8_t destination;
    uint8_t left;
    uint8_t right;
    uint8_t transform;
} geo_fixed_program_instruction_t;

typedef struct {
    const geo_fixed_program_instruction_t *instructions;
    size_t instruction_count;
    size_t register_count;
    uint8_t root_register;
} geo_fixed_program_t;

geo_fixed_geb_result_t geo_fixed_program_value_from_cl20(geo_fixed_cl20_t value);
geo_fixed_geb_result_t geo_fixed_program_value_from_scalar(geo_fixed_t value);

/*
 * Executes a flat, nonrecursive program over the complete frozen GEB-36
 * target set. Every arithmetic failure is returned before the destination
 * register is modified.
 */
geo_fixed_program_status_t geo_fixed_program_execute(
    const geo_fixed_program_t *program,
    geo_fixed_geb_result_t *registers,
    size_t register_capacity
);

geo_fixed_program_status_t geo_fixed_program_read_cl20(
    const geo_fixed_geb_result_t *value,
    geo_fixed_cl20_t *output
);

#ifdef __cplusplus
}
#endif

#endif
