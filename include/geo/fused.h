#ifndef GEO_FUSED_H
#define GEO_FUSED_H

#include <stddef.h>
#include <stdint.h>

#include "geo/structured_program.h"
#include "geo/geb36.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_FUSED_CL20_ADD = 0,
    GEO_FUSED_CL20_SUBTRACT = 1,
    GEO_FUSED_GEOMETRIC_PRODUCT = 2,
    GEO_FUSED_VECTOR_DOT = 3,
    GEO_FUSED_VECTOR_WEDGE = 4,
    GEO_FUSED_COMMUTATOR = 5,
    GEO_FUSED_ANTICOMMUTATOR = 6,
    GEO_FUSED_ROTOR_ACTION = 7,
    GEO_FUSED_PROJECTION_NUMERATOR = 8
} geo_fused_opcode_t;

typedef struct {
    uint8_t opcode;
    uint8_t destination;
    uint8_t left;
    uint8_t right;
    uint8_t auxiliary;
} geo_fused_instruction_t;

typedef struct {
    const geo_fused_instruction_t *instructions;
    size_t instruction_count;
    size_t register_count;
    uint8_t root_register;
} geo_fused_program_t;

geo_status_t geo_fused_execute(
    const geo_fused_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
);

/* Emits a one-instruction specialized program for supported GEB targets. */
geo_status_t geo_fused_program_for_target(
    uint8_t target_id,
    geo_fused_instruction_t *instruction,
    geo_fused_program_t *program
);

#ifdef __cplusplus
}
#endif

#endif
