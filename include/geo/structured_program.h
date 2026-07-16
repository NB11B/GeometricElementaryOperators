#ifndef GEO_STRUCTURED_PROGRAM_H
#define GEO_STRUCTURED_PROGRAM_H

#include <stddef.h>
#include <stdint.h>

#include "geo/structured.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_STRUCT_VALUE_CL20 = 0,
    GEO_STRUCT_VALUE_UNIPOTENT = 1,
    GEO_STRUCT_VALUE_ORDERED_PAIR = 2,
    GEO_STRUCT_VALUE_HADAMARD_PAIR = 3
} geo_struct_value_kind_t;

typedef struct {
    uint8_t kind;
    union {
        geo_cl20_t cl20;
        geo_unipotent_t unipotent;
        geo_ordered_pair_t ordered;
        geo_hadamard_pair_t hadamard;
    } as;
} geo_struct_value_t;

typedef enum {
    GEO_STRUCT_OP_COPY = 0,
    GEO_STRUCT_OP_CL20_NEGATE = 1,
    GEO_STRUCT_OP_UNIPOTENT_ENCODE = 2,
    GEO_STRUCT_OP_UNIPOTENT_COMPOSE = 3,
    GEO_STRUCT_OP_UNIPOTENT_EXTRACT = 4,
    GEO_STRUCT_OP_ORDERED_PRODUCTS = 5,
    GEO_STRUCT_OP_HADAMARD_EXACT = 6,
    GEO_STRUCT_OP_HADAMARD_PROJECTIVE = 7,
    GEO_STRUCT_OP_SELECT_SYMMETRIC = 8,
    GEO_STRUCT_OP_SELECT_ANTISYMMETRIC = 9
} geo_struct_opcode_t;

typedef struct {
    uint8_t opcode;
    uint8_t destination;
    uint8_t left;
    uint8_t right;
} geo_struct_instruction_t;

typedef struct {
    const geo_struct_instruction_t *instructions;
    size_t instruction_count;
    size_t register_count;
    uint8_t root_register;
} geo_struct_program_t;

geo_struct_value_t geo_struct_value_from_cl20(geo_cl20_t value);

geo_status_t geo_struct_program_execute(
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
);

geo_status_t geo_struct_read_cl20(
    const geo_struct_value_t *value,
    geo_cl20_t *output
);

#ifdef __cplusplus
}
#endif

#endif
