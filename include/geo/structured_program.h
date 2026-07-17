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
    GEO_STRUCT_VALUE_HADAMARD_PAIR = 3,
    GEO_STRUCT_VALUE_SCALAR = 4,
    GEO_STRUCT_VALUE_SCALED_CL20 = 5
} geo_struct_value_kind_t;

typedef struct {
    uint8_t kind;
    union {
        geo_cl20_t cl20;
        geo_unipotent_t unipotent;
        geo_ordered_pair_t ordered;
        geo_hadamard_pair_t hadamard;
        geo_real_t scalar;
        geo_scaled_cl20_t scaled_cl20;
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
    GEO_STRUCT_OP_SELECT_ANTISYMMETRIC = 9,
    GEO_STRUCT_OP_CL20_SUBTRACT = 10,
    GEO_STRUCT_OP_CL20_SCALE = 11,
    GEO_STRUCT_OP_SCALAR_EXTRACT = 12,
    GEO_STRUCT_OP_SCALAR_MULTIPLY = 13,
    GEO_STRUCT_OP_VECTOR_NORM_SQUARED = 14,
    GEO_STRUCT_OP_DISTANCE_SQUARED = 15,
    GEO_STRUCT_OP_PROJECTION_NUMERATOR = 16,
    GEO_STRUCT_OP_REJECTION_NUMERATOR = 17,
    GEO_STRUCT_OP_REFLECTION_NUMERATOR = 18,
    GEO_STRUCT_OP_VECTOR_INVERSE_PROJECTIVE = 19,
    GEO_STRUCT_OP_ANGLE_COSINE_NUMERATOR = 20,
    GEO_STRUCT_OP_SCALED_CL20_NORMALIZE = 21,
    GEO_STRUCT_OP_CL20_REVERSE = 22,
    GEO_STRUCT_OP_CL20_GRADE_INVOLUTION = 23,
    GEO_STRUCT_OP_CL20_CLIFFORD_CONJUGATE = 24,
    GEO_STRUCT_OP_PROJECT_SCALAR = 25,
    GEO_STRUCT_OP_PROJECT_VECTOR = 26,
    GEO_STRUCT_OP_PROJECT_BIVECTOR = 27,
    GEO_STRUCT_OP_PROJECT_EVEN = 28,
    GEO_STRUCT_OP_PROJECT_ODD = 29,
    GEO_STRUCT_OP_DUAL = 30,
    GEO_STRUCT_OP_ROTOR_NORM_SQUARED = 31
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
geo_struct_value_t geo_struct_value_from_scalar(geo_real_t value);
geo_struct_value_t geo_struct_value_from_scaled_cl20(geo_scaled_cl20_t value);

geo_status_t geo_struct_program_execute_impl(
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
);

geo_status_t geo_struct_program_execute(
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
);

#if defined(GEO_STRUCTURED_PROGRAM_IMPLEMENTATION)
#define geo_struct_program_execute geo_struct_program_execute_impl
#endif

geo_status_t geo_struct_read_cl20(const geo_struct_value_t *value, geo_cl20_t *output);
geo_status_t geo_struct_read_scalar(const geo_struct_value_t *value, geo_real_t *output);
geo_status_t geo_struct_read_scaled_cl20(const geo_struct_value_t *value, geo_scaled_cl20_t *output);
geo_status_t geo_struct_read_unipotent(const geo_struct_value_t *value, geo_unipotent_t *output);

#ifdef __cplusplus
}
#endif

#endif
