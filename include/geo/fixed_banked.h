#ifndef GEO_FIXED_BANKED_H
#define GEO_FIXED_BANKED_H

#include <stddef.h>
#include <stdint.h>

#include "geo/fixed_omega.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_FIXED_BANKED_OK = 0,
    GEO_FIXED_BANKED_OVERFLOW = 1,
    GEO_FIXED_BANKED_DIVIDE_BY_ZERO = 2,
    GEO_FIXED_BANKED_LOG_DOMAIN = 3,
    GEO_FIXED_BANKED_NULL_ARGUMENT = 4,
    GEO_FIXED_BANKED_STORAGE_RANGE = 5,
    GEO_FIXED_BANKED_TYPE_MISMATCH = 6,
    GEO_FIXED_BANKED_BAD_LANES = 7,
    GEO_FIXED_BANKED_INVALID_SCALE = 8,
    GEO_FIXED_BANKED_BAD_OPCODE = 9
} geo_fixed_banked_status_t;

typedef enum {
    GEO_FIXED_BANK_SCALAR = 0,
    GEO_FIXED_BANK_GEOMETRIC = 1,
    GEO_FIXED_BANK_UNIFIED = 2
} geo_fixed_bank_kind_t;

typedef struct {
    geo_fixed_opposite_t value;
    geo_fixed_scale_t scale;
} geo_fixed_geometric_register_t;

typedef struct {
    uint8_t kind;
    uint8_t index;
} geo_fixed_banked_ref_t;

typedef struct {
    uint8_t requested_lanes;
    geo_fixed_banked_ref_t destination;
    geo_fixed_banked_ref_t left;
    geo_fixed_banked_ref_t right;
} geo_fixed_banked_instruction_t;

typedef struct {
    const geo_fixed_banked_instruction_t *instructions;
    size_t instruction_count;
    geo_fixed_banked_ref_t root;
    size_t scalar_count;
    size_t geometric_count;
    size_t unified_count;
} geo_fixed_banked_program_t;

typedef struct {
    geo_fixed_t *scalars;
    size_t scalar_capacity;
    geo_fixed_geometric_register_t *geometrics;
    size_t geometric_capacity;
    geo_fixed_state_t *unified;
    size_t unified_capacity;
} geo_fixed_banked_storage_t;

geo_fixed_banked_status_t geo_fixed_banked_read_state(
    const geo_fixed_banked_storage_t *storage,
    geo_fixed_banked_ref_t reference,
    geo_fixed_state_t *output
);

geo_fixed_banked_status_t geo_fixed_banked_write_state(
    geo_fixed_banked_storage_t *storage,
    geo_fixed_banked_ref_t reference,
    const geo_fixed_state_t *value
);

/*
 * Executes Omega instructions directly over physically separate scalar,
 * geometric, and unified banks. A failed instruction does not modify its
 * destination register.
 */
geo_fixed_banked_status_t geo_fixed_banked_execute(
    const geo_fixed_banked_program_t *program,
    geo_fixed_banked_storage_t *storage
);

#ifdef __cplusplus
}
#endif

#endif
