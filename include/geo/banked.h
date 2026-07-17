#ifndef GEO_BANKED_H
#define GEO_BANKED_H

#include <stddef.h>
#include <stdint.h>

#include "geo/folding.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    geo_opposite_t value;
    geo_scale_t scale;
} geo_geometric_register_t;

typedef struct {
    uint8_t kind;
    uint8_t index;
} geo_banked_ref_t;

typedef struct {
    uint8_t requested_lanes;
    geo_banked_ref_t destination;
    geo_banked_ref_t left;
    geo_banked_ref_t right;
} geo_banked_instruction_t;

typedef struct {
    geo_banked_instruction_t *instructions;
    size_t instruction_capacity;
    geo_banked_ref_t *logical_refs;
    size_t logical_ref_capacity;
} geo_banked_plan_workspace_t;

typedef struct {
    const geo_banked_instruction_t *instructions;
    size_t instruction_count;
    geo_banked_ref_t root;
    size_t scalar_count;
    size_t geometric_count;
    size_t unified_count;
    size_t required_bytes;
} geo_banked_program_t;

typedef struct {
    geo_real_t *scalars;
    size_t scalar_capacity;
    geo_geometric_register_t *geometrics;
    size_t geometric_capacity;
    geo_state_t *unified;
    size_t unified_capacity;
} geo_banked_storage_t;

geo_status_t geo_banked_plan(
    const geo_folded_program_t *folded,
    geo_banked_plan_workspace_t *workspace,
    geo_banked_program_t *output
);

geo_status_t geo_banked_initialize(
    const geo_folded_program_t *folded,
    const geo_banked_program_t *program,
    const geo_banked_ref_t *logical_refs,
    geo_banked_storage_t *storage
);

/*
 * Stable public executor symbol. The implementation entry point remains
 * available for source compatibility, but callers should use this function.
 */
geo_status_t geo_banked_execute(
    const geo_banked_program_t *program,
    geo_banked_storage_t *storage
);

geo_status_t geo_banked_execute_impl(
    const geo_banked_program_t *program,
    geo_banked_storage_t *storage
);

#if defined(GEO_BANKED_IMPLEMENTATION)
#define geo_banked_execute geo_banked_execute_impl
#endif

geo_status_t geo_banked_read_state(
    const geo_banked_storage_t *storage,
    geo_banked_ref_t reference,
    geo_state_t *output
);

#ifdef __cplusplus
}
#endif

#endif
