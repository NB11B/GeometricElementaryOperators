#ifndef GEO_FOLDING_H
#define GEO_FOLDING_H

#include <stddef.h>
#include <stdint.h>

#include "geo/optimizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_REGISTER_UNUSED = 0,
    GEO_REGISTER_SCALAR = 1,
    GEO_REGISTER_GEOMETRIC = 2,
    GEO_REGISTER_UNIFIED = 3
} geo_register_kind_t;

typedef struct {
    geo_instruction_t *instructions;
    size_t instruction_capacity;

    geo_state_t *initial_registers;
    size_t initial_register_capacity;

    uint8_t *old_to_new;
    size_t old_to_new_capacity;

    uint8_t *constant_flags;
    size_t constant_flag_capacity;

    uint8_t *register_kinds;
    size_t register_kind_capacity;
} geo_folding_workspace_t;

typedef struct {
    geo_program_t program;
    const geo_state_t *initial_registers;
    uint8_t root_register;

    size_t original_instruction_count;
    size_t folded_instruction_count;
    size_t folded_constant_nodes;

    size_t scalar_register_count;
    size_t geometric_register_count;
    size_t unified_register_count;
    size_t estimated_typed_bytes;
    size_t estimated_unified_bytes;
} geo_folded_program_t;

/*
 * Folds instructions whose two inputs are compile-time constants. Terminal
 * constants are selected by terminal_constant_flags. The resulting register
 * image is caller-owned and can be copied directly into the runtime bank.
 *
 * This pass also classifies each live register as scalar-only, geometric-only,
 * or unified and reports the memory required by a future banked allocator.
 */
geo_status_t geo_program_fold_constants(
    const geo_optimized_witness_t *input,
    const geo_state_t *terminal_values,
    const uint8_t *terminal_constant_flags,
    size_t terminal_count,
    geo_folding_workspace_t *workspace,
    geo_folded_program_t *output
);

#ifdef __cplusplus
}
#endif

#endif
