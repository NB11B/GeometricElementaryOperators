#ifndef GEO_OPTIMIZER_H
#define GEO_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>

#include "geo/compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    geo_instruction_t *instructions;
    size_t instruction_capacity;
    uint8_t *node_registers;
    size_t node_register_capacity;
    uint8_t *live_lanes;
    size_t live_lane_capacity;
    uint16_t *representatives;
    size_t representative_capacity;
} geo_optimizer_workspace_t;

typedef struct {
    geo_program_t program;
    uint8_t root_register;
    size_t terminal_count;
    size_t original_instruction_count;
    size_t optimized_instruction_count;
    size_t eliminated_dead_nodes;
    size_t eliminated_duplicate_nodes;
} geo_optimized_witness_t;

/*
 * Compiles only the subgraph reachable from the selected root, propagates
 * lane liveness backward, merges duplicate Omega nodes, and compacts result
 * registers. The authoritative terminal_count is preserved for later passes.
 */
geo_status_t geo_witness_compile_optimized(
    const geo_witness_tree_t *tree,
    geo_optimizer_workspace_t *workspace,
    geo_optimized_witness_t *output
);

#ifdef __cplusplus
}
#endif

#endif
