#ifndef GEO_COMPILER_H
#define GEO_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#include "geo/omega.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_WITNESS_TERMINAL = 0,
    GEO_WITNESS_OMEGA = 1
} geo_witness_kind_t;

typedef struct {
    uint8_t kind;
    uint8_t requested_lanes;
    uint16_t left;
    uint16_t right;
    uint16_t terminal_index;
} geo_witness_node_t;

typedef struct {
    const geo_witness_node_t *nodes;
    size_t node_count;
    size_t terminal_count;
    uint16_t root;
} geo_witness_tree_t;

typedef struct {
    geo_instruction_t *instructions;
    size_t instruction_capacity;
    uint8_t *node_registers;
    size_t node_register_capacity;
} geo_compile_workspace_t;

typedef struct {
    geo_program_t program;
    uint8_t root_register;
} geo_compiled_witness_t;

/*
 * Witness nodes must be stored in topological order. Every nonterminal node may
 * reference only earlier nodes. Terminals map directly to preloaded registers
 * [0, terminal_count). Omega results are allocated consecutively afterward.
 */
geo_status_t geo_witness_validate(const geo_witness_tree_t *tree);

geo_status_t geo_witness_compile(
    const geo_witness_tree_t *tree,
    geo_compile_workspace_t *workspace,
    geo_compiled_witness_t *output
);

#ifdef __cplusplus
}
#endif

#endif
