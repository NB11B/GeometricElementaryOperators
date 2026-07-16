#ifndef GEO_REPORT_H
#define GEO_REPORT_H

#include <stddef.h>

#include "geo/banked.h"
#include "geo/lowering.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t original_nodes;
    size_t original_instructions;
    size_t optimized_instructions;
    size_t folded_instructions;
    size_t folded_constants;
    size_t dead_nodes;
    size_t duplicate_nodes;
    size_t scalar_registers;
    size_t geometric_registers;
    size_t unified_registers;
    size_t runtime_bytes;
} geo_compiler_report_t;

void geo_compiler_report_build(
    const geo_witness_tree_t *tree,
    const geo_optimized_witness_t *optimized,
    const geo_folded_program_t *folded,
    const geo_banked_program_t *banked,
    geo_compiler_report_t *report
);

int geo_compiler_report_json(
    const geo_compiler_report_t *report,
    char *buffer,
    size_t capacity
);

int geo_emit_banked_program_c(
    const geo_banked_program_t *program,
    const char *symbol_prefix,
    char *buffer,
    size_t capacity
);

#ifdef __cplusplus
}
#endif

#endif
