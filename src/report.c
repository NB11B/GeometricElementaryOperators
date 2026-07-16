#include "geo/report.h"

#include <stdio.h>
#include <string.h>

void geo_compiler_report_build(
    const geo_witness_tree_t *tree,
    const geo_optimized_witness_t *optimized,
    const geo_folded_program_t *folded,
    const geo_banked_program_t *banked,
    geo_compiler_report_t *report
) {
    if (report == NULL) return;
    memset(report, 0, sizeof(*report));
    if (tree != NULL) report->original_nodes = tree->node_count;
    if (optimized != NULL) {
        report->original_instructions = optimized->original_instruction_count;
        report->optimized_instructions = optimized->optimized_instruction_count;
        report->dead_nodes = optimized->eliminated_dead_nodes;
        report->duplicate_nodes = optimized->eliminated_duplicate_nodes;
    }
    if (folded != NULL) {
        report->folded_instructions = folded->folded_instruction_count;
        report->folded_constants = folded->folded_constant_nodes;
        report->scalar_registers = folded->scalar_register_count;
        report->geometric_registers = folded->geometric_register_count;
        report->unified_registers = folded->unified_register_count;
    }
    if (banked != NULL) report->runtime_bytes = banked->required_bytes;
}

int geo_compiler_report_json(
    const geo_compiler_report_t *report,
    char *buffer,
    size_t capacity
) {
    if (report == NULL || buffer == NULL || capacity == 0u) return -1;
    return snprintf(
        buffer,
        capacity,
        "{\"original_nodes\":%zu,\"original_instructions\":%zu,"
        "\"optimized_instructions\":%zu,\"folded_instructions\":%zu,"
        "\"folded_constants\":%zu,\"dead_nodes\":%zu,"
        "\"duplicate_nodes\":%zu,\"scalar_registers\":%zu,"
        "\"geometric_registers\":%zu,\"unified_registers\":%zu,"
        "\"runtime_bytes\":%zu}",
        report->original_nodes,
        report->original_instructions,
        report->optimized_instructions,
        report->folded_instructions,
        report->folded_constants,
        report->dead_nodes,
        report->duplicate_nodes,
        report->scalar_registers,
        report->geometric_registers,
        report->unified_registers,
        report->runtime_bytes
    );
}

static const char *kind_name(uint8_t kind) {
    switch ((geo_register_kind_t)kind) {
        case GEO_REGISTER_SCALAR: return "GEO_REGISTER_SCALAR";
        case GEO_REGISTER_GEOMETRIC: return "GEO_REGISTER_GEOMETRIC";
        case GEO_REGISTER_UNIFIED: return "GEO_REGISTER_UNIFIED";
        default: return "GEO_REGISTER_UNUSED";
    }
}

int geo_emit_banked_program_c(
    const geo_banked_program_t *program,
    const char *symbol_prefix,
    char *buffer,
    size_t capacity
) {
    size_t used = 0u;
    size_t i;
    int written;

    if (program == NULL || symbol_prefix == NULL || buffer == NULL || capacity == 0u) return -1;
    if (program->instruction_count != 0u && program->instructions == NULL) return -1;
    if (program->root.kind > GEO_REGISTER_UNIFIED) return -1;

#define APPEND(...) \
    do { \
        if (used >= capacity) return -1; \
        written = snprintf(buffer + used, capacity - used, __VA_ARGS__); \
        if (written < 0 || (size_t)written >= capacity - used) return -1; \
        used += (size_t)written; \
    } while (0)

    if (program->instruction_count == 0u) {
        APPEND("static const geo_banked_instruction_t *const %s_instructions = NULL;\n", symbol_prefix);
    } else {
        APPEND("static const geo_banked_instruction_t %s_instructions[%zu] = {\n", symbol_prefix, program->instruction_count);
        for (i = 0u; i < program->instruction_count; ++i) {
            const geo_banked_instruction_t ins = program->instructions[i];
            if (ins.destination.kind > GEO_REGISTER_UNIFIED ||
                ins.left.kind > GEO_REGISTER_UNIFIED ||
                ins.right.kind > GEO_REGISTER_UNIFIED) {
                return -1;
            }
            APPEND(
                "  {%u,{%s,%u},{%s,%u},{%s,%u}},\n",
                (unsigned)ins.requested_lanes,
                kind_name(ins.destination.kind), (unsigned)ins.destination.index,
                kind_name(ins.left.kind), (unsigned)ins.left.index,
                kind_name(ins.right.kind), (unsigned)ins.right.index
            );
        }
        APPEND("};\n");
    }

    APPEND(
        "static const geo_banked_program_t %s_program = {%s_instructions,%zu,{%s,%u},%zu,%zu,%zu,%zu};\n",
        symbol_prefix,
        symbol_prefix,
        program->instruction_count,
        kind_name(program->root.kind),
        (unsigned)program->root.index,
        program->scalar_count,
        program->geometric_count,
        program->unified_count,
        program->required_bytes
    );

#undef APPEND
    return (int)used;
}
