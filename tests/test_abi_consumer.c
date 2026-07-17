#include "geo/banked.h"
#include "geo/optimizer.h"
#include "geo/structured_program.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    geo_program_t program;
    uint8_t root_register;
    size_t original_instruction_count;
    size_t optimized_instruction_count;
    size_t eliminated_dead_nodes;
    size_t eliminated_duplicate_nodes;
} geo_optimized_witness_legacy_layout_t;

_Static_assert(
    sizeof(geo_optimized_witness_t) == sizeof(geo_optimized_witness_legacy_layout_t),
    "geo_optimized_witness_t must retain its legacy caller-owned size"
);
_Static_assert(
    offsetof(geo_optimized_witness_t, program) ==
        offsetof(geo_optimized_witness_legacy_layout_t, program),
    "program offset changed"
);
_Static_assert(
    offsetof(geo_optimized_witness_t, root_register) ==
        offsetof(geo_optimized_witness_legacy_layout_t, root_register),
    "root_register offset changed"
);
_Static_assert(
    offsetof(geo_optimized_witness_t, original_instruction_count) ==
        offsetof(geo_optimized_witness_legacy_layout_t, original_instruction_count),
    "original_instruction_count offset changed"
);
_Static_assert(
    offsetof(geo_optimized_witness_t, optimized_instruction_count) ==
        offsetof(geo_optimized_witness_legacy_layout_t, optimized_instruction_count),
    "optimized_instruction_count offset changed"
);
_Static_assert(
    offsetof(geo_optimized_witness_t, eliminated_dead_nodes) ==
        offsetof(geo_optimized_witness_legacy_layout_t, eliminated_dead_nodes),
    "eliminated_dead_nodes offset changed"
);
_Static_assert(
    offsetof(geo_optimized_witness_t, eliminated_duplicate_nodes) ==
        offsetof(geo_optimized_witness_legacy_layout_t, eliminated_duplicate_nodes),
    "eliminated_duplicate_nodes offset changed"
);

int main(void) {
    if (geo_struct_program_execute(NULL, NULL, 0u) != GEO_STATUS_NULL_ARGUMENT) {
        fprintf(stderr, "public structured executor did not link or reject null arguments\n");
        return EXIT_FAILURE;
    }
    if (geo_struct_program_execute_impl(NULL, NULL, 0u) != GEO_STATUS_NULL_ARGUMENT) {
        fprintf(stderr, "structured implementation executor did not link\n");
        return EXIT_FAILURE;
    }
    if (geo_banked_execute(NULL, NULL) != GEO_STATUS_NULL_ARGUMENT) {
        fprintf(stderr, "public banked executor did not link or reject null arguments\n");
        return EXIT_FAILURE;
    }
    if (geo_banked_execute_impl(NULL, NULL) != GEO_STATUS_NULL_ARGUMENT) {
        fprintf(stderr, "banked implementation executor did not link\n");
        return EXIT_FAILURE;
    }

    puts("ABI consumer linked against public and implementation executor symbols.");
    return EXIT_SUCCESS;
}
