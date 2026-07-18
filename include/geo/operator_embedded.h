#ifndef GEO_OPERATOR_EMBEDDED_H
#define GEO_OPERATOR_EMBEDDED_H

#include "geo/operator_kernel.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_OPERATOR_EMBEDDED_NO_HEAP 1
#define GEO_OPERATOR_EMBEDDED_NO_RUNTIME_PARSER 1
#define GEO_OPERATOR_EMBEDDED_MAX_PLAN_BYTES ((size_t)sizeof(geo_operator_plan_i32_t))
#define GEO_OPERATOR_EMBEDDED_MAX_I32_SCRATCH_BYTES ((size_t)sizeof(geo_operator_mv_i32_t))
#define GEO_OPERATOR_EMBEDDED_MAX_Q_I32_SCRATCH_BYTES ((size_t)sizeof(geo_operator_mv_i32_t))
#define GEO_OPERATOR_EMBEDDED_MAX_F64_SCRATCH_BYTES ((size_t)sizeof(geo_operator_mv_f64_t))

#if defined(__GNUC__) || defined(__clang__)
#define GEO_OPERATOR_FLASH __attribute__((section(".rodata")))
#else
#define GEO_OPERATOR_FLASH
#endif

typedef struct {
    uint8_t maximum_dimension;
    uint8_t maximum_blades;
    uint16_t maximum_terms;
    size_t maximum_plan_bytes;
    size_t i32_scratch_bytes;
    size_t q_i32_scratch_bytes;
    size_t f64_scratch_bytes;
    uint32_t maximum_contributions_per_output;
    bool heap_required;
    bool runtime_parser_required;
} geo_operator_embedded_limits_t;

static inline geo_operator_embedded_limits_t geo_operator_embedded_limits(void) {
    geo_operator_embedded_limits_t limits;
    limits.maximum_dimension = GEO_OPERATOR_MAX_DIMENSION;
    limits.maximum_blades = GEO_OPERATOR_MAX_BLADES;
    limits.maximum_terms = GEO_OPERATOR_MAX_TERMS;
    limits.maximum_plan_bytes = GEO_OPERATOR_EMBEDDED_MAX_PLAN_BYTES;
    limits.i32_scratch_bytes = GEO_OPERATOR_EMBEDDED_MAX_I32_SCRATCH_BYTES;
    limits.q_i32_scratch_bytes = GEO_OPERATOR_EMBEDDED_MAX_Q_I32_SCRATCH_BYTES;
    limits.f64_scratch_bytes = GEO_OPERATOR_EMBEDDED_MAX_F64_SCRATCH_BYTES;
    limits.maximum_contributions_per_output = GEO_OPERATOR_MAX_TERMS;
    limits.heap_required = false;
    limits.runtime_parser_required = false;
    return limits;
}

#ifdef __cplusplus
}
#endif

#endif
