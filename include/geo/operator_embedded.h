#ifndef GEO_OPERATOR_EMBEDDED_H
#define GEO_OPERATOR_EMBEDDED_H

#include "geo/operator_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_OPERATOR_EMBEDDED_NO_HEAP 1
#define GEO_OPERATOR_EMBEDDED_NO_RUNTIME_PARSER 1
#define GEO_OPERATOR_EMBEDDED_MAX_I32_SCRATCH_BYTES ((size_t)sizeof(geo_operator_mv_i32_t))
#define GEO_OPERATOR_EMBEDDED_MAX_FIXED_SCRATCH_BYTES ((size_t)sizeof(geo_operator_mv_fixed_t))
#define GEO_OPERATOR_EMBEDDED_MAX_REAL_SCRATCH_BYTES ((size_t)sizeof(geo_operator_mv_real_t))

typedef struct {
    uint8_t maximum_dimension;
    uint8_t maximum_blades;
    uint16_t maximum_terms;
    size_t i32_scratch_bytes;
    size_t fixed_scratch_bytes;
    size_t real_scratch_bytes;
    bool heap_required;
    bool runtime_parser_required;
} geo_operator_embedded_limits_t;

static inline geo_operator_embedded_limits_t geo_operator_embedded_limits(void) {
    geo_operator_embedded_limits_t limits;
    limits.maximum_dimension = GEO_OPERATOR_MAX_DIMENSION;
    limits.maximum_blades = GEO_OPERATOR_MAX_BLADES;
    limits.maximum_terms = GEO_OPERATOR_MAX_TERMS;
    limits.i32_scratch_bytes = GEO_OPERATOR_EMBEDDED_MAX_I32_SCRATCH_BYTES;
    limits.fixed_scratch_bytes = GEO_OPERATOR_EMBEDDED_MAX_FIXED_SCRATCH_BYTES;
    limits.real_scratch_bytes = GEO_OPERATOR_EMBEDDED_MAX_REAL_SCRATCH_BYTES;
    limits.heap_required = false;
    limits.runtime_parser_required = false;
    return limits;
}

#ifdef __cplusplus
}
#endif

#endif
