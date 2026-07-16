#ifndef GEO_LOWERING_H
#define GEO_LOWERING_H

#include <stddef.h>
#include <stdint.h>

#include "geo/banked.h"
#include "geo/control.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_ROUTE_UNSUPPORTED = 0,
    GEO_ROUTE_ZERO = 1,
    GEO_ROUTE_IDENTITY = 2,
    GEO_ROUTE_NEGATE = 3,
    GEO_ROUTE_KEEP_FIRST = 4,
    GEO_ROUTE_KEEP_SECOND = 5,
    GEO_ROUTE_SECOND_TO_FIRST = 6,
    GEO_ROUTE_FIRST_TO_SECOND = 7,
    GEO_ROUTE_EXCHANGE = 8
} geo_route_kind_t;

typedef struct {
    geo_scale_t *register_scales;
    size_t register_scale_capacity;
    uint8_t *known_flags;
    size_t known_flag_capacity;
} geo_scale_workspace_t;

typedef struct {
    const geo_scale_t *register_scales;
    const uint8_t *known_flags;
    size_t register_count;
    size_t propagated_geometric_results;
    size_t unit_scale_results;
} geo_scale_plan_t;

/*
 * Propagates rational projective scales through geometric Omega products.
 * Scalar-only instructions receive unit scale. A geometric result has known
 * scale exactly when both geometric operands have known valid scales.
 */
geo_status_t geo_propagate_scales(
    const geo_folded_program_t *program,
    geo_scale_workspace_t *workspace,
    geo_scale_plan_t *output
);

/* Recognizes fixed control matrices and replaces them with routing enums. */
geo_route_kind_t geo_route_classify(geo_mat2_t control, geo_real_t tolerance);

/*
 * Applies a compiled route to a two-lane state pair. This is data movement and
 * optional sign inversion only; it performs no runtime matrix multiplication.
 */
geo_status_t geo_route_apply(
    geo_route_kind_t route,
    const geo_state_t *first,
    const geo_state_t *second,
    geo_state_t *out_first,
    geo_state_t *out_second
);

#ifdef __cplusplus
}
#endif

#endif
