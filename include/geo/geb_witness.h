#ifndef GEO_GEB_WITNESS_H
#define GEO_GEB_WITNESS_H

#include <stddef.h>
#include <stdint.h>

#include "geo/geb36.h"
#include "geo/compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_WITNESS_OUTPUT_FORWARD = 0,
    GEO_WITNESS_OUTPUT_REVERSE = 1
} geo_witness_output_t;

typedef struct {
    uint8_t target_id;
    const char *name;
    geo_witness_output_t output;
    geo_witness_tree_t tree;
} geo_geb_witness_t;

/*
 * Returns the currently reconstructed and executable GEB witness catalog.
 * These are real Omega trees, not direct calls to the GEB reference API.
 */
const geo_geb_witness_t *geo_geb_witness_catalog(size_t *count);
const geo_geb_witness_t *geo_geb_witness_for_target(uint8_t target_id);

/*
 * Extracts the selected geometric output from an executed witness root state.
 */
geo_status_t geo_geb_witness_extract(
    const geo_geb_witness_t *witness,
    const geo_state_t *root_state,
    geo_cl20_t *output
);

#ifdef __cplusplus
}
#endif

#endif
