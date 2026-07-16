#include "geo/geb_witness.h"

static const geo_witness_node_t W_PSEUDOSCALAR[] = {
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
    {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0u, 1u, 0u}
};

static const geo_witness_node_t W_PRODUCT[] = {
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
    {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0u, 1u, 0u}
};

static const geo_witness_node_t W_SANDWICH[] = {
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 0u},
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 1u},
    {GEO_WITNESS_TERMINAL, GEO_LANE_NONE, 0u, 0u, 2u},
    {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 0u, 1u, 0u},
    {GEO_WITNESS_OMEGA, GEO_LANE_GEOMETRIC, 3u, 2u, 0u}
};

static const geo_geb_witness_t CATALOG[] = {
    {
        GEO_GEB_PSEUDOSCALAR,
        "pseudoscalar",
        GEO_WITNESS_OUTPUT_FORWARD,
        {W_PSEUDOSCALAR, 3u, 2u, 2u}
    },
    {
        GEO_GEB_GEOMETRIC_PRODUCT,
        "geometric_product",
        GEO_WITNESS_OUTPUT_FORWARD,
        {W_PRODUCT, 3u, 2u, 2u}
    },
    {
        GEO_GEB_REVERSE_PRODUCT,
        "reverse_product",
        GEO_WITNESS_OUTPUT_REVERSE,
        {W_PRODUCT, 3u, 2u, 2u}
    },
    {
        GEO_GEB_ROTOR_ACTION,
        "rotor_action",
        GEO_WITNESS_OUTPUT_FORWARD,
        {W_SANDWICH, 5u, 3u, 4u}
    },
    {
        GEO_GEB_ROTOR_COMPOSITION,
        "rotor_composition",
        GEO_WITNESS_OUTPUT_FORWARD,
        {W_PRODUCT, 3u, 2u, 2u}
    },
    {
        GEO_GEB_DILATION,
        "dilation",
        GEO_WITNESS_OUTPUT_FORWARD,
        {W_SANDWICH, 5u, 3u, 4u}
    }
};

const geo_geb_witness_t *geo_geb_witness_catalog(size_t *count) {
    if (count != NULL) {
        *count = sizeof(CATALOG) / sizeof(CATALOG[0]);
    }
    return CATALOG;
}

const geo_geb_witness_t *geo_geb_witness_for_target(uint8_t target_id) {
    size_t count;
    size_t index;
    const geo_geb_witness_t *catalog = geo_geb_witness_catalog(&count);

    for (index = 0u; index < count; ++index) {
        if (catalog[index].target_id == target_id) {
            return &catalog[index];
        }
    }
    return NULL;
}

geo_status_t geo_geb_witness_extract(
    const geo_geb_witness_t *witness,
    const geo_state_t *root_state,
    geo_cl20_t *output
) {
    if (witness == NULL || root_state == NULL || output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    if ((root_state->active_lanes & GEO_LANE_GEOMETRIC) == 0u) {
        return GEO_STATUS_BAD_TREE;
    }

    if (witness->output == GEO_WITNESS_OUTPUT_REVERSE) {
        *output = root_state->geometric.reverse;
    } else {
        *output = root_state->geometric.forward;
    }
    return GEO_STATUS_OK;
}
