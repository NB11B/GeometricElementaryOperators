#include "geo/geb_witness.h"
#include "geo/optimizer.h"
#include "geo/folding.h"
#include "geo/banked.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define TOL ((geo_real_t)1e-12)
#else
#define TOL ((geo_real_t)1e-5f)
#endif

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static geo_status_t run_witness(
    const geo_geb_witness_t *witness,
    const geo_state_t *terminals,
    geo_cl20_t *output
) {
    geo_instruction_t optimized_instructions[8];
    uint8_t node_registers[8];
    uint8_t live_lanes[8];
    uint16_t representatives[8];
    geo_optimizer_workspace_t optimizer_workspace = {
        optimized_instructions, 8u,
        node_registers, 8u,
        live_lanes, 8u,
        representatives, 8u
    };
    geo_optimized_witness_t optimized;

    geo_instruction_t folded_instructions[8];
    geo_state_t initial_registers[8];
    uint8_t old_to_new[8];
    uint8_t constant_flags[8];
    uint8_t register_kinds[8];
    geo_folding_workspace_t folding_workspace = {
        folded_instructions, 8u,
        initial_registers, 8u,
        old_to_new, 8u,
        constant_flags, 8u,
        register_kinds, 8u
    };
    geo_folded_program_t folded;

    geo_banked_instruction_t banked_instructions[8];
    geo_banked_ref_t logical_refs[8];
    geo_banked_plan_workspace_t plan_workspace = {
        banked_instructions, 8u,
        logical_refs, 8u
    };
    geo_banked_program_t banked;

    geo_real_t scalars[8];
    geo_geometric_register_t geometrics[8];
    geo_state_t unified[8];
    geo_banked_storage_t storage = {
        scalars, 8u,
        geometrics, 8u,
        unified, 8u
    };
    geo_state_t root;
    uint8_t terminal_constant_flags[3] = {0u, 0u, 0u};
    geo_status_t status;

    status = geo_witness_compile_optimized(&witness->tree, &optimizer_workspace, &optimized);
    if (status != GEO_STATUS_OK) return status;

    status = geo_program_fold_constants(
        &optimized,
        terminals,
        terminal_constant_flags,
        witness->tree.terminal_count,
        &folding_workspace,
        &folded
    );
    if (status != GEO_STATUS_OK) return status;

    status = geo_banked_plan(&folded, &plan_workspace, &banked);
    if (status != GEO_STATUS_OK) return status;

    status = geo_banked_initialize(&folded, &banked, logical_refs, &storage);
    if (status != GEO_STATUS_OK) return status;

    status = geo_banked_execute(&banked, &storage);
    if (status != GEO_STATUS_OK) return status;

    status = geo_banked_read_state(&storage, banked.root, &root);
    if (status != GEO_STATUS_OK) return status;

    return geo_geb_witness_extract(witness, &root, output);
}

static void test_catalog(void) {
    size_t count = 0u;
    const geo_geb_witness_t *catalog = geo_geb_witness_catalog(&count);
    expect(catalog != NULL, "catalog exists");
    expect(count == 6u, "six reconstructed executable witnesses");
}

static void test_pseudoscalar(void) {
    geo_state_t terminals[2] = {
        geo_state_from_cl20(geo_geb_e1()),
        geo_state_from_cl20(geo_geb_e2())
    };
    geo_cl20_t actual;
    const geo_geb_witness_t *w = geo_geb_witness_for_target(GEO_GEB_PSEUDOSCALAR);
    expect(run_witness(w, terminals, &actual) == GEO_STATUS_OK, "run pseudoscalar witness");
    expect(geo_cl20_near(actual, geo_geb_pseudoscalar(), TOL), "compiled pseudoscalar matches reference");
}

static void test_product_and_reverse(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)1, (geo_real_t)2, (geo_real_t)-1, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)-2, (geo_real_t)1, (geo_real_t)3, (geo_real_t)-0.25);
    geo_state_t terminals[2] = {geo_state_from_cl20(a), geo_state_from_cl20(b)};
    geo_cl20_t actual;

    expect(run_witness(geo_geb_witness_for_target(GEO_GEB_GEOMETRIC_PRODUCT), terminals, &actual) == GEO_STATUS_OK,
           "run product witness");
    expect(geo_cl20_near(actual, geo_geb_geometric_product(a, b), TOL), "compiled product matches reference");

    expect(run_witness(geo_geb_witness_for_target(GEO_GEB_REVERSE_PRODUCT), terminals, &actual) == GEO_STATUS_OK,
           "run reverse-product witness");
    expect(geo_cl20_near(actual, geo_geb_reverse_product(a, b), TOL), "compiled reverse product matches reference");
}

static void test_sandwich_targets(void) {
    const geo_real_t c = (geo_real_t)cos(0.25);
    const geo_real_t s = (geo_real_t)sin(0.25);
    const geo_cl20_t rotor = geo_cl20_make(c, 0, 0, -s);
    const geo_cl20_t value = geo_cl20_make(0, 2, 1, 0);
    const geo_cl20_t reverse = geo_cl20_reverse(rotor);
    geo_state_t terminals[3] = {
        geo_state_from_cl20(rotor),
        geo_state_from_cl20(value),
        geo_state_from_cl20(reverse)
    };
    geo_cl20_t actual;

    expect(run_witness(geo_geb_witness_for_target(GEO_GEB_ROTOR_ACTION), terminals, &actual) == GEO_STATUS_OK,
           "run rotor-action witness");
    expect(geo_cl20_near(actual, geo_geb_rotor_action(rotor, value), TOL), "compiled rotor action matches reference");

    expect(run_witness(geo_geb_witness_for_target(GEO_GEB_DILATION), terminals, &actual) == GEO_STATUS_OK,
           "run dilation witness");
    expect(geo_cl20_near(actual, geo_geb_dilation(rotor, value), TOL), "compiled dilation matches reference");
}

static void test_rotor_composition(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)0.9, 0, 0, (geo_real_t)-0.2);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)0.8, 0, 0, (geo_real_t)-0.3);
    geo_state_t terminals[2] = {geo_state_from_cl20(a), geo_state_from_cl20(b)};
    geo_cl20_t actual;
    expect(run_witness(geo_geb_witness_for_target(GEO_GEB_ROTOR_COMPOSITION), terminals, &actual) == GEO_STATUS_OK,
           "run rotor-composition witness");
    expect(geo_cl20_near(actual, geo_geb_rotor_composition(a, b), TOL), "compiled rotor composition matches reference");
}

int main(void) {
    test_catalog();
    test_pseudoscalar();
    test_product_and_reverse();
    test_sandwich_targets();
    test_rotor_composition();

    if (failures != 0) {
        fprintf(stderr, "%d witness assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All compiled GEB witness tests passed.");
    return EXIT_SUCCESS;
}
