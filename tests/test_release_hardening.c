#include "geo/banked.h"
#include "geo/cl20.h"
#include "geo/cl30.h"
#include "geo/control.h"
#include "geo/eml_embedded.h"
#include "geo/fixed.h"
#include "geo/fixed_geb36.h"
#include "geo/fused.h"
#include "geo/optimizer.h"
#include "geo/report.h"
#include "geo/structured_program.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    "optimized witness ABI size changed"
);

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_malformed_programs(void) {
    geo_struct_value_t registers[1];
    geo_struct_program_t structured = {NULL, 1u, 1u, 0u};
    geo_fused_program_t fused = {NULL, 1u, 1u, 0u};
    geo_banked_program_t banked;
    geo_banked_storage_t storage;
    char buffer[512];

    registers[0] = geo_struct_value_from_cl20(geo_cl20_zero());
    expect(
        geo_struct_program_execute(&structured, registers, 1u) == GEO_STATUS_NULL_ARGUMENT,
        "public structured executor rejects null instruction stream"
    );
    expect(
        geo_fused_execute(&fused, registers, 1u) == GEO_STATUS_NULL_ARGUMENT,
        "fused executor rejects null instruction stream"
    );

    memset(&banked, 0, sizeof(banked));
    memset(&storage, 0, sizeof(storage));
    banked.instruction_count = 1u;
    banked.root.kind = GEO_REGISTER_SCALAR;
    expect(
        geo_banked_execute(&banked, &storage) == GEO_STATUS_NULL_ARGUMENT,
        "public banked executor rejects null instruction stream"
    );
    expect(
        geo_emit_banked_program_c(&banked, "bad", buffer, sizeof(buffer)) == -1,
        "static emitter rejects null instruction stream"
    );

    banked.instruction_count = 0u;
    expect(
        geo_emit_banked_program_c(&banked, "empty", buffer, sizeof(buffer)) > 0,
        "static emitter supports empty programs"
    );
    expect(strstr(buffer, "[0]") == NULL, "empty program emits no zero-length array");
    expect(strstr(buffer, "= NULL") != NULL, "empty program emits null pointer");
}

static void test_nan_rejection(void) {
    geo_cl20_t a = geo_cl20_zero();
    geo_cl30_t c = geo_cl30_zero();
    geo_mat2_t m = geo_mat2_zero();
    a.scalar = (geo_real_t)NAN;
    c.c[0] = (geo_real_t)NAN;
    m.m00 = (geo_real_t)NAN;
    expect(!geo_cl20_near(a, a, (geo_real_t)1), "Cl20 near rejects NaN");
    expect(!geo_cl30_near(c, c, (geo_real_t)1), "Cl30 near rejects NaN");
    expect(!geo_mat2_near(m, m, (geo_real_t)1), "control near rejects NaN");
    expect(!geo_cl20_near(geo_cl20_zero(), geo_cl20_zero(), (geo_real_t)NAN),
        "near rejects NaN tolerance");
}

static void test_eml_nonfinite_log_rejection(void) {
    const geo_real_t sentinel = (geo_real_t)123.25;
    geo_real_t output = sentinel;

    expect(
        geo_eml_log((geo_real_t)NAN, GEO_EML_BALANCED, &output) == GEO_EML_DOMAIN,
        "embedded log rejects NaN"
    );
    expect(output == sentinel, "embedded log leaves output unchanged for NaN");

    output = sentinel;
    expect(
        geo_eml_log((geo_real_t)INFINITY, GEO_EML_BALANCED, &output) == GEO_EML_DOMAIN,
        "embedded log rejects positive infinity"
    );
    expect(output == sentinel, "embedded log leaves output unchanged for positive infinity");

    output = sentinel;
    expect(
        geo_eml_log((geo_real_t)-INFINITY, GEO_EML_BALANCED, &output) == GEO_EML_DOMAIN,
        "embedded log rejects negative infinity"
    );
    expect(output == sentinel, "embedded log leaves output unchanged for negative infinity");
}

static void test_checked_involutions(void) {
    geo_fixed_cl20_t value = {0, 0, 0, INT32_MIN};
    geo_fixed_cl20_t output;
    geo_fixed_geb_result_t result;
    expect(
        geo_fixed_cl20_reverse_checked(value, &output) == GEO_FIXED_OVERFLOW,
        "checked reverse reports unrepresentable negation"
    );
    expect(
        geo_fixed_geb36_execute(GEO_GEB_REVERSION, value, value, value, &result) ==
            GEO_FIXED_OVERFLOW,
        "fixed GEB reversion propagates overflow"
    );
}

static void test_optimizer_abi_layout(void) {
    expect(
        sizeof(geo_optimized_witness_t) == sizeof(geo_optimized_witness_legacy_layout_t),
        "optimized witness retains legacy caller-owned size"
    );
    expect(
        offsetof(geo_optimized_witness_t, eliminated_duplicate_nodes) ==
            offsetof(geo_optimized_witness_legacy_layout_t, eliminated_duplicate_nodes),
        "optimized witness final legacy field offset is unchanged"
    );
}

int main(void) {
    test_malformed_programs();
    test_nan_rejection();
    test_eml_nonfinite_log_rejection();
    test_checked_involutions();
    test_optimizer_abi_layout();
    if (failures != 0) {
        fprintf(stderr, "%d release-hardening assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All release-hardening regressions passed.");
    return EXIT_SUCCESS;
}
