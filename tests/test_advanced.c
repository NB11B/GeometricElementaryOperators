#include "geo/cl30.h"
#include "geo/eml_embedded.h"
#include "geo/fused.h"
#include "geo/report.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_ADV_TOL ((geo_real_t)1e-7)
#else
#define GEO_ADV_TOL ((geo_real_t)2e-5f)
#endif

static int failures = 0;

static void expect_true(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_eml(void) {
    geo_real_t value;
    geo_real_t reference;
    expect_true(geo_eml_exp((geo_real_t)0.75, GEO_EML_PRECISE, &value) == GEO_EML_OK, "embedded exp status");
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    reference = exp((geo_real_t)0.75);
#else
    reference = expf((geo_real_t)0.75f);
#endif
    expect_true(fabs((double)(value - reference)) < (double)GEO_ADV_TOL, "embedded exp accuracy");

    expect_true(geo_eml_log((geo_real_t)3.25, GEO_EML_PRECISE, &value) == GEO_EML_OK, "embedded log status");
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    reference = log((geo_real_t)3.25);
#else
    reference = logf((geo_real_t)3.25f);
#endif
    expect_true(fabs((double)(value - reference)) < (double)GEO_ADV_TOL, "embedded log accuracy");
    expect_true(geo_eml_log((geo_real_t)0, GEO_EML_PRECISE, &value) == GEO_EML_DOMAIN, "embedded log domain");
}

static void test_cl30(void) {
    const geo_cl30_t one = geo_cl30_basis(GEO_CL30_SCALAR);
    const geo_cl30_t e1 = geo_cl30_basis(GEO_CL30_E1);
    const geo_cl30_t e2 = geo_cl30_basis(GEO_CL30_E2);
    const geo_cl30_t e3 = geo_cl30_basis(GEO_CL30_E3);
    const geo_cl30_t e12 = geo_cl30_basis(GEO_CL30_E12);
    const geo_cl30_t e123 = geo_cl30_basis(GEO_CL30_E123);
    const geo_real_t tolerance = (geo_real_t)1e-6;

    expect_true(geo_cl30_near(geo_cl30_mul(e1, e1), one, tolerance), "Cl30 e1 square");
    expect_true(geo_cl30_near(geo_cl30_mul(e2, e2), one, tolerance), "Cl30 e2 square");
    expect_true(geo_cl30_near(geo_cl30_mul(e3, e3), one, tolerance), "Cl30 e3 square");
    expect_true(geo_cl30_near(geo_cl30_mul(e1, e2), e12, tolerance), "Cl30 e1e2");
    expect_true(geo_cl30_near(geo_cl30_mul(e12, e3), e123, tolerance), "Cl30 e12e3");
    expect_true(geo_cl30_near(geo_cl30_reverse(e123), (geo_cl30_t){{0,0,0,0,0,0,0,-1}}, tolerance), "Cl30 trivector reverse");
}

static void test_fused(void) {
    geo_fused_instruction_t instruction;
    geo_fused_program_t program;
    geo_struct_value_t registers[4];
    geo_cl20_t result;
    const geo_cl20_t a = geo_cl20_make((geo_real_t)1, (geo_real_t)2, (geo_real_t)-1, (geo_real_t)0.5);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)-2, (geo_real_t)1, (geo_real_t)3, (geo_real_t)-0.25);

    registers[0] = geo_struct_value_from_cl20(a);
    registers[1] = geo_struct_value_from_cl20(b);
    registers[2] = geo_struct_value_from_cl20(geo_cl20_reverse(a));
    registers[3] = geo_struct_value_from_cl20(geo_cl20_zero());

    expect_true(geo_fused_program_for_target(GEO_GEB_ADDITION, &instruction, &program) == GEO_STATUS_OK, "fused add plan");
    expect_true(geo_fused_execute(&program, registers, 4) == GEO_STATUS_OK, "fused add execute");
    expect_true(geo_struct_read_cl20(&registers[program.root_register], &result) == GEO_STATUS_OK, "fused add read");
    expect_true(geo_cl20_near(result, geo_geb_addition(a, b), (geo_real_t)1e-6), "fused add result");

    registers[0] = geo_struct_value_from_cl20(a);
    registers[1] = geo_struct_value_from_cl20(b);
    registers[2] = geo_struct_value_from_cl20(geo_cl20_reverse(a));
    expect_true(geo_fused_program_for_target(GEO_GEB_ROTOR_ACTION, &instruction, &program) == GEO_STATUS_OK, "fused rotor plan");
    expect_true(geo_fused_execute(&program, registers, 4) == GEO_STATUS_OK, "fused rotor execute");
    expect_true(geo_struct_read_cl20(&registers[program.root_register], &result) == GEO_STATUS_OK, "fused rotor read");
    expect_true(geo_cl20_near(result, geo_geb_rotor_action(a, b), (geo_real_t)1e-6), "fused rotor result");
}

static void test_report_and_emitter(void) {
    geo_banked_instruction_t instruction = {
        GEO_LANE_GEOMETRIC,
        {GEO_REGISTER_GEOMETRIC, 2u},
        {GEO_REGISTER_GEOMETRIC, 0u},
        {GEO_REGISTER_GEOMETRIC, 1u}
    };
    geo_banked_program_t program = {
        &instruction, 1u, {GEO_REGISTER_GEOMETRIC, 2u}, 0u, 3u, 0u,
        3u * sizeof(geo_geometric_register_t)
    };
    geo_compiler_report_t report = {5u, 3u, 2u, 1u, 1u, 1u, 1u, 0u, 3u, 0u, program.required_bytes};
    char json[512];
    char source[2048];

    expect_true(geo_compiler_report_json(&report, json, sizeof(json)) > 0, "report json emit");
    expect_true(strstr(json, "\"runtime_bytes\"") != NULL, "report json field");
    expect_true(geo_emit_banked_program_c(&program, "demo", source, sizeof(source)) > 0, "static C emit");
    expect_true(strstr(source, "demo_instructions") != NULL, "static C symbol");
}

int main(void) {
    test_eml();
    test_cl30();
    test_fused();
    test_report_and_emitter();

    if (failures != 0) {
        fprintf(stderr, "%d advanced assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All advanced backend tests passed.");
    return EXIT_SUCCESS;
}
