#include "geo/geb36.h"
#include "geo/structured_program.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define TOL ((geo_real_t)1e-12)
#else
#define TOL ((geo_real_t)1e-5f)
#endif

static int failures = 0;

static void expect_true(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void expect_real(geo_real_t actual, geo_real_t expected, const char *message) {
    if (fabs((double)(actual - expected)) > (double)TOL) {
        ++failures;
        fprintf(stderr, "FAIL: %s actual=%.17g expected=%.17g\n",
            message, (double)actual, (double)expected);
    }
}

static void expect_cl20(geo_cl20_t actual, geo_cl20_t expected, const char *message) {
    if (!geo_cl20_near(actual, expected, TOL)) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static geo_struct_value_t run_binary(
    geo_struct_opcode_t opcode,
    geo_struct_value_t left,
    geo_struct_value_t right
) {
    const geo_struct_instruction_t instructions[] = {
        {(uint8_t)opcode, 2u, 0u, 1u}
    };
    const geo_struct_program_t program = {instructions, 1u, 3u, 2u};
    geo_struct_value_t registers[3];
    registers[0] = left;
    registers[1] = right;
    expect_true(
        geo_struct_program_execute(&program, registers, 3u) == GEO_STATUS_OK,
        "structured binary execution"
    );
    return registers[2];
}

static geo_struct_value_t run_unary(
    geo_struct_opcode_t opcode,
    geo_struct_value_t input
) {
    const geo_struct_instruction_t instructions[] = {
        {(uint8_t)opcode, 1u, 0u, 0u}
    };
    const geo_struct_program_t program = {instructions, 1u, 2u, 1u};
    geo_struct_value_t registers[2];
    registers[0] = input;
    expect_true(
        geo_struct_program_execute(&program, registers, 2u) == GEO_STATUS_OK,
        "structured unary execution"
    );
    return registers[1];
}

int main(void) {
    const geo_cl20_t a = geo_cl20_make((geo_real_t)0, (geo_real_t)2, (geo_real_t)-3, (geo_real_t)0);
    const geo_cl20_t b = geo_cl20_make((geo_real_t)0, (geo_real_t)5, (geo_real_t)7, (geo_real_t)0);
    geo_struct_value_t result;
    geo_real_t scalar;
    geo_scaled_cl20_t scaled;

    result = run_unary(GEO_STRUCT_OP_VECTOR_NORM_SQUARED, geo_struct_value_from_cl20(a));
    expect_true(geo_struct_read_scalar(&result, &scalar) == GEO_STATUS_OK, "read norm squared");
    expect_real(scalar, geo_geb_vector_norm_squared(a), "norm squared witness");

    result = run_binary(
        GEO_STRUCT_OP_DISTANCE_SQUARED,
        geo_struct_value_from_cl20(a),
        geo_struct_value_from_cl20(b)
    );
    expect_true(geo_struct_read_scalar(&result, &scalar) == GEO_STATUS_OK, "read distance squared");
    expect_real(scalar, geo_geb_distance_squared(a, b), "distance squared witness");

    result = run_binary(
        GEO_STRUCT_OP_PROJECTION_NUMERATOR,
        geo_struct_value_from_cl20(a),
        geo_struct_value_from_cl20(b)
    );
    expect_true(geo_struct_read_scaled_cl20(&result, &scaled) == GEO_STATUS_OK, "read projection numerator");
    expect_cl20(scaled.represented, geo_geb_projection_numerator(a, b), "projection numerator witness");
    expect_true(scaled.scale.numerator == 1 && scaled.scale.denominator == 1, "projection scale");

    result = run_binary(
        GEO_STRUCT_OP_REJECTION_NUMERATOR,
        geo_struct_value_from_cl20(a),
        geo_struct_value_from_cl20(b)
    );
    expect_true(geo_struct_read_scaled_cl20(&result, &scaled) == GEO_STATUS_OK, "read rejection numerator");
    expect_cl20(scaled.represented, geo_geb_rejection_numerator(a, b), "rejection numerator witness");

    result = run_binary(
        GEO_STRUCT_OP_REFLECTION_NUMERATOR,
        geo_struct_value_from_cl20(a),
        geo_struct_value_from_cl20(b)
    );
    expect_true(geo_struct_read_scaled_cl20(&result, &scaled) == GEO_STATUS_OK, "read reflection numerator");
    expect_cl20(scaled.represented, geo_geb_reflection_numerator(a, b), "reflection numerator witness");

    result = run_unary(
        GEO_STRUCT_OP_VECTOR_INVERSE_PROJECTIVE,
        geo_struct_value_from_cl20(a)
    );
    expect_true(geo_struct_read_scaled_cl20(&result, &scaled) == GEO_STATUS_OK, "read projective inverse");
    expect_cl20(scaled.represented, geo_geb_vector_inverse_projective(a), "vector inverse projective witness");

    result = run_binary(
        GEO_STRUCT_OP_ANGLE_COSINE_NUMERATOR,
        geo_struct_value_from_cl20(a),
        geo_struct_value_from_cl20(b)
    );
    expect_true(geo_struct_read_scalar(&result, &scalar) == GEO_STATUS_OK, "read angle numerator");
    expect_real(scalar, geo_geb_angle_cosine_numerator(a, b), "angle numerator witness");

    {
        const geo_scaled_cl20_t projective = {geo_cl20_scale(a, (geo_real_t)2), {2, 1}};
        geo_cl20_t normalized;
        result = run_unary(
            GEO_STRUCT_OP_SCALED_CL20_NORMALIZE,
            geo_struct_value_from_scaled_cl20(projective)
        );
        expect_true(geo_struct_read_cl20(&result, &normalized) == GEO_STATUS_OK, "read normalized value");
        expect_cl20(normalized, a, "deferred normalization witness");
    }

    if (failures != 0) {
        fprintf(stderr, "%d metric program assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }

    puts("All metric/projective structured program tests passed.");
    return EXIT_SUCCESS;
}
