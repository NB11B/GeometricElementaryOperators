#include "geo/fixed_banked.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static geo_fixed_t fixed(double value) {
    geo_fixed_t output = 0;
    if (geo_fixed_from_double(value, &output) != GEO_FIXED_OK) {
        fprintf(stderr, "fixture conversion failed: %.17g\n", value);
        exit(EXIT_FAILURE);
    }
    return output;
}

static geo_fixed_geometric_register_t geometric_register(
    geo_fixed_cl20_t value
) {
    geo_fixed_geometric_register_t output;
    if (geo_fixed_opposite_from_cl20(value, &output.value) != GEO_FIXED_OMEGA_OK) {
        fprintf(stderr, "geometric fixture construction failed\n");
        exit(EXIT_FAILURE);
    }
    output.scale = (geo_fixed_scale_t){1, 1};
    return output;
}

static void test_scalar_bank(void) {
    geo_fixed_t scalars[3] = {fixed(0.0), fixed(1.0), fixed(0.25)};
    geo_fixed_banked_storage_t storage = {
        scalars, 3u,
        NULL, 0u,
        NULL, 0u
    };
    const geo_fixed_banked_instruction_t instruction = {
        GEO_FIXED_LANE_SCALAR,
        {GEO_FIXED_BANK_SCALAR, 2u},
        {GEO_FIXED_BANK_SCALAR, 0u},
        {GEO_FIXED_BANK_SCALAR, 1u}
    };
    const geo_fixed_banked_program_t program = {
        &instruction,
        1u,
        {GEO_FIXED_BANK_SCALAR, 2u},
        3u,
        0u,
        0u
    };
    geo_fixed_state_t result;

    expect(
        geo_fixed_banked_execute(&program, &storage) == GEO_FIXED_BANKED_OK,
        "scalar bank executes fixed EML"
    );
    expect(scalars[2] == fixed(1.0), "scalar bank stores EML result");
    expect(
        geo_fixed_banked_read_state(
            &storage,
            program.root,
            &result
        ) == GEO_FIXED_BANKED_OK,
        "scalar root can be read as a state"
    );
    expect(result.active_lanes == GEO_FIXED_LANE_SCALAR &&
        result.scalar == fixed(1.0),
        "scalar root preserves lane and value"
    );
}

static void test_geometric_bank(void) {
    geo_fixed_geometric_register_t geometrics[3];
    geo_fixed_banked_storage_t storage;
    const geo_fixed_banked_instruction_t instruction = {
        GEO_FIXED_LANE_GEOMETRIC,
        {GEO_FIXED_BANK_GEOMETRIC, 2u},
        {GEO_FIXED_BANK_GEOMETRIC, 0u},
        {GEO_FIXED_BANK_GEOMETRIC, 1u}
    };
    const geo_fixed_banked_program_t program = {
        &instruction,
        1u,
        {GEO_FIXED_BANK_GEOMETRIC, 2u},
        0u,
        3u,
        0u
    };
    geo_fixed_state_t result;

    geometrics[0] = geometric_register(
        (geo_fixed_cl20_t){0, fixed(1.0), 0, 0}
    );
    geometrics[1] = geometric_register(
        (geo_fixed_cl20_t){0, 0, fixed(1.0), 0}
    );
    memset(&geometrics[2], 0, sizeof(geometrics[2]));
    geometrics[0].scale = (geo_fixed_scale_t){6, 8};
    geometrics[1].scale = (geo_fixed_scale_t){-10, 15};
    storage = (geo_fixed_banked_storage_t){
        NULL, 0u,
        geometrics, 3u,
        NULL, 0u
    };

    expect(
        geo_fixed_banked_execute(&program, &storage) == GEO_FIXED_BANKED_OK,
        "geometric bank executes opposite product"
    );
    expect(geometrics[2].value.forward.e12 == fixed(1.0),
        "geometric bank stores forward product");
    expect(geometrics[2].value.reverse.e12 == fixed(-1.0),
        "geometric bank stores propagated reverse");
    expect(geometrics[2].scale.numerator == -1 &&
        geometrics[2].scale.denominator == 2,
        "geometric bank stores reduced projective scale");
    expect(
        geo_fixed_banked_read_state(&storage, program.root, &result) ==
            GEO_FIXED_BANKED_OK &&
        result.active_lanes == GEO_FIXED_LANE_GEOMETRIC,
        "geometric root reads with the geometric lane"
    );
}

static void test_unified_bank(void) {
    geo_fixed_state_t unified[3];
    geo_fixed_banked_storage_t storage = {
        NULL, 0u,
        NULL, 0u,
        unified, 3u
    };
    const geo_fixed_banked_instruction_t instruction = {
        GEO_FIXED_LANE_ALL,
        {GEO_FIXED_BANK_UNIFIED, 2u},
        {GEO_FIXED_BANK_UNIFIED, 0u},
        {GEO_FIXED_BANK_UNIFIED, 1u}
    };
    const geo_fixed_banked_program_t program = {
        &instruction,
        1u,
        {GEO_FIXED_BANK_UNIFIED, 2u},
        0u,
        0u,
        3u
    };

    expect(
        geo_fixed_state_from_cl20(
            (geo_fixed_cl20_t){0, fixed(1.0), 0, 0},
            &unified[0]
        ) == GEO_FIXED_OMEGA_OK &&
        geo_fixed_state_from_cl20(
            (geo_fixed_cl20_t){0, 0, fixed(1.0), 0},
            &unified[1]
        ) == GEO_FIXED_OMEGA_OK,
        "construct unified bank fixtures"
    );
    unified[0].scalar = fixed(0.0);
    unified[1].scalar = fixed(1.0);
    unified[0].active_lanes = GEO_FIXED_LANE_ALL;
    unified[1].active_lanes = GEO_FIXED_LANE_ALL;
    unified[2] = geo_fixed_state_from_scalar(fixed(0.25));

    expect(
        geo_fixed_banked_execute(&program, &storage) == GEO_FIXED_BANKED_OK,
        "unified bank executes both lanes"
    );
    expect(unified[2].scalar == fixed(1.0), "unified bank stores scalar result");
    expect(unified[2].geometric.forward.e12 == fixed(1.0),
        "unified bank stores geometric result");
    expect(unified[2].active_lanes == GEO_FIXED_LANE_ALL,
        "unified bank marks both lanes active");
}

static void test_failures_leave_destination_unchanged(void) {
    geo_fixed_t scalars[3] = {fixed(0.0), fixed(1.0), fixed(0.25)};
    geo_fixed_geometric_register_t geometrics[3];
    geo_fixed_state_t unified[3];
    geo_fixed_banked_storage_t storage = {
        scalars, 3u,
        geometrics, 3u,
        unified, 3u
    };
    geo_fixed_banked_instruction_t instruction = {
        GEO_FIXED_LANE_ALL,
        {GEO_FIXED_BANK_GEOMETRIC, 2u},
        {GEO_FIXED_BANK_UNIFIED, 0u},
        {GEO_FIXED_BANK_UNIFIED, 1u}
    };
    geo_fixed_banked_program_t program = {
        &instruction,
        1u,
        {GEO_FIXED_BANK_GEOMETRIC, 2u},
        3u,
        3u,
        3u
    };
    geo_fixed_geometric_register_t sentinel;
    geo_fixed_banked_program_t malformed = {
        NULL,
        1u,
        {GEO_FIXED_BANK_SCALAR, 0u},
        1u,
        0u,
        0u
    };

    memset(geometrics, 0, sizeof(geometrics));
    memset(unified, 0, sizeof(unified));
    expect(
        geo_fixed_state_from_cl20(
            (geo_fixed_cl20_t){0, fixed(1.0), 0, 0},
            &unified[0]
        ) == GEO_FIXED_OMEGA_OK &&
        geo_fixed_state_from_cl20(
            (geo_fixed_cl20_t){0, 0, fixed(1.0), 0},
            &unified[1]
        ) == GEO_FIXED_OMEGA_OK,
        "construct failure fixtures"
    );
    unified[0].scalar = fixed(0.0);
    unified[1].scalar = fixed(1.0);
    unified[0].active_lanes = GEO_FIXED_LANE_ALL;
    unified[1].active_lanes = GEO_FIXED_LANE_ALL;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    geometrics[2] = sentinel;

    expect(
        geo_fixed_banked_execute(&program, &storage) ==
            GEO_FIXED_BANKED_TYPE_MISMATCH,
        "banked executor rejects destination/lane mismatch"
    );
    expect(memcmp(&geometrics[2], &sentinel, sizeof(sentinel)) == 0,
        "type failure leaves destination unchanged");

    instruction = (geo_fixed_banked_instruction_t){
        GEO_FIXED_LANE_GEOMETRIC,
        {GEO_FIXED_BANK_GEOMETRIC, 2u},
        {GEO_FIXED_BANK_GEOMETRIC, 0u},
        {GEO_FIXED_BANK_GEOMETRIC, 1u}
    };
    program.root = instruction.destination;
    geometrics[0] = geometric_register(
        (geo_fixed_cl20_t){INT32_MAX, 0, 0, 0}
    );
    geometrics[1] = geometric_register(
        (geo_fixed_cl20_t){fixed(2.0), 0, 0, 0}
    );
    geometrics[2] = sentinel;
    expect(
        geo_fixed_banked_execute(&program, &storage) ==
            GEO_FIXED_BANKED_OVERFLOW,
        "banked executor propagates fixed multiplication overflow"
    );
    expect(memcmp(&geometrics[2], &sentinel, sizeof(sentinel)) == 0,
        "arithmetic failure leaves destination unchanged");

    expect(
        geo_fixed_banked_execute(&malformed, &storage) ==
            GEO_FIXED_BANKED_NULL_ARGUMENT,
        "banked executor rejects null instruction stream"
    );
}

int main(void) {
    test_scalar_bank();
    test_geometric_bank();
    test_unified_bank();
    test_failures_leave_destination_unchanged();

    if (failures != 0) {
        fprintf(stderr, "%d fixed-banked assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All fixed-banked tests passed.");
    return EXIT_SUCCESS;
}
