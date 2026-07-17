#include "geo/cl20.h"
#include "geo/fixed_program.h"
#include "geo/geb36.h"
#include "geo_esp32.h"

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
#define GEO_ESP32_TOLERANCE ((geo_real_t)1.0e-12)
#else
#define GEO_ESP32_TOLERANCE ((geo_real_t)2.0e-5f)
#endif

#define GEO_ESP32_SOAK_ITERATIONS UINT64_C(10000)
#define GEO_ESP32_SMOKE_ITERATIONS UINT64_C(100000)

static const char *TAG = "geo_example";
static volatile geo_real_t float_sink = (geo_real_t)0;
static volatile uint32_t fixed_sink = UINT32_C(0);

static void fail(const char *message) {
    ESP_LOGE(TAG, "GEO_FAILURE,%s", message);
    abort();
}

static geo_fixed_t fixed_value(double value) {
    geo_fixed_t output = 0;
    if (geo_fixed_from_double(value, &output) != GEO_FIXED_OK) {
        fail("fixed fixture conversion");
    }
    return output;
}

static geo_fixed_cl20_t fixed_mv(double scalar, double e1, double e2, double e12) {
    geo_fixed_cl20_t value;
    value.scalar = fixed_value(scalar);
    value.e1 = fixed_value(e1);
    value.e2 = fixed_value(e2);
    value.e12 = fixed_value(e12);
    return value;
}

static geo_cl20_t fixed_to_cl20(geo_fixed_cl20_t value) {
    return geo_cl20_make(
        (geo_real_t)geo_fixed_to_double(value.scalar),
        (geo_real_t)geo_fixed_to_double(value.e1),
        (geo_real_t)geo_fixed_to_double(value.e2),
        (geo_real_t)geo_fixed_to_double(value.e12)
    );
}

static void check_cl20(geo_cl20_t actual, geo_cl20_t expected, const char *message) {
    if (!geo_cl20_near(actual, expected, GEO_ESP32_TOLERANCE)) {
        ESP_LOGE(
            TAG,
            "GEO_MISMATCH,name=%s,actual=%.9g|%.9g|%.9g|%.9g,expected=%.9g|%.9g|%.9g|%.9g",
            message,
            (double)actual.scalar,
            (double)actual.e1,
            (double)actual.e2,
            (double)actual.e12,
            (double)expected.scalar,
            (double)expected.e1,
            (double)expected.e2,
            (double)expected.e12
        );
        fail(message);
    }
}

static void run_reference_checks(void) {
    const geo_cl20_t a = geo_cl20_make(
        (geo_real_t)0.25,
        (geo_real_t)0.5,
        (geo_real_t)-0.25,
        (geo_real_t)0.125
    );
    const geo_cl20_t b = geo_cl20_make(
        (geo_real_t)-0.125,
        (geo_real_t)0.25,
        (geo_real_t)0.375,
        (geo_real_t)-0.0625
    );
    const geo_cl20_t rotor = geo_cl20_make(
        (geo_real_t)0.9689124217106447,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)-0.24740395925452294
    );
    const geo_fixed_program_instruction_t instructions[] = {
        {(uint8_t)GEO_GEB_ADDITION, 3u, 0u, 1u, 2u},
        {(uint8_t)GEO_GEB_ROTOR_ACTION, 4u, 3u, 0u, 2u}
    };
    const geo_fixed_program_t program = {instructions, 2u, 5u, 4u};
    geo_fixed_geb_result_t registers[5];
    geo_fixed_cl20_t fixed_output;
    geo_cl20_t expected;

    check_cl20(
        geo_geb_addition(a, b),
        geo_cl20_add(a, b),
        "float addition"
    );
    check_cl20(
        geo_geb_geometric_product(a, b),
        geo_cl20_mul(a, b),
        "float geometric product"
    );
    check_cl20(
        geo_geb_rotor_action(rotor, a),
        geo_cl20_mul(geo_cl20_mul(rotor, a), geo_cl20_reverse(rotor)),
        "float rotor action"
    );

    registers[0] = geo_fixed_program_value_from_cl20(
        fixed_mv(0.25, 0.5, -0.25, 0.125)
    );
    registers[1] = geo_fixed_program_value_from_cl20(
        fixed_mv(-0.125, 0.25, 0.375, -0.0625)
    );
    registers[2] = geo_fixed_program_value_from_cl20(
        fixed_mv(0.9689124217106447, 0.0, 0.0, -0.24740395925452294)
    );

    if (geo_fixed_program_execute(&program, registers, 5u) !=
        GEO_FIXED_PROGRAM_OK) {
        fail("fixed program reference execution");
    }
    if (geo_fixed_program_read_cl20(&registers[4], &fixed_output) !=
        GEO_FIXED_PROGRAM_OK) {
        fail("fixed program result read");
    }
    expected = geo_geb_rotor_action(rotor, geo_geb_addition(a, b));
    if (!geo_cl20_near(
            fixed_to_cl20(fixed_output),
            expected,
            (geo_real_t)(128.0 / (double)(INT64_C(1) << GEO_FIXED_FRACTION_BITS)))) {
        fail("fixed program numerical agreement");
    }

    ESP_LOGI(TAG, "GEO_CHECKS,status=pass");
}

static void run_soak_round(uint32_t round_index) {
    const geo_cl20_t a = geo_cl20_make(
        (geo_real_t)0.125,
        (geo_real_t)0.25,
        (geo_real_t)-0.375,
        (geo_real_t)0.0625
    );
    const geo_cl20_t b = geo_cl20_make(
        (geo_real_t)-0.25,
        (geo_real_t)0.125,
        (geo_real_t)0.25,
        (geo_real_t)-0.03125
    );
    const geo_cl20_t rotor = geo_cl20_make(
        (geo_real_t)0.9689124217106447,
        (geo_real_t)0,
        (geo_real_t)0,
        (geo_real_t)-0.24740395925452294
    );
    const geo_fixed_program_instruction_t instructions[] = {
        {(uint8_t)GEO_GEB_GEOMETRIC_PRODUCT, 3u, 0u, 1u, 2u},
        {(uint8_t)GEO_GEB_ROTOR_ACTION, 4u, 3u, 0u, 2u}
    };
    const geo_fixed_program_t program = {instructions, 2u, 5u, 4u};
    geo_fixed_geb_result_t registers[5];
    const size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    const size_t largest_before = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    int64_t float_begin;
    int64_t float_end;
    int64_t fixed_begin;
    int64_t fixed_end;
    uint64_t index;
    size_t heap_after;
    size_t largest_after;

    registers[0] = geo_fixed_program_value_from_cl20(
        fixed_mv(0.125, 0.25, -0.375, 0.0625)
    );
    registers[1] = geo_fixed_program_value_from_cl20(
        fixed_mv(-0.25, 0.125, 0.25, -0.03125)
    );
    registers[2] = geo_fixed_program_value_from_cl20(
        fixed_mv(0.9689124217106447, 0.0, 0.0, -0.24740395925452294)
    );

    float_begin = esp_timer_get_time();
    for (index = 0u; index < GEO_ESP32_SOAK_ITERATIONS; ++index) {
        const geo_cl20_t product = geo_geb_geometric_product(a, b);
        const geo_cl20_t result = geo_geb_rotor_action(rotor, product);
        float_sink += result.e1;
    }
    float_end = esp_timer_get_time();

    fixed_begin = esp_timer_get_time();
    for (index = 0u; index < GEO_ESP32_SOAK_ITERATIONS; ++index) {
        if (geo_fixed_program_execute(&program, registers, 5u) !=
            GEO_FIXED_PROGRAM_OK) {
            fail("fixed soak execution");
        }
        fixed_sink ^= (uint32_t)registers[4].as.cl20.e1;
    }
    fixed_end = esp_timer_get_time();

    heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    largest_after = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    if (heap_after != heap_before || largest_after != largest_before) {
        ESP_LOGE(
            TAG,
            "GEO_HEAP_DRIFT,round=%" PRIu32 ",free_before=%u,free_after=%u,largest_before=%u,largest_after=%u",
            round_index,
            (unsigned int)heap_before,
            (unsigned int)heap_after,
            (unsigned int)largest_before,
            (unsigned int)largest_after
        );
        fail("heap drift during allocation-free soak");
    }

    ESP_LOGI(
        TAG,
        "GEO_RESULT,round=%" PRIu32 ",backend=esp32_float,operation=product_rotor_chain,iterations=%" PRIu64 ",us_per_op=%.6f,free_heap=%u,largest_block=%u,min_free_heap=%u",
        round_index,
        GEO_ESP32_SOAK_ITERATIONS,
        (double)(float_end - float_begin) / (double)GEO_ESP32_SOAK_ITERATIONS,
        (unsigned int)heap_after,
        (unsigned int)largest_after,
        (unsigned int)esp_get_minimum_free_heap_size()
    );
    ESP_LOGI(
        TAG,
        "GEO_RESULT,round=%" PRIu32 ",backend=esp32_fixed,operation=product_rotor_chain,iterations=%" PRIu64 ",us_per_op=%.6f,free_heap=%u,largest_block=%u,min_free_heap=%u",
        round_index,
        GEO_ESP32_SOAK_ITERATIONS,
        (double)(fixed_end - fixed_begin) / (double)GEO_ESP32_SOAK_ITERATIONS,
        (unsigned int)heap_after,
        (unsigned int)largest_after,
        (unsigned int)esp_get_minimum_free_heap_size()
    );
}

void app_main(void) {
    esp_chip_info_t chip_info;
    uint32_t round_index = 0u;

    esp_chip_info(&chip_info);
    ESP_LOGI(
        TAG,
        "GEO_DEVICE,model=%d,cores=%d,revision=%d,features=0x%" PRIx64 ",free_heap=%u",
        (int)chip_info.model,
        (int)chip_info.cores,
        (int)chip_info.revision,
        (uint64_t)chip_info.features,
        (unsigned int)esp_get_free_heap_size()
    );

    run_reference_checks();
    geo_esp32_print_memory_report();
    geo_esp32_run_smoke_benchmarks(GEO_ESP32_SMOKE_ITERATIONS);

    for (;;) {
        run_soak_round(round_index);
        ++round_index;
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
