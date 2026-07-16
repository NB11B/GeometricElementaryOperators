#include "geo_esp32.h"

#include <inttypes.h>

#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "geo_kernel";

static uint64_t geo_esp32_now(void *context) {
    (void)context;
    return (uint64_t)esp_timer_get_time();
}

geo_cycle_source_t geo_esp32_cycle_source(void) {
    geo_cycle_source_t source;
    source.now = geo_esp32_now;
    source.context = NULL;
    source.frequency_hz = UINT64_C(1000000);
    return source;
}

void geo_esp32_print_memory_report(void) {
    geo_memory_report_t report;
    geo_memory_report(&report);

    ESP_LOGI(TAG, "precision bytes=%u", (unsigned)report.sizeof_real);
    ESP_LOGI(TAG, "geo_cl20_t bytes=%u", (unsigned)report.sizeof_cl20);
    ESP_LOGI(TAG, "geo_opposite_t bytes=%u", (unsigned)report.sizeof_opposite);
    ESP_LOGI(TAG, "geo_state_t bytes=%u", (unsigned)report.sizeof_state);
    ESP_LOGI(TAG, "geometric bank register bytes=%u", (unsigned)report.sizeof_geometric_register);
    ESP_LOGI(TAG, "flat instruction bytes=%u", (unsigned)report.sizeof_instruction);
    ESP_LOGI(TAG, "banked instruction bytes=%u", (unsigned)report.sizeof_banked_instruction);
    ESP_LOGI(TAG, "structured value bytes=%u", (unsigned)report.sizeof_struct_value);
}

void geo_esp32_run_smoke_benchmarks(uint64_t iterations) {
    const geo_cycle_source_t clock = geo_esp32_cycle_source();
    geo_benchmark_result_t result;
    geo_status_t status;

    status = geo_benchmark_cl20_product(&clock, iterations, &result);
    if (status == GEO_STATUS_OK) {
        ESP_LOGI(
            TAG,
            "%s iterations=%" PRIu64 " us/op=%.6f",
            result.name,
            result.iterations,
            result.ticks_per_iteration
        );
    } else {
        ESP_LOGE(TAG, "cl20 benchmark failed: %d", (int)status);
    }

    status = geo_benchmark_direct_rotor_action(&clock, iterations, &result);
    if (status == GEO_STATUS_OK) {
        ESP_LOGI(
            TAG,
            "%s iterations=%" PRIu64 " us/op=%.6f",
            result.name,
            result.iterations,
            result.ticks_per_iteration
        );
    } else {
        ESP_LOGE(TAG, "rotor benchmark failed: %d", (int)status);
    }
}
