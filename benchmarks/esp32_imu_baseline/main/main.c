#include <stdio.h>
#include "benchmark_common.h"
#include "conventional_filter.h"
#include "tinyml_baseline.h"

static conventional_filter_t conventional_state;
static tinyml_baseline_t tinyml_state;

void app_main(void)
{
    benchmark_impl_t implementations[] = {
        {
            .name = "C_conventional_quaternion",
            .state = &conventional_state,
            .reset = conventional_filter_reset,
            .step = conventional_filter_step,
            .state_bytes = sizeof(conventional_state),
        },
        {
            .name = "D_quantized_tinyml",
            .state = &tinyml_state,
            .reset = tinyml_baseline_reset,
            .step = tinyml_baseline_step,
            .state_bytes = sizeof(tinyml_state),
        },
    };

    printf("GEO_ESP32_IMU_BASELINE,mode=replay,sample_rate_hz=%u,samples=%u,runs=%u\n",
           BENCH_SAMPLE_RATE_HZ, BENCH_SAMPLE_COUNT, BENCH_RUNS);
    benchmark_print_csv_header();

    for (size_t implementation = 0;
         implementation < sizeof(implementations) / sizeof(implementations[0]);
         ++implementation) {
        for (uint32_t run = 0; run < BENCH_RUNS; ++run) {
            const benchmark_result_t result = benchmark_run(&implementations[implementation], run);
            benchmark_print_csv_row(&implementations[implementation], run, &result);
        }
    }

    printf("GEO_ESP32_IMU_BASELINE,status=complete\n");
}
