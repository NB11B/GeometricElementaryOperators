#include <stdio.h>
#include <stdlib.h>

#include "benchmark_common.h"
#include "conventional_filter.h"
#include "geo_filter.h"
#include "geo_filter_fused.h"
#include "tinyml_baseline.h"

static geo_float_filter_t geo_float_generic_state;
static geo_fixed_filter_t geo_fixed_generic_state;
static geo_float_fused_filter_t geo_float_fused_state;
static geo_fixed_fused_filter_t geo_fixed_fused_state;
static conventional_filter_t conventional_state;
static tinyml_baseline_t tinyml_state;

void app_main(void)
{
    benchmark_impl_t implementations[] = {
        {
            .name = "A0_geo_float_generic",
            .state = &geo_float_generic_state,
            .reset = geo_float_filter_reset,
            .step = geo_float_filter_step,
            .state_bytes = sizeof(geo_float_generic_state),
        },
        {
            .name = "B0_geo_fixed_q16_generic",
            .state = &geo_fixed_generic_state,
            .reset = geo_fixed_filter_reset,
            .step = geo_fixed_filter_step,
            .state_bytes = sizeof(geo_fixed_generic_state),
        },
        {
            .name = "A1_geo_float_fused",
            .state = &geo_float_fused_state,
            .reset = geo_float_fused_filter_reset,
            .step = geo_float_fused_filter_step,
            .state_bytes = sizeof(geo_float_fused_state),
        },
        {
            .name = "B1_geo_fixed_q16_fused",
            .state = &geo_fixed_fused_state,
            .reset = geo_fixed_fused_filter_reset,
            .step = geo_fixed_fused_filter_step,
            .state_bytes = sizeof(geo_fixed_fused_state),
        },
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

    if (!geo_filter_self_test()) {
        printf("GEO_AB_FUSION_CHECKS,status=fail\n");
        abort();
    }
    printf("GEO_AB_FUSION_CHECKS,status=pass\n");

    printf(
        "GEO_ESP32_IMU_BENCHMARK,mode=replay,sample_rate_hz=%u,samples=%u,"
        "runs=%u,implementations=%u,q_fraction_bits=%u\n",
        BENCH_SAMPLE_RATE_HZ,
        BENCH_SAMPLE_COUNT,
        BENCH_RUNS,
        (unsigned)(sizeof(implementations) / sizeof(implementations[0])),
        (unsigned)GEO_FIXED_FRACTION_BITS
    );
    benchmark_print_csv_header();

    for (size_t implementation = 0;
         implementation < sizeof(implementations) / sizeof(implementations[0]);
         ++implementation) {
        for (uint32_t run = 0; run < BENCH_RUNS; ++run) {
            const benchmark_result_t result = benchmark_run(
                &implementations[implementation],
                run
            );
            benchmark_print_csv_row(
                &implementations[implementation],
                run,
                &result
            );
        }
    }

    printf("GEO_ESP32_IMU_BENCHMARK,status=complete\n");
}
