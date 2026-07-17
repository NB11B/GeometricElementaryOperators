idf_component_register(
    SRCS
        "main.c"
        "benchmark_common.c"
        "replay_trace.c"
        "geo_filter.c"
        "geo_filter_fused.c"
        "geo_filter_self_test.c"
        "conventional_filter.c"
        "tinyml_baseline.c"
    INCLUDE_DIRS "."
    REQUIRES
        esp32
        esp_timer
        heap
)

target_compile_options(
    ${COMPONENT_LIB}
    PRIVATE
        -O3
        -Wall
        -Wextra
        -Wpedantic
)
