idf_component_register(
    SRCS
        "main.c"
        "benchmark_common.c"
        "replay_trace.c"
        "conventional_filter.c"
        "tinyml_baseline.c"
    INCLUDE_DIRS "."
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -Wall -Wextra -Wpedantic)
