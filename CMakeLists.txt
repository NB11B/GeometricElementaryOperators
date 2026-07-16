cmake_minimum_required(VERSION 3.16)

project(GeometricElementaryOperators
    VERSION 0.16.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(GEO_BUILD_TESTS "Build host-side verification tests" ON)
option(GEO_BUILD_BENCHMARKS "Build host-side benchmarks" ON)
option(GEO_USE_DOUBLE "Use double precision instead of float" ON)
option(GEO_ENABLE_STACK_USAGE "Emit compiler stack-usage reports" OFF)
option(GEO_ENABLE_LINK_MAP "Emit linker map for benchmark executable" OFF)

add_library(geo_kernel STATIC
    src/cl20.c
    src/cl30.c
    src/opposite.c
    src/omega.c
    src/eml_embedded.c
    src/structured.c
    src/structured_program.c
    src/fused.c
    src/control.c
    src/geb36.c
    src/compiler.c
    src/optimizer.c
    src/folding.c
    src/banked.c
    src/lowering.c
    src/report.c
    src/geb_witness.c
    src/perf.c
    src/fixed.c
)

target_include_directories(geo_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(GEO_USE_DOUBLE)
    target_compile_definitions(geo_kernel PUBLIC GEO_REAL_IS_DOUBLE=1)
endif()

if(MSVC)
    target_compile_options(geo_kernel PRIVATE /W4)
else()
    target_compile_options(geo_kernel PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    if(GEO_ENABLE_STACK_USAGE)
        target_compile_options(geo_kernel PRIVATE -fstack-usage)
    endif()
endif()

if(GEO_BUILD_TESTS)
    enable_testing()

    foreach(test_name
        cl20
        omega
        structured
        structured_program
        metric_program
        closure_program
        control
        geb36
        compiler
        optimizer
        folding
        banked
        lowering
        geb_witness
        fixed
        randomized
        advanced
        fixed_extended
    )
        add_executable(test_${test_name} tests/test_${test_name}.c)
        target_link_libraries(test_${test_name} PRIVATE geo_kernel m)
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()
endif()

if(GEO_BUILD_BENCHMARKS)
    add_executable(bench_host benchmarks/bench_host.c)
    target_link_libraries(bench_host PRIVATE geo_kernel m)

    add_executable(bench_compare benchmarks/bench_compare.c)
    target_link_libraries(bench_compare PRIVATE geo_kernel m)

    if(GEO_ENABLE_LINK_MAP AND NOT MSVC)
        target_link_options(bench_host PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_host.map")
        target_link_options(bench_compare PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_compare.map")
    endif()
endif()
