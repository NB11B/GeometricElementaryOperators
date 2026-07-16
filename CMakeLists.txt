cmake_minimum_required(VERSION 3.16)

project(GeometricElementaryOperators
    VERSION 0.16.1
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
option(GEO_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

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

set_source_files_properties(src/structured_program.c PROPERTIES
    COMPILE_DEFINITIONS GEO_STRUCTURED_PROGRAM_IMPLEMENTATION=1
)
set_source_files_properties(src/banked.c PROPERTIES
    COMPILE_DEFINITIONS GEO_BANKED_IMPLEMENTATION=1
)

if(GEO_USE_DOUBLE)
    target_compile_definitions(geo_kernel PUBLIC GEO_REAL_IS_DOUBLE=1)
endif()

if(MSVC)
    target_compile_options(geo_kernel PRIVATE /W4)
    set(GEO_MATH_LIBS "")
else()
    target_compile_options(geo_kernel PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    set(GEO_MATH_LIBS m)
    if(GEO_ENABLE_STACK_USAGE)
        target_compile_options(geo_kernel PRIVATE -fstack-usage)
    endif()
    if(GEO_ENABLE_SANITIZERS)
        target_compile_options(geo_kernel PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(geo_kernel PUBLIC -fsanitize=address,undefined)
    endif()
endif()

function(geo_link_target target_name)
    target_link_libraries(${target_name} PRIVATE geo_kernel ${GEO_MATH_LIBS})
    if(GEO_ENABLE_SANITIZERS AND NOT MSVC)
        target_compile_options(${target_name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

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
        safety
    )
        add_executable(test_${test_name} tests/test_${test_name}.c)
        geo_link_target(test_${test_name})
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()
endif()

if(GEO_BUILD_BENCHMARKS)
    add_executable(bench_host benchmarks/bench_host.c)
    geo_link_target(bench_host)

    add_executable(bench_compare benchmarks/bench_compare.c)
    geo_link_target(bench_compare)

    if(GEO_ENABLE_LINK_MAP AND NOT MSVC)
        target_link_options(bench_host PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_host.map")
        target_link_options(bench_compare PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_compare.map")
    endif()
endif()
