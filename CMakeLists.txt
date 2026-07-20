cmake_minimum_required(VERSION 3.16)

project(GeometricElementaryOperators
    VERSION 0.18.2
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(GEO_BUILD_TESTS "Build host-side verification tests" ON)
option(GEO_BUILD_BENCHMARKS "Build host-side benchmarks" ON)
option(GEO_BUILD_TOOLS "Build standalone GEO developer tools" ON)
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
    src/executor_api.c
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
    src/fixed_geb36.c
    src/native_generated.c
    src/operator_kernel.c
    src/autodiff_v7.c
    src/full_cycle_v8.c
    src/batch_gp.c
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

find_package(Python3 COMPONENTS Interpreter)

if(GEO_BUILD_TOOLS)
    add_executable(geo_grad tools/geo_grad.c)
    geo_link_target(geo_grad)

    add_executable(geo_cycle
        tools/geo_cycle.c
        tools/geo_cycle_common.c
        tools/geo_cycle_loader.c
        tools/geo_cycle_io.c
    )
    set_source_files_properties(tools/geo_cycle_common.c PROPERTIES
        COMPILE_DEFINITIONS geo_cycle_load_model=geo_cycle_load_model_legacy
    )
    target_include_directories(geo_cycle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    geo_link_target(geo_cycle)
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
        safety
        fixed_geb36
        native_generated
        release_hardening
        abi_consumer
        operator_kernel
        operator_gradients
        autodiff_v7
        geo_grad_validation_v7_2
        full_cycle_v8
        batch_gp
    )
        add_executable(test_${test_name} tests/test_${test_name}.c)
        geo_link_target(test_${test_name})
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME generators
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_generators.py
        )
        set_tests_properties(generators PROPERTIES
            ENVIRONMENT "CC=${CMAKE_C_COMPILER}"
        )

        if(GEO_BUILD_TOOLS)
            add_test(
                NAME geo_grad_cli
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geo_grad_cli.py
                    --exe $<TARGET_FILE:geo_grad>
            )
            add_test(
                NAME geo_grad_validation_v7_2_cli
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geo_grad_validation_v7_2.py
                    --exe $<TARGET_FILE:geo_grad>
            )
            set_tests_properties(geo_grad_validation_v7_2_cli PROPERTIES TIMEOUT 300)

            add_test(
                NAME geo_cycle_v8_cli
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_geo_cycle_v8.py
                    --compiler ${CMAKE_CURRENT_SOURCE_DIR}/tools/geo_model.py
                    --runner $<TARGET_FILE:geo_cycle>
            )
            set_tests_properties(geo_cycle_v8_cli PROPERTIES TIMEOUT 600)
        endif()
    endif()
endif()

if(GEO_BUILD_BENCHMARKS)
    add_executable(bench_host benchmarks/bench_host.c)
    geo_link_target(bench_host)

    add_executable(bench_compare benchmarks/bench_compare.c)
    geo_link_target(bench_compare)

    add_executable(bench_complete benchmarks/bench_complete.c)
    geo_link_target(bench_complete)

    add_executable(bench_v8_full_cycle benchmarks/bench_v8_full_cycle.c)
    geo_link_target(bench_v8_full_cycle)

    add_executable(bench_v8_batch_gp benchmarks/bench_v8_batch_gp.c)
    geo_link_target(bench_v8_batch_gp)

    if(GEO_BUILD_TESTS)
        add_test(
            NAME selected_path_benchmark
            COMMAND bench_complete --iterations 1000
        )
    endif()

    if(GEO_ENABLE_LINK_MAP AND NOT MSVC)
        target_link_options(bench_host PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_host.map")
        target_link_options(bench_compare PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_compare.map")
        target_link_options(bench_complete PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_complete.map")
        target_link_options(bench_v8_full_cycle PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_v8_full_cycle.map")
        target_link_options(bench_v8_batch_gp PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_v8_batch_gp.map")
    endif()

    if(Python3_Interpreter_FOUND)
        add_custom_target(benchmark_report
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_report.py
                $<TARGET_FILE:bench_complete>
                --repetitions 9
                --out-dir ${CMAKE_CURRENT_BINARY_DIR}/benchmark-report
            DEPENDS bench_complete
            VERBATIM
        )
    endif()
endif()

if(Python3_Interpreter_FOUND)
    add_custom_target(generate_cl30
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_clpq.py
            --p 3 --q 0
            --prefix generated_cl30
            --out-dir ${CMAKE_CURRENT_BINARY_DIR}/generated-cl30
            --self-test
        VERBATIM
    )

    add_custom_target(generate_rtl
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rtl.py
            --out-dir ${CMAKE_CURRENT_BINARY_DIR}/rtl
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/rotor_action_schedule.json
            --self-test
        VERBATIM
    )
endif()
