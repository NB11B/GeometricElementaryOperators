cmake_minimum_required(VERSION 3.16)

project(GeometricElementaryOperators
    VERSION 0.19.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(GEO_BUILD_TESTS "Build host-side verification tests" ON)
option(GEO_BUILD_BENCHMARKS "Build host-side benchmarks" ON)
option(GEO_BUILD_CUDA "Build the optional CUDA 13.x backend and harness" OFF)
option(GEO_USE_DOUBLE "Use double precision instead of float" ON)
option(GEO_ENABLE_STACK_USAGE "Emit compiler stack-usage reports" OFF)
option(GEO_ENABLE_LINK_MAP "Emit linker map for benchmark executable" OFF)
option(GEO_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
set(GEO_FIXED_FRACTION_BITS 16 CACHE STRING
    "Fractional bits for signed 32-bit fixed-point arithmetic (1..30)"
)
if(GEO_FIXED_FRACTION_BITS LESS 1 OR GEO_FIXED_FRACTION_BITS GREATER 30)
    message(FATAL_ERROR "GEO_FIXED_FRACTION_BITS must be in the range 1..30")
endif()

if(GEO_USE_DOUBLE)
    set(GEO_PRECISION_NAME double)
else()
    set(GEO_PRECISION_NAME float)
endif()

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
    src/cortex_m_dwt.c
    src/fixed.c
    src/fixed_geb36.c
    src/fixed_program.c
    src/fixed_omega.c
    src/fixed_banked.c
    src/fixed_control.c
    src/native_generated.c
)

target_include_directories(geo_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(geo_kernel PUBLIC
    GEO_FIXED_FRACTION_BITS=${GEO_FIXED_FRACTION_BITS}
)

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

if(GEO_BUILD_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.24)
        message(FATAL_ERROR "GEO_BUILD_CUDA requires CMake 3.24 or newer")
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "GEO_BUILD_CUDA=ON requires the CUDA 13.x compiler")
    endif()
    enable_language(CUDA)

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0 OR
       CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
        message(FATAL_ERROR
            "CUDA Toolkit 13.x is required; detected ${CMAKE_CUDA_COMPILER_VERSION}"
        )
    endif()

    find_package(CUDAToolkit 13.0 REQUIRED)

    add_library(geo_cuda STATIC cuda/geo_cuda.cu)
    target_include_directories(geo_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(geo_cuda PUBLIC geo_kernel CUDA::cudart ${GEO_MATH_LIBS})
    target_compile_features(geo_cuda PUBLIC cuda_std_17)
    set_target_properties(geo_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
endif()

find_package(Python3 COMPONENTS Interpreter)

if(GEO_BUILD_CUDA AND Python3_Interpreter_FOUND)
    set(GEO_CUDA_SCHEDULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cuda-schedules)
    set(GEO_CUDA_SCHEDULE_NAMES
        addition
        geometric_product
        vector_dot
        vector_wedge
        rotor_action
    )
    set(GEO_CUDA_SCHEDULE_SOURCES)
    set(GEO_CUDA_SCHEDULE_HEADERS)
    foreach(schedule_name IN LISTS GEO_CUDA_SCHEDULE_NAMES)
        list(APPEND GEO_CUDA_SCHEDULE_SOURCES
            ${GEO_CUDA_SCHEDULE_DIR}/geo_cuda_schedule_${schedule_name}.cu
        )
        list(APPEND GEO_CUDA_SCHEDULE_HEADERS
            ${GEO_CUDA_SCHEDULE_DIR}/geo_cuda_schedule_${schedule_name}.h
        )
    endforeach()

    add_custom_command(
        OUTPUT
            ${GEO_CUDA_SCHEDULE_DIR}/cuda_schedule_manifest.json
            ${GEO_CUDA_SCHEDULE_SOURCES}
            ${GEO_CUDA_SCHEDULE_HEADERS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GEO_CUDA_SCHEDULE_DIR}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cuda_schedule.py
            --out-dir ${GEO_CUDA_SCHEDULE_DIR}
            --self-test
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/addition_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/geometric_product_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/vector_dot_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/vector_wedge_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/rotor_action_schedule.json
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cuda_schedule.py
            ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/addition_schedule.json
            ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/geometric_product_schedule.json
            ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/vector_dot_schedule.json
            ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/vector_wedge_schedule.json
            ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/rotor_action_schedule.json
        VERBATIM
    )
    set_source_files_properties(${GEO_CUDA_SCHEDULE_SOURCES} PROPERTIES GENERATED TRUE)
    add_library(geo_cuda_schedules STATIC ${GEO_CUDA_SCHEDULE_SOURCES})
    target_include_directories(geo_cuda_schedules PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GEO_CUDA_SCHEDULE_DIR}
    )
    target_link_libraries(geo_cuda_schedules PUBLIC geo_cuda)
    target_compile_features(geo_cuda_schedules PUBLIC cuda_std_17)
    set_target_properties(geo_cuda_schedules PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    add_custom_target(generate_cuda_schedules
        DEPENDS
            ${GEO_CUDA_SCHEDULE_DIR}/cuda_schedule_manifest.json
            ${GEO_CUDA_SCHEDULE_SOURCES}
            ${GEO_CUDA_SCHEDULE_HEADERS}
    )
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
        fixed_program
        fixed_omega
        fixed_banked
        fixed_control
        cortex_m_dwt
        native_generated
        release_hardening
        abi_consumer
    )
        add_executable(test_${test_name} tests/test_${test_name}.c)
        geo_link_target(test_${test_name})
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()

    if(GEO_BUILD_CUDA)
        add_executable(test_cuda tests/test_cuda.cu)
        target_link_libraries(test_cuda PRIVATE geo_cuda)
        target_compile_features(test_cuda PRIVATE cuda_std_17)
        add_test(NAME cuda_equivalence COMMAND test_cuda)
        set_tests_properties(cuda_equivalence PROPERTIES SKIP_RETURN_CODE 77)

        if(TARGET geo_cuda_schedules)
            add_executable(test_cuda_schedules tests/test_cuda_schedules.cu)
            add_dependencies(test_cuda_schedules generate_cuda_schedules)
            target_link_libraries(test_cuda_schedules PRIVATE geo_cuda_schedules)
            target_compile_features(test_cuda_schedules PRIVATE cuda_std_17)
            add_test(NAME cuda_schedule_equivalence COMMAND test_cuda_schedules)
            set_tests_properties(cuda_schedule_equivalence PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME generators
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_generators.py
        )
        add_test(
            NAME workload_report_contract
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_workload_report.py
        )
        add_test(
            NAME numerical_report_contract
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_numerical_report.py
        )
        set_tests_properties(generators PROPERTIES
            ENVIRONMENT "CC=${CMAKE_C_COMPILER};PYTHONDONTWRITEBYTECODE=1"
        )
        set_tests_properties(
            workload_report_contract numerical_report_contract
            PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
    endif()
endif()

if(GEO_BUILD_BENCHMARKS)
    add_executable(bench_host benchmarks/bench_host.c)
    geo_link_target(bench_host)

    add_executable(bench_compare benchmarks/bench_compare.c)
    geo_link_target(bench_compare)

    add_executable(bench_complete benchmarks/bench_complete.c)
    geo_link_target(bench_complete)

    add_executable(bench_workloads benchmarks/bench_workloads.c)
    geo_link_target(bench_workloads)

    add_executable(bench_numerical benchmarks/bench_numerical.c)
    geo_link_target(bench_numerical)

    if(GEO_BUILD_CUDA)
        add_executable(bench_cuda benchmarks/bench_cuda.cu)
        target_link_libraries(bench_cuda PRIVATE geo_cuda)
        if(TARGET geo_cuda_schedules)
            add_dependencies(bench_cuda generate_cuda_schedules)
            target_link_libraries(bench_cuda PRIVATE geo_cuda_schedules)
            target_compile_definitions(bench_cuda PRIVATE
                GEO_HAVE_GENERATED_CUDA_SCHEDULES=1
            )
        endif()
        target_compile_features(bench_cuda PRIVATE cuda_std_17)
    endif()

    if(GEO_BUILD_TESTS)
        add_test(
            NAME selected_path_benchmark
            COMMAND bench_complete --iterations 1000
        )
        add_test(
            NAME workload_smoke
            COMMAND bench_workloads --iterations 100 --warmup 10 --seed 608135816
        )
        add_test(
            NAME numerical_smoke
            COMMAND bench_numerical --samples 256 --seed 1779033703
        )
        if(GEO_BUILD_CUDA)
            add_test(
                NAME cuda_benchmark_smoke
                COMMAND bench_cuda
                    --batch 257
                    --iterations 2
                    --warmup 1
                    --operation all
            )
            set_tests_properties(cuda_benchmark_smoke PROPERTIES SKIP_RETURN_CODE 77)
            add_test(
                NAME cuda_benchmark_reverse_smoke
                COMMAND bench_cuda
                    --batch 257
                    --iterations 2
                    --warmup 1
                    --operation reverse
            )
            set_tests_properties(cuda_benchmark_reverse_smoke PROPERTIES
                SKIP_RETURN_CODE 77
            )
        endif()
    endif()

    if(GEO_ENABLE_LINK_MAP AND NOT MSVC)
        target_link_options(bench_host PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_host.map")
        target_link_options(bench_compare PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_compare.map")
        target_link_options(bench_complete PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_complete.map")
        target_link_options(bench_workloads PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_workloads.map")
        target_link_options(bench_numerical PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_numerical.map")
        if(GEO_BUILD_CUDA)
            target_link_options(bench_cuda PRIVATE "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bench_cuda.map")
        endif()
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

        if(GEO_BUILD_CUDA)
            add_custom_target(workload_report
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/workload_report.py
                    --cpu-executable $<TARGET_FILE:bench_workloads>
                    --cuda-executable $<TARGET_FILE:bench_cuda>
                    --require-cuda
                    --precision ${GEO_PRECISION_NAME}
                    --out-dir ${CMAKE_CURRENT_BINARY_DIR}/workload-report
                DEPENDS bench_workloads bench_cuda
                VERBATIM
            )
        else()
            add_custom_target(workload_report
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/workload_report.py
                    --cpu-executable $<TARGET_FILE:bench_workloads>
                    --precision ${GEO_PRECISION_NAME}
                    --out-dir ${CMAKE_CURRENT_BINARY_DIR}/workload-report
                DEPENDS bench_workloads
                VERBATIM
            )
        endif()

        add_custom_target(numerical_report
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/numerical_report.py
                --executable $<TARGET_FILE:bench_numerical>
                --precision ${GEO_PRECISION_NAME}
                --q-fraction-bits ${GEO_FIXED_FRACTION_BITS}
                --out-dir ${CMAKE_CURRENT_BINARY_DIR}/numerical-report
            DEPENDS bench_numerical
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

    add_custom_target(generate_rtl_schedules
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rtl_schedule.py
            --out-dir ${CMAKE_CURRENT_BINARY_DIR}/rtl-schedules
            --self-test
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/addition_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/geometric_product_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/vector_dot_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/vector_wedge_schedule.json
            --schedule-json ${CMAKE_CURRENT_SOURCE_DIR}/rtl/examples/rotor_action_schedule.json
        VERBATIM
    )
endif()
