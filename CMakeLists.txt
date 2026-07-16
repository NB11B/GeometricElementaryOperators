cmake_minimum_required(VERSION 3.16)

project(GeometricElementaryOperators
    VERSION 0.14.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(GEO_BUILD_TESTS "Build host-side verification tests" ON)
option(GEO_USE_DOUBLE "Use double precision instead of float" ON)

add_library(geo_kernel STATIC
    src/cl20.c
    src/opposite.c
    src/omega.c
    src/structured.c
    src/structured_program.c
    src/control.c
    src/geb36.c
    src/compiler.c
    src/optimizer.c
    src/folding.c
    src/banked.c
    src/lowering.c
    src/geb_witness.c
)

target_include_directories(geo_kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(GEO_USE_DOUBLE)
    target_compile_definitions(geo_kernel PUBLIC GEO_REAL_IS_DOUBLE=1)
endif()

if(MSVC)
    target_compile_options(geo_kernel PRIVATE /W4)
else()
    target_compile_options(geo_kernel PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(GEO_BUILD_TESTS)
    enable_testing()

    add_executable(test_cl20 tests/test_cl20.c)
    target_link_libraries(test_cl20 PRIVATE geo_kernel m)
    add_test(NAME cl20 COMMAND test_cl20)

    add_executable(test_omega tests/test_omega.c)
    target_link_libraries(test_omega PRIVATE geo_kernel m)
    add_test(NAME omega COMMAND test_omega)

    add_executable(test_structured tests/test_structured.c)
    target_link_libraries(test_structured PRIVATE geo_kernel m)
    add_test(NAME structured COMMAND test_structured)

    add_executable(test_structured_program tests/test_structured_program.c)
    target_link_libraries(test_structured_program PRIVATE geo_kernel m)
    add_test(NAME structured_program COMMAND test_structured_program)

    add_executable(test_metric_program tests/test_metric_program.c)
    target_link_libraries(test_metric_program PRIVATE geo_kernel m)
    add_test(NAME metric_program COMMAND test_metric_program)

    add_executable(test_closure_program tests/test_closure_program.c)
    target_link_libraries(test_closure_program PRIVATE geo_kernel m)
    add_test(NAME closure_program COMMAND test_closure_program)

    add_executable(test_control tests/test_control.c)
    target_link_libraries(test_control PRIVATE geo_kernel m)
    add_test(NAME control COMMAND test_control)

    add_executable(test_geb36 tests/test_geb36.c)
    target_link_libraries(test_geb36 PRIVATE geo_kernel m)
    add_test(NAME geb36 COMMAND test_geb36)

    add_executable(test_compiler tests/test_compiler.c)
    target_link_libraries(test_compiler PRIVATE geo_kernel m)
    add_test(NAME compiler COMMAND test_compiler)

    add_executable(test_optimizer tests/test_optimizer.c)
    target_link_libraries(test_optimizer PRIVATE geo_kernel m)
    add_test(NAME optimizer COMMAND test_optimizer)

    add_executable(test_folding tests/test_folding.c)
    target_link_libraries(test_folding PRIVATE geo_kernel m)
    add_test(NAME folding COMMAND test_folding)

    add_executable(test_banked tests/test_banked.c)
    target_link_libraries(test_banked PRIVATE geo_kernel m)
    add_test(NAME banked COMMAND test_banked)

    add_executable(test_lowering tests/test_lowering.c)
    target_link_libraries(test_lowering PRIVATE geo_kernel m)
    add_test(NAME lowering COMMAND test_lowering)

    add_executable(test_geb_witness tests/test_geb_witness.c)
    target_link_libraries(test_geb_witness PRIVATE geo_kernel m)
    add_test(NAME geb_witness COMMAND test_geb_witness)
endif()
