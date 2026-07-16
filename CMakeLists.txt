cmake_minimum_required(VERSION 3.16)

project(GeometricElementaryOperators
    VERSION 0.1.0
    LANGUAGES C
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(GEO_BUILD_TESTS "Build host-side verification tests" ON)
option(GEO_USE_DOUBLE "Use double precision instead of float" ON)

add_library(geo_kernel STATIC
    src/cl20.c
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

    add_executable(test_cl20
        tests/test_cl20.c
    )

    target_link_libraries(test_cl20
        PRIVATE
            geo_kernel
            m
    )

    add_test(NAME cl20 COMMAND test_cl20)
endif()
