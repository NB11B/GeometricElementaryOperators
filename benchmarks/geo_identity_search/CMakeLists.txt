cmake_minimum_required(VERSION 3.24)

project(geo_identity_search LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_executable(
    geo_identity_search
    identity_search.cu
)

target_include_directories(
    geo_identity_search
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/generated"
)

target_compile_features(
    geo_identity_search
    PRIVATE
        cxx_std_17
        cuda_std_17
)

if(MSVC)
    target_compile_options(
        geo_identity_search
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2;/W4;/permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-diag-suppress=128;-Xcompiler=/O2;-Xcompiler=/W4;-Xcompiler=/wd4127>
    )
else()
    target_compile_options(
        geo_identity_search
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
endif()

set_target_properties(
    geo_identity_search
    PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_executable(
    geo_identity_host_smoke
    host_smoke.cpp
)

target_include_directories(
    geo_identity_host_smoke
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/generated"
)

target_compile_features(
    geo_identity_host_smoke
    PRIVATE
        cxx_std_17
)

if(MSVC)
    target_compile_options(
        geo_identity_host_smoke
        PRIVATE
            /O2
            /W4
            /permissive-
            /wd4127
    )
else()
    target_compile_options(
        geo_identity_host_smoke
        PRIVATE
            -O3
            -Wall
            -Wextra
            -Wpedantic
    )
endif()
