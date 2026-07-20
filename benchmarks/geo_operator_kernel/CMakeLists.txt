cmake_minimum_required(VERSION 3.24)

if(WIN32)
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} --use-local-env")
endif()

project(geo_operator_kernel_v5_1 LANGUAGES C CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_library(geo_operator_kernel_v5_1 STATIC
    ../../src/operator_kernel.c
    ../../src/fixed.c
)

target_include_directories(geo_operator_kernel_v5_1 PUBLIC ../../include)
target_compile_features(geo_operator_kernel_v5_1 PUBLIC c_std_11)

if(MSVC)
    target_compile_options(geo_operator_kernel_v5_1 PRIVATE /W4 /permissive-)
else()
    target_compile_options(geo_operator_kernel_v5_1 PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    target_link_libraries(geo_operator_kernel_v5_1 PUBLIC m)
endif()

add_executable(geo_operator_host_smoke ../../tests/test_operator_kernel.c)
target_link_libraries(geo_operator_host_smoke PRIVATE geo_operator_kernel_v5_1)
target_compile_features(geo_operator_host_smoke PRIVATE c_std_11)

add_executable(geo_operator_cuda_smoke cuda_smoke.cu)
target_compile_features(geo_operator_cuda_smoke PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(geo_operator_cuda_smoke PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF
)
if(MSVC)
    target_compile_options(geo_operator_cuda_smoke PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=/O2;-Xcompiler=/W4;-Xcompiler=/wd4127>
    )
else()
    target_compile_options(geo_operator_cuda_smoke PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
endif()
