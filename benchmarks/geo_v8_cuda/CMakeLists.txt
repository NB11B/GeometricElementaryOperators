cmake_minimum_required(VERSION 3.24)

project(GEOV8Cuda LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

get_filename_component(GEO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_library(geo_v8_cuda STATIC
    "${GEO_ROOT}/src/batch_gp.c"
    "${GEO_ROOT}/src/batch_gp_cuda.cu"
    "${GEO_ROOT}/src/hand_cuda_comparator.cu"
)

target_include_directories(geo_v8_cuda PUBLIC "${GEO_ROOT}/include")
target_compile_definitions(geo_v8_cuda PUBLIC GEO_REAL_IS_DOUBLE=1)
set_target_properties(geo_v8_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
)

if(MSVC)
    target_compile_options(geo_v8_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:/W4>)
else()
    target_compile_options(geo_v8_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>)
endif()

add_executable(test_v8_cuda_correctness "${GEO_ROOT}/tests/test_batch_gp_cuda.cu")
target_link_libraries(test_v8_cuda_correctness PRIVATE geo_v8_cuda)
set_target_properties(test_v8_cuda_correctness PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

add_executable(bench_v8_cuda "${GEO_ROOT}/benchmarks/bench_v8_cuda.cu")
target_link_libraries(bench_v8_cuda PRIVATE geo_v8_cuda)
set_target_properties(bench_v8_cuda PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

enable_testing()
add_test(NAME v8_cuda_correctness COMMAND test_v8_cuda_correctness)
set_tests_properties(v8_cuda_correctness PROPERTIES TIMEOUT 1800)
