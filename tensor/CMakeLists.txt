cmake_minimum_required(VERSION 3.24)

project(GEOTensorLinear LANGUAGES C CXX CUDA)

option(GEO_TENSOR_USE_DOUBLE "Build the tensor ABI with double precision" OFF)

add_library(geo_tensor_linear STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../src/tensor_linear.c
    ${CMAKE_CURRENT_LIST_DIR}/../src/tensor_linear_cuda.cu
)

target_include_directories(geo_tensor_linear PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../include
)

target_compile_features(geo_tensor_linear PUBLIC c_std_11 cxx_std_17)
set_target_properties(geo_tensor_linear PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(GEO_TENSOR_USE_DOUBLE)
    target_compile_definitions(geo_tensor_linear PUBLIC GEO_REAL_IS_DOUBLE=1)
else()
    target_compile_definitions(geo_tensor_linear PUBLIC GEO_REAL_IS_DOUBLE=0)
endif()

if(MSVC)
    target_compile_options(geo_tensor_linear PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else()
    target_compile_options(geo_tensor_linear PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()

add_executable(test_tensor_linear_host
    ${CMAKE_CURRENT_LIST_DIR}/../tests/test_tensor_linear.c
)
target_link_libraries(test_tensor_linear_host PRIVATE geo_tensor_linear)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME tensor_linear_host COMMAND test_tensor_linear_host)
endif()
