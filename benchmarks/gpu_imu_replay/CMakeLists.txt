cmake_minimum_required(VERSION 3.24)

project(geo_gpu_imu_replay LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

add_executable(
    geo_gpu_imu_replay
    gpu_imu_replay.cu
)

target_include_directories(
    geo_gpu_imu_replay
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/generated"
)

target_compile_features(
    geo_gpu_imu_replay
    PRIVATE
        cxx_std_17
        cuda_std_17
)

if(MSVC)
    target_compile_options(
        geo_gpu_imu_replay
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2;/W4;/permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--prec-div=true;--prec-sqrt=true;-lineinfo;-diag-suppress=128;-Xcompiler=/O2;-Xcompiler=/W4>
    )
else()
    target_compile_options(
        geo_gpu_imu_replay
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--prec-div=true;--prec-sqrt=true;-lineinfo>
    )
endif()

set_target_properties(
    geo_gpu_imu_replay
    PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
