cmake_minimum_required(VERSION 3.24)

project(geo_gpu_imu_replay LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
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

target_compile_options(
    geo_gpu_imu_replay
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--prec-div=true;--prec-sqrt=true;-lineinfo>
)

set_target_properties(
    geo_gpu_imu_replay
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
