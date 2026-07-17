/*
 * Generated file. Do not edit by hand.
 * Source: benchmarks/esp32_imu_baseline/schedules/imu_orientation_sparse_v1.json
 * Schedule: imu_orientation_sparse_v1
 */
#ifndef GEO_CUDA_GENERATED_IMU_ORIENTATION_SPARSE_V1_CUH
#define GEO_CUDA_GENERATED_IMU_ORIENTATION_SPARSE_V1_CUH

#include <stdint.h>

#if !defined(__CUDACC__)
#error "This generated header requires a CUDA compiler"
#endif

#define GEO_IMU_CUDA_INLINE __device__ __forceinline__

GEO_IMU_CUDA_INLINE void geo_gpu_generated_float_gravity(
    float qw,
    float qx,
    float qy,
    float qz,
    float *gx,
    float *gy,
    float *gz
)
{
    *gx = 2.0f * (qx * qz - qw * qy);
    *gy = 2.0f * (qw * qx + qy * qz);
    *gz = qw * qw - qx * qx - qy * qy + qz * qz;
}

GEO_IMU_CUDA_INLINE void geo_gpu_generated_q32_gravity(
    int32_t qw,
    int32_t qx,
    int32_t qy,
    int32_t qz,
    int64_t *gx,
    int64_t *gy,
    int64_t *gz
)
{
    *gx = INT64_C(2) * ((int64_t)qx * (int64_t)qz - (int64_t)qw * (int64_t)qy);
    *gy = INT64_C(2) * ((int64_t)qw * (int64_t)qx + (int64_t)qy * (int64_t)qz);
    *gz = (int64_t)qw * (int64_t)qw - (int64_t)qx * (int64_t)qx - (int64_t)qy * (int64_t)qy + (int64_t)qz * (int64_t)qz;
}

GEO_IMU_CUDA_INLINE void geo_gpu_generated_float_q_times_vector_quaternion(
    float qw,
    float qx,
    float qy,
    float qz,
    float wx,
    float wy,
    float wz,
    float *dw,
    float *dx,
    float *dy,
    float *dz
)
{
    *dw = -qx * wx - qy * wy - qz * wz;
    *dx = qw * wx + qy * wz - qz * wy;
    *dy = qw * wy - qx * wz + qz * wx;
    *dz = qw * wz + qx * wy - qy * wx;
}

GEO_IMU_CUDA_INLINE void geo_gpu_generated_q32_q_times_vector_quaternion(
    int32_t qw,
    int32_t qx,
    int32_t qy,
    int32_t qz,
    int32_t wx,
    int32_t wy,
    int32_t wz,
    int64_t *dw,
    int64_t *dx,
    int64_t *dy,
    int64_t *dz
)
{
    *dw = -(int64_t)qx * (int64_t)wx - (int64_t)qy * (int64_t)wy - (int64_t)qz * (int64_t)wz;
    *dx = (int64_t)qw * (int64_t)wx + (int64_t)qy * (int64_t)wz - (int64_t)qz * (int64_t)wy;
    *dy = (int64_t)qw * (int64_t)wy - (int64_t)qx * (int64_t)wz + (int64_t)qz * (int64_t)wx;
    *dz = (int64_t)qw * (int64_t)wz + (int64_t)qx * (int64_t)wy - (int64_t)qy * (int64_t)wx;
}

#undef GEO_IMU_CUDA_INLINE

#endif /* GEO_CUDA_GENERATED_IMU_ORIENTATION_SPARSE_V1_CUH */
