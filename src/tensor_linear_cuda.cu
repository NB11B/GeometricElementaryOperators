#include "geo/tensor_linear_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>

static int geo_cuda_mul_overflows(size_t a, size_t b) {
    return a != 0u && b > SIZE_MAX / a;
}

static geo_tensor_status geo_cuda_validate(
    const void *x,
    const void *weight,
    const void *output,
    geo_tensor_linear_shape shape
) {
    if (x == NULL || weight == NULL || output == NULL ||
        shape.rows == 0u || shape.in_features == 0u || shape.out_features == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    if (geo_cuda_mul_overflows(shape.rows, shape.in_features) ||
        geo_cuda_mul_overflows(shape.rows, shape.out_features) ||
        geo_cuda_mul_overflows(shape.out_features, shape.in_features)) {
        return GEO_TENSOR_OVERFLOW;
    }
    return GEO_TENSOR_OK;
}

static cublasHandle_t get_cublas_handle() {
    thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

extern "C" geo_tensor_status geo_tensor_linear_cuda_forward(
    const geo_real_t *x,
    const geo_real_t *weight,
    geo_real_t *y,
    const geo_tensor_linear_shape *shape,
    void *stream_ptr
) {
    if (shape == NULL) return GEO_TENSOR_INVALID_ARGUMENT;
    geo_tensor_status status = geo_cuda_validate(x, weight, y, *shape);
    if (status != GEO_TENSOR_OK) {
        return status;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const int m = static_cast<int>(shape->rows);
    const int k = static_cast<int>(shape->in_features);
    const int n = static_cast<int>(shape->out_features);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Y = X * W^T
    // Col-major equivalent: Y^T (N x M) = W (N x K) * X^T (K x M)
    // op(W) = CUBLAS_OP_T -> W^T (K x N), lda = K
    // op(X) = CUBLAS_OP_N -> X (K x M), ldb = K
    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        weight, k,
        x, k,
        &beta,
        y, n
    );

    return stat == CUBLAS_STATUS_SUCCESS ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

extern "C" geo_tensor_status geo_tensor_linear_cuda_vjp(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    const geo_tensor_linear_shape *shape,
    void *stream_ptr
) {
    if (shape == NULL) return GEO_TENSOR_INVALID_ARGUMENT;
    geo_tensor_status status = geo_cuda_validate(x, weight, grad_y, *shape);
    if (status != GEO_TENSOR_OK || grad_x == NULL || grad_weight == NULL) {
        return status == GEO_TENSOR_OK ? GEO_TENSOR_INVALID_ARGUMENT : status;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const int m = static_cast<int>(shape->rows);
    const int k = static_cast<int>(shape->in_features);
    const int n = static_cast<int>(shape->out_features);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 1. grad_x = grad_y * weight
    // Col-major equivalent: grad_x^T (K x M) = weight^T (K x N) * grad_y^T (N x M)
    // op(weight) = CUBLAS_OP_N -> W (K x N), lda = K
    // op(grad_y) = CUBLAS_OP_N -> dY (N x M), ldb = N
    cublasStatus_t stat1 = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        weight, k,
        grad_y, n,
        &beta,
        grad_x, k
    );

    if (stat1 != CUBLAS_STATUS_SUCCESS) return GEO_TENSOR_CUDA_ERROR;

    // 2. grad_weight = grad_y^T * x
    // Col-major equivalent: grad_weight^T (K x N) = x^T (K x M) * grad_y (M x N)
    // op(x) = CUBLAS_OP_N -> X (K x M), lda = K
    // op(grad_y) = CUBLAS_OP_T -> dY^T (N x M), ldb = N
    cublasStatus_t stat2 = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        k, n, m,
        &alpha,
        x, k,
        grad_y, n,
        &beta,
        grad_weight, k
    );

    return stat2 == CUBLAS_STATUS_SUCCESS ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}

extern "C" geo_tensor_status geo_tensor_linear_cuda_vjp_decomposed_profile(
    const geo_real_t *x,
    const geo_real_t *weight,
    const geo_real_t *grad_y,
    geo_real_t *grad_x,
    geo_real_t *grad_weight,
    const geo_tensor_linear_shape *shape,
    float *dx_ms,
    float *dw_ms,
    void *stream_ptr
) {
    if (shape == NULL || dx_ms == NULL || dw_ms == NULL) return GEO_TENSOR_INVALID_ARGUMENT;
    geo_tensor_status status = geo_cuda_validate(x, weight, grad_y, *shape);
    if (status != GEO_TENSOR_OK || grad_x == NULL || grad_weight == NULL) {
        return status == GEO_TENSOR_OK ? GEO_TENSOR_INVALID_ARGUMENT : status;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    const int m = static_cast<int>(shape->rows);
    const int k = static_cast<int>(shape->in_features);
    const int n = static_cast<int>(shape->out_features);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t ev0, ev1, ev2;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventCreate(&ev2);

    cudaEventRecord(ev0, stream);
    cublasStatus_t stat1 = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        weight, k,
        grad_y, n,
        &beta,
        grad_x, k
    );
    cudaEventRecord(ev1, stream);

    cublasStatus_t stat2 = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        k, n, m,
        &alpha,
        x, k,
        grad_y, n,
        &beta,
        grad_weight, k
    );
    cudaEventRecord(ev2, stream);

    cudaStreamSynchronize(stream);

    cudaEventElapsedTime(dx_ms, ev0, ev1);
    cudaEventElapsedTime(dw_ms, ev1, ev2);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);

    return (stat1 == CUBLAS_STATUS_SUCCESS && stat2 == CUBLAS_STATUS_SUCCESS) ? GEO_TENSOR_OK : GEO_TENSOR_CUDA_ERROR;
}
