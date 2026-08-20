#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "geo/tensor_relational_cuda.h"

#define ASSERT_CUDA_OK(err, msg) \
    do { \
        if ((err) != cudaSuccess) { \
            fprintf(stderr, "CUDA ERROR: %s (%s) (%s:%d)\n", msg, cudaGetErrorString(err), __FILE__, __LINE__); \
            fflush(stderr); \
            return 1; \
        } \
    } while (0)

#define ASSERT_GEO_OK(st, msg) \
    do { \
        if ((st) != GEO_RELATIONAL_OK) { \
            fprintf(stderr, "GEO ERROR: %s (%d: %s) (%s:%d)\n", msg, (st), geo_relational_status_string(st), __FILE__, __LINE__); \
            fflush(stderr); \
            return 1; \
        } \
    } while (0)

static int is_close(geo_real_t cand, geo_real_t ref, double atol, double rtol) {
    double diff = fabs((double)cand - (double)ref);
    double threshold = atol + rtol * fabs((double)ref);
    return diff <= threshold;
}

static int run_physical_cuda_correctness_matrix(void) {
    int total_comparisons = 0;
    const int target_comparisons = 102400;

    size_t stream_options[] = { 2, 4, 8 };
    size_t feature_options[] = { 16, 64, 128, 512 };

    const double atol = 1e-5;
    const double rtol = 1e-4;

    for (int case_idx = 0; case_idx < 100; ++case_idx) {
        size_t P = stream_options[case_idx % 3];
        size_t D = feature_options[(case_idx / 3) % 4];
        size_t G = 4;
        size_t M = (case_idx % 2 == 0) ? 1 : G;

        geo_relational_shape shape;
        shape.groups = G;
        shape.streams = P;
        shape.features = D;
        shape.matrix_count = M;

        geo_relational_projection_options options;
        options.abi_version = GEO_RELATIONAL_ABI_VERSION;
        options.iterations = 20;
        options.epsilon = (geo_real_t)1e-2;
        options.mode = GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN;
        options.fail_on_nonfinite = 1;
        options.require_certificate = 0;
        options.reserved = 0;

        size_t logits_size = M * P * P * sizeof(geo_real_t);
        size_t rel_size = M * P * P * sizeof(geo_real_t);
        size_t state_size = G * P * D * sizeof(geo_real_t);
        size_t ws_elems = geo_relational_projection_workspace_elements(M, P, 20, 1);
        size_t ws_size = ws_elems * sizeof(geo_real_t);

        geo_real_t *h_logits = (geo_real_t *)malloc(logits_size);
        geo_real_t *h_rel_cpu = (geo_real_t *)malloc(rel_size);
        geo_real_t *h_rel_gpu = (geo_real_t *)malloc(rel_size);
        geo_real_t *h_state = (geo_real_t *)malloc(state_size);
        geo_real_t *h_out_cpu = (geo_real_t *)malloc(state_size);
        geo_real_t *h_out_gpu = (geo_real_t *)malloc(state_size);
        geo_real_t *h_ws_cpu = (geo_real_t *)malloc(ws_size);

        geo_real_t kappa = (geo_real_t)log((1.0 - (double)(P - 1) * 1e-3) / 1e-3);
        for (size_t m = 0; m < M; ++m) {
            for (size_t p = 0; p < P; ++p) {
                for (size_t q = 0; q < P; ++q) {
                    size_t idx = m * P * P + p * P + q;
                    geo_real_t base = (p == q) ? kappa : (geo_real_t)0.0;
                    geo_real_t var = (geo_real_t)(((idx + case_idx) % 5) - 2) * 0.01f;
                    h_logits[idx] = base + var;
                }
            }
        }

        geo_relational_certificate cert_cpu;
        geo_relational_status st_cpu = geo_relational_project_forward(
            h_logits, h_rel_cpu, h_ws_cpu, ws_elems, &shape, &options, &cert_cpu
        );
        ASSERT_GEO_OK(st_cpu, "Host CPU project forward");

        geo_real_t *d_logits, *d_rel, *d_ws, *d_state, *d_out;
        ASSERT_CUDA_OK(cudaMalloc(&d_logits, logits_size), "cudaMalloc logits");
        ASSERT_CUDA_OK(cudaMalloc(&d_rel, rel_size), "cudaMalloc rel");
        ASSERT_CUDA_OK(cudaMalloc(&d_ws, ws_size), "cudaMalloc ws");
        ASSERT_CUDA_OK(cudaMalloc(&d_state, state_size), "cudaMalloc state");
        ASSERT_CUDA_OK(cudaMalloc(&d_out, state_size), "cudaMalloc out");

        ASSERT_CUDA_OK(cudaMemcpy(d_logits, h_logits, logits_size, cudaMemcpyHostToDevice), "cudaMemcpy logits");

        geo_relational_certificate cert_gpu;
        geo_relational_status st_gpu = geo_relational_project_forward_cuda(
            d_logits, d_rel, d_ws, ws_elems, &shape, &options, &cert_gpu, NULL
        );
        ASSERT_GEO_OK(st_gpu, "Device CUDA project forward");
        ASSERT_CUDA_OK(cudaDeviceSynchronize(), "project sync");
        ASSERT_CUDA_OK(cudaMemcpy(h_rel_gpu, d_rel, rel_size, cudaMemcpyDeviceToHost), "cudaMemcpy rel gpu");

        for (size_t i = 0; i < M * P * P; ++i) {
            if (!is_close(h_rel_cpu[i], h_rel_gpu[i], atol, rtol)) {
                fprintf(stderr, "Parity failure in case %d at element %d: CPU=%f GPU=%f\n", case_idx, (int)i, (double)h_rel_cpu[i], (double)h_rel_gpu[i]);
                fflush(stderr);
                return 1;
            }
        }

        for (int s_idx = 0; s_idx < 1024; ++s_idx) {
            for (size_t j = 0; j < G * P * D; ++j) {
                h_state[j] = (geo_real_t)(((j + s_idx + case_idx) % 17) - 8) * 0.1f;
            }

            geo_relational_mix_forward(h_state, h_rel_cpu, h_out_cpu, &shape);

            ASSERT_CUDA_OK(cudaMemcpy(d_state, h_state, state_size, cudaMemcpyHostToDevice), "cudaMemcpy state");
            geo_relational_mix_forward_cuda(d_state, d_rel, d_out, &shape, NULL);
            ASSERT_CUDA_OK(cudaDeviceSynchronize(), "mix sync");
            ASSERT_CUDA_OK(cudaMemcpy(h_out_gpu, d_out, state_size, cudaMemcpyDeviceToHost), "cudaMemcpy out gpu");

            for (size_t element_idx = 0; element_idx < G * P * D; ++element_idx) {
                if (!is_close(h_out_gpu[element_idx], h_out_cpu[element_idx], atol, rtol)) {
                    double cpu_val = (double)h_out_cpu[element_idx];
                    double gpu_val = (double)h_out_gpu[element_idx];
                    fprintf(stderr, "Mix parity failure: case=%d s_idx=%d elem=%d CPU=%f GPU=%f\n",
                            case_idx, s_idx, (int)element_idx, cpu_val, gpu_val);
                    fflush(stderr);
                    return 1;
                }
            }
            total_comparisons++;
        }

        cudaFree(d_logits); cudaFree(d_rel); cudaFree(d_ws); cudaFree(d_state); cudaFree(d_out);
        free(h_logits); free(h_rel_cpu); free(h_rel_gpu); free(h_state); free(h_out_cpu); free(h_out_gpu); free(h_ws_cpu);
    }

    if (total_comparisons < target_comparisons) {
        fprintf(stderr, "Total comparisons %d less than target %d\n", total_comparisons, target_comparisons);
        fflush(stderr);
        return 1;
    }

    printf("GEO_RELATIONAL_CUDA_CORRECTNESS: PASS comparisons=%d fallback=NONE\n", total_comparisons);
    fflush(stdout);
    return 0;
}

int main(void) {
    printf("Running Physical CUDA Relational Operator Correctness Suite...\n");
    fflush(stdout);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "CUDA device unavailable\n");
        fflush(stderr);
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    fflush(stdout);

    if (run_physical_cuda_correctness_matrix() != 0) {
        return 1;
    }

    return 0;
}
