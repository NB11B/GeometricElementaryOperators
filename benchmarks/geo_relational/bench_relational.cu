#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "geo/tensor_relational_cuda.h"

int main(void) {
    printf("Running CUDA Relational Operator Benchmark Sweep...\n");

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        printf("CUDA device unavailable for benchmarking.\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Benchmark GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);

    size_t group_sweep[] = { 128, 512, 2048 };
    size_t stream_sweep[] = { 2, 4, 8 };
    size_t feature_sweep[] = { 64, 128, 512, 960 };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("%-8s %-8s %-10s %-8s %-12s %-12s\n", "Groups", "Streams", "Features", "M_count", "Proj (ms)", "Mix (ms)");

    for (size_t gi = 0; gi < 3; ++gi) {
        for (size_t si = 0; si < 3; ++si) {
            for (size_t fi = 0; fi < 4; ++fi) {
                for (size_t M = 1; M <= group_sweep[gi]; M = (M == 1) ? group_sweep[gi] : group_sweep[gi] + 1) {
                    size_t G = group_sweep[gi];
                    size_t P = stream_sweep[si];
                    size_t D = feature_sweep[fi];

                    geo_relational_shape shape = { G, P, D, M };
                    geo_relational_projection_options options = {
                        GEO_RELATIONAL_ABI_VERSION, 20, (geo_real_t)1e-7,
                        GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN, 1, 0, 0
                    };

                    size_t logits_bytes = M * P * P * sizeof(geo_real_t);
                    size_t rel_bytes = M * P * P * sizeof(geo_real_t);
                    size_t state_bytes = G * P * D * sizeof(geo_real_t);
                    size_t ws_elems = geo_relational_projection_workspace_elements(M, P, 20, 1);

                    geo_real_t *d_logits, *d_rel, *d_ws, *d_state, *d_out;
                    cudaMalloc(&d_logits, logits_bytes);
                    cudaMalloc(&d_rel, rel_bytes);
                    cudaMalloc(&d_ws, ws_elems * sizeof(geo_real_t));
                    cudaMalloc(&d_state, state_bytes);
                    cudaMalloc(&d_out, state_bytes);

                    cudaMemset(d_logits, 0, logits_bytes);
                    cudaMemset(d_state, 0, state_bytes);

                    // Warmup
                    geo_relational_project_forward_cuda(d_logits, d_rel, d_ws, ws_elems, &shape, &options, NULL, NULL);
                    geo_relational_mix_forward_cuda(d_state, d_rel, d_out, &shape, NULL);
                    cudaDeviceSynchronize();

                    // Measure Projection
                    cudaEventRecord(start);
                    int iters = 50;
                    for (int it = 0; it < iters; ++it) {
                        geo_relational_project_forward_cuda(d_logits, d_rel, d_ws, ws_elems, &shape, &options, NULL, NULL);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float proj_ms = 0.0f;
                    cudaEventElapsedTime(&proj_ms, start, stop);
                    proj_ms /= iters;

                    // Measure Mix
                    cudaEventRecord(start);
                    for (int it = 0; it < iters; ++it) {
                        geo_relational_mix_forward_cuda(d_state, d_rel, d_out, &shape, NULL);
                    }
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float mix_ms = 0.0f;
                    cudaEventElapsedTime(&mix_ms, start, stop);
                    mix_ms /= iters;

                    printf("%-8zu %-8zu %-10zu %-8zu %-12.4f %-12.4f\n", G, P, D, M, proj_ms, mix_ms);

                    cudaFree(d_logits); cudaFree(d_rel); cudaFree(d_ws); cudaFree(d_state); cudaFree(d_out);
                }
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("CUDA Relational Operator Benchmark Complete.\n");
    return 0;
}
