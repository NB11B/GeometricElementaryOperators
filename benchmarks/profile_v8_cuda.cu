#include "geo/batch_gp.h"
#include "geo/batch_gp_cuda.h"
#include "geo/hand_cuda_comparator.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

static void check_geo(geo_batch_gp_cuda_status_t status, const char *what) {
    if (status != GEO_BATCH_GP_CUDA_OK) {
        std::fprintf(stderr, "%s: %s\n", what, geo_batch_gp_cuda_status_string(status));
        std::exit(3);
    }
}

static double value_for(size_t index, int salt) {
    return (double)((int)((index * 23u + (size_t)salt * 11u) % 41u) - 20) / 31.0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "usage: %s DIMENSION BATCH BACKEND MODE ITERATIONS\n"
            "backend: reference|planned|hand\n"
            "mode: inference|training\n",
            argv[0]);
        return 1;
    }

    const int dimension = std::atoi(argv[1]);
    const size_t batch = (size_t)std::strtoull(argv[2], nullptr, 10);
    const char *backend = argv[3];
    const char *mode = argv[4];
    const int iterations = std::atoi(argv[5]);
    if (dimension < 2 || dimension > 6 || batch == 0 || iterations <= 0) return 1;
    if (std::strcmp(backend, "reference") != 0 &&
        std::strcmp(backend, "planned") != 0 &&
        std::strcmp(backend, "hand") != 0) return 1;
    if (std::strcmp(mode, "inference") != 0 && std::strcmp(mode, "training") != 0) return 1;

    int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 1, 1, 1, 1};
    geo_batch_gp_plan_t host_plan;
    if (geo_batch_gp_plan_init(&host_plan, (uint8_t)dimension, signature) != GEO_BATCH_GP_OK) return 4;
    geo_batch_gp_cuda_plan_t plan;
    check_geo(geo_batch_gp_cuda_plan_upload(&plan, &host_plan), "plan upload");

    const size_t blades = (size_t)1u << dimension;
    const size_t values = batch * blades;
    std::vector<double> inputs(values), targets(values), parameter(blades);
    for (size_t i = 0; i < values; ++i) {
        inputs[i] = value_for(i, dimension);
        targets[i] = value_for(i, dimension + 2);
    }
    for (size_t i = 0; i < blades; ++i) parameter[i] = value_for(i, dimension + 5);

    double *d_inputs = nullptr, *d_targets = nullptr, *d_parameter = nullptr;
    double *d_outputs = nullptr, *d_gradient = nullptr, *d_loss = nullptr;
    check_cuda(cudaMalloc((void **)&d_inputs, values * sizeof(double)), "malloc inputs");
    check_cuda(cudaMalloc((void **)&d_targets, values * sizeof(double)), "malloc targets");
    check_cuda(cudaMalloc((void **)&d_parameter, blades * sizeof(double)), "malloc parameter");
    check_cuda(cudaMalloc((void **)&d_outputs, values * sizeof(double)), "malloc outputs");
    check_cuda(cudaMalloc((void **)&d_gradient, blades * sizeof(double)), "malloc gradient");
    check_cuda(cudaMalloc((void **)&d_loss, sizeof(double)), "malloc loss");
    check_cuda(cudaMemcpy(d_inputs, inputs.data(), values * sizeof(double), cudaMemcpyHostToDevice), "copy inputs");
    check_cuda(cudaMemcpy(d_targets, targets.data(), values * sizeof(double), cudaMemcpyHostToDevice), "copy targets");
    check_cuda(cudaMemcpy(d_parameter, parameter.data(), blades * sizeof(double), cudaMemcpyHostToDevice), "copy parameter");

    for (int warmup = 0; warmup < 20; ++warmup) {
        if (std::strcmp(mode, "inference") == 0) {
            if (std::strcmp(backend, "reference") == 0) {
                check_geo(geo_batch_gp_cuda_reference_forward_f64(&plan, d_inputs, batch, d_parameter, 0, d_outputs, 0), "reference forward");
            } else if (std::strcmp(backend, "planned") == 0) {
                check_geo(geo_batch_gp_cuda_planned_forward_f64(&plan, d_inputs, batch, d_parameter, 0, d_outputs, 0), "planned forward");
            } else {
                check_geo(geo_hand_cuda_forward_f64(&plan, d_inputs, batch, d_parameter, 0, d_outputs, 0), "hand forward");
            }
        } else {
            check_cuda(cudaMemcpy(d_parameter, parameter.data(), blades * sizeof(double), cudaMemcpyHostToDevice), "reset parameter");
            if (std::strcmp(backend, "reference") == 0) {
                check_geo(geo_batch_gp_cuda_reference_mse_sgd_step_f64(&plan, d_inputs, d_targets, batch, 0.0001, 0,
                    d_parameter, d_outputs, d_gradient, d_loss, 0), "reference training");
            } else if (std::strcmp(backend, "planned") == 0) {
                check_geo(geo_batch_gp_cuda_mse_sgd_step_f64(&plan, d_inputs, d_targets, batch, 0.0001, 0,
                    d_parameter, d_outputs, d_gradient, d_loss, 0), "planned training");
            } else {
                check_geo(geo_hand_cuda_mse_sgd_step_f64(&plan, d_inputs, d_targets, batch, 0.0001, 0,
                    d_parameter, d_outputs, d_gradient, d_loss, 0), "hand training");
            }
        }
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    for (int i = 0; i < iterations; ++i) {
        if (std::strcmp(mode, "inference") == 0) {
            if (std::strcmp(backend, "reference") == 0) {
                check_geo(geo_batch_gp_cuda_reference_forward_f64(&plan, d_inputs, batch, d_parameter, 0, d_outputs, 0), "reference forward");
            } else if (std::strcmp(backend, "planned") == 0) {
                check_geo(geo_batch_gp_cuda_planned_forward_f64(&plan, d_inputs, batch, d_parameter, 0, d_outputs, 0), "planned forward");
            } else {
                check_geo(geo_hand_cuda_forward_f64(&plan, d_inputs, batch, d_parameter, 0, d_outputs, 0), "hand forward");
            }
        } else {
            check_cuda(cudaMemcpy(d_parameter, parameter.data(), blades * sizeof(double), cudaMemcpyHostToDevice), "reset parameter");
            if (std::strcmp(backend, "reference") == 0) {
                check_geo(geo_batch_gp_cuda_reference_mse_sgd_step_f64(&plan, d_inputs, d_targets, batch, 0.0001, 0,
                    d_parameter, d_outputs, d_gradient, d_loss, 0), "reference training");
            } else if (std::strcmp(backend, "planned") == 0) {
                check_geo(geo_batch_gp_cuda_mse_sgd_step_f64(&plan, d_inputs, d_targets, batch, 0.0001, 0,
                    d_parameter, d_outputs, d_gradient, d_loss, 0), "planned training");
            } else {
                check_geo(geo_hand_cuda_mse_sgd_step_f64(&plan, d_inputs, d_targets, batch, 0.0001, 0,
                    d_parameter, d_outputs, d_gradient, d_loss, 0), "hand training");
            }
        }
    }
    check_cuda(cudaDeviceSynchronize(), "profile sync");

    cudaFree(d_inputs); cudaFree(d_targets); cudaFree(d_parameter);
    cudaFree(d_outputs); cudaFree(d_gradient); cudaFree(d_loss);
    geo_batch_gp_cuda_plan_destroy(&plan);

    std::printf("GEO_V8_CUDA_PROFILE: PASS dimension=%d batch=%zu backend=%s mode=%s iterations=%d\n",
        dimension, batch, backend, mode, iterations);
    return 0;
}
