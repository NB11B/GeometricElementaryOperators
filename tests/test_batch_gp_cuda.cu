#include "geo/batch_gp.h"
#include "geo/batch_gp_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

static void check_status(geo_batch_gp_cuda_status_t status, const char *what) {
    if (status != GEO_BATCH_GP_CUDA_OK) {
        std::fprintf(stderr, "%s: %s\n", what, geo_batch_gp_cuda_status_string(status));
        std::exit(3);
    }
}

static double value_for(size_t i, int salt) {
    const int centered = (int)((i * 17u + (size_t)salt * 13u) % 29u) - 14;
    return (double)centered / 19.0;
}

static void compare(const std::vector<double> &expected, const std::vector<double> &actual, const char *label) {
    double max_abs = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(expected[i] - actual[i]);
        if (error > max_abs) max_abs = error;
        const double tolerance = 1.0e-10 + 1.0e-10 * std::fabs(expected[i]);
        if (!(error <= tolerance)) {
            std::fprintf(stderr, "%s mismatch index=%zu expected=%.17g actual=%.17g error=%.17g\n",
                label, i, expected[i], actual[i], error);
            std::exit(4);
        }
    }
    std::printf("%s max_abs=%.3e\n", label, max_abs);
}

int main(void) {
    const size_t batches[] = {1u, 16u, 64u, 256u};
    size_t cases = 0u;
    for (uint8_t dimension = 2u; dimension <= 6u; ++dimension) {
        const size_t blades = (size_t)1u << dimension;
        for (uint8_t negative = 0u; negative <= dimension; ++negative) {
            int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {0};
            for (uint8_t axis = 0u; axis < dimension; ++axis) {
                signature[axis] = axis < (uint8_t)(dimension - negative) ? 1 : -1;
            }
            geo_batch_gp_plan_t host_plan;
            if (geo_batch_gp_plan_init(&host_plan, dimension, signature) != GEO_BATCH_GP_OK) return 5;
            geo_batch_gp_cuda_plan_t device_plan;
            check_status(geo_batch_gp_cuda_plan_upload(&device_plan, &host_plan), "plan upload");

            for (int parameter_on_left = 0; parameter_on_left <= 1; ++parameter_on_left) {
                for (size_t batch : batches) {
                    const size_t count = batch * blades;
                    std::vector<double> inputs(count), targets(count), cotangents(count), parameter(blades);
                    for (size_t i = 0; i < count; ++i) {
                        inputs[i] = value_for(i, 1 + parameter_on_left);
                        targets[i] = value_for(i, 3 + negative);
                        cotangents[i] = value_for(i, 5 + dimension);
                    }
                    for (size_t i = 0; i < blades; ++i) parameter[i] = value_for(i, 7 + negative);

                    std::vector<double> cpu_forward(count), cpu_gradient(blades), cpu_parameter = parameter;
                    double cpu_loss = 0.0;
                    geo_batch_gp_status_t cpu_status = parameter_on_left ?
                        geo_batch_gp_left_forward_f64(&host_plan, parameter.data(), inputs.data(), batch, cpu_forward.data()) :
                        geo_batch_gp_right_forward_f64(&host_plan, inputs.data(), batch, parameter.data(), cpu_forward.data());
                    if (cpu_status != GEO_BATCH_GP_OK) return 6;
                    cpu_status = parameter_on_left ?
                        geo_batch_gp_left_vjp_parameter_f64(&host_plan, inputs.data(), cotangents.data(), batch, cpu_gradient.data()) :
                        geo_batch_gp_right_vjp_parameter_f64(&host_plan, inputs.data(), cotangents.data(), batch, cpu_gradient.data());
                    if (cpu_status != GEO_BATCH_GP_OK) return 7;
                    cpu_status = parameter_on_left ?
                        geo_batch_gp_left_mse_sgd_step_f64(&host_plan, inputs.data(), targets.data(), batch, 0.001, cpu_parameter.data(), &cpu_loss) :
                        geo_batch_gp_right_mse_sgd_step_f64(&host_plan, inputs.data(), targets.data(), batch, 0.001, cpu_parameter.data(), &cpu_loss);
                    if (cpu_status != GEO_BATCH_GP_OK) return 8;

                    double *d_inputs = nullptr, *d_targets = nullptr, *d_cotangents = nullptr;
                    double *d_parameter = nullptr, *d_outputs = nullptr, *d_gradient = nullptr, *d_loss = nullptr;
                    check_cuda(cudaMalloc((void **)&d_inputs, count * sizeof(double)), "malloc inputs");
                    check_cuda(cudaMalloc((void **)&d_targets, count * sizeof(double)), "malloc targets");
                    check_cuda(cudaMalloc((void **)&d_cotangents, count * sizeof(double)), "malloc cotangents");
                    check_cuda(cudaMalloc((void **)&d_parameter, blades * sizeof(double)), "malloc parameter");
                    check_cuda(cudaMalloc((void **)&d_outputs, count * sizeof(double)), "malloc outputs");
                    check_cuda(cudaMalloc((void **)&d_gradient, blades * sizeof(double)), "malloc gradient");
                    check_cuda(cudaMalloc((void **)&d_loss, sizeof(double)), "malloc loss");
                    check_cuda(cudaMemcpy(d_inputs, inputs.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy inputs");
                    check_cuda(cudaMemcpy(d_targets, targets.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy targets");
                    check_cuda(cudaMemcpy(d_cotangents, cotangents.data(), count * sizeof(double), cudaMemcpyHostToDevice), "copy cotangents");
                    check_cuda(cudaMemcpy(d_parameter, parameter.data(), blades * sizeof(double), cudaMemcpyHostToDevice), "copy parameter");

                    check_status(geo_batch_gp_cuda_reference_forward_f64(&device_plan, d_inputs, batch, d_parameter,
                        parameter_on_left, d_outputs, 0), "reference forward");
                    check_cuda(cudaDeviceSynchronize(), "sync reference");
                    std::vector<double> gpu_reference(count);
                    check_cuda(cudaMemcpy(gpu_reference.data(), d_outputs, count * sizeof(double), cudaMemcpyDeviceToHost), "copy reference");
                    compare(cpu_forward, gpu_reference, "cuda_reference_forward");

                    check_status(geo_batch_gp_cuda_planned_forward_f64(&device_plan, d_inputs, batch, d_parameter,
                        parameter_on_left, d_outputs, 0), "planned forward");
                    check_cuda(cudaDeviceSynchronize(), "sync planned");
                    std::vector<double> gpu_forward(count);
                    check_cuda(cudaMemcpy(gpu_forward.data(), d_outputs, count * sizeof(double), cudaMemcpyDeviceToHost), "copy planned");
                    compare(cpu_forward, gpu_forward, "cuda_planned_forward");

                    check_status(geo_batch_gp_cuda_parameter_vjp_f64(&device_plan, d_inputs, d_cotangents, batch,
                        parameter_on_left, d_gradient, 0), "parameter vjp");
                    check_cuda(cudaDeviceSynchronize(), "sync vjp");
                    std::vector<double> gpu_gradient(blades);
                    check_cuda(cudaMemcpy(gpu_gradient.data(), d_gradient, blades * sizeof(double), cudaMemcpyDeviceToHost), "copy gradient");
                    compare(cpu_gradient, gpu_gradient, "cuda_parameter_vjp");

                    check_cuda(cudaMemcpy(d_parameter, parameter.data(), blades * sizeof(double), cudaMemcpyHostToDevice), "reset parameter");
                    check_status(geo_batch_gp_cuda_mse_sgd_step_f64(&device_plan, d_inputs, d_targets, batch, 0.001,
                        parameter_on_left, d_parameter, d_outputs, d_gradient, d_loss, 0), "mse sgd");
                    check_cuda(cudaDeviceSynchronize(), "sync sgd");
                    std::vector<double> gpu_parameter(blades);
                    double gpu_loss_sum = 0.0;
                    check_cuda(cudaMemcpy(gpu_parameter.data(), d_parameter, blades * sizeof(double), cudaMemcpyDeviceToHost), "copy updated parameter");
                    check_cuda(cudaMemcpy(&gpu_loss_sum, d_loss, sizeof(double), cudaMemcpyDeviceToHost), "copy loss");
                    compare(cpu_parameter, gpu_parameter, "cuda_mse_sgd_parameter");
                    const double gpu_mean_loss = gpu_loss_sum / (double)batch;
                    if (std::fabs(cpu_loss - gpu_mean_loss) > 1.0e-9 * (1.0 + std::fabs(cpu_loss))) {
                        std::fprintf(stderr, "loss mismatch cpu=%.17g gpu=%.17g\n", cpu_loss, gpu_mean_loss);
                        return 9;
                    }

                    cudaFree(d_inputs); cudaFree(d_targets); cudaFree(d_cotangents); cudaFree(d_parameter);
                    cudaFree(d_outputs); cudaFree(d_gradient); cudaFree(d_loss);
                    ++cases;
                }
            }
            geo_batch_gp_cuda_plan_destroy(&device_plan);
        }
    }
    std::printf("GEO_V8_CUDA_CORRECTNESS: PASS cases=%zu dimensions=2-6 batches=1,16,64,256 sides=2\n", cases);
    return 0;
}
