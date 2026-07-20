#include "geo/batch_gp.h"
#include "geo/batch_gp_cuda.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

static void die_geo(geo_batch_gp_cuda_status_t status, const char *what) {
    if (status != GEO_BATCH_GP_CUDA_OK) {
        std::fprintf(stderr, "%s: %s\n", what, geo_batch_gp_cuda_status_string(status));
        std::exit(3);
    }
}

static double deterministic_value(size_t index, int salt) {
    return (double)((int)((index * 23u + (size_t)salt * 11u) % 41u) - 20) / 31.0;
}

struct DeviceBuffers {
    double *inputs = nullptr;
    double *targets = nullptr;
    double *parameter = nullptr;
    double *outputs = nullptr;
    double *gradient = nullptr;
    double *loss = nullptr;
};

static DeviceBuffers allocate_buffers(size_t values, size_t blades) {
    DeviceBuffers b;
    die_cuda(cudaMalloc((void **)&b.inputs, values * sizeof(double)), "cudaMalloc inputs");
    die_cuda(cudaMalloc((void **)&b.targets, values * sizeof(double)), "cudaMalloc targets");
    die_cuda(cudaMalloc((void **)&b.parameter, blades * sizeof(double)), "cudaMalloc parameter");
    die_cuda(cudaMalloc((void **)&b.outputs, values * sizeof(double)), "cudaMalloc outputs");
    die_cuda(cudaMalloc((void **)&b.gradient, blades * sizeof(double)), "cudaMalloc gradient");
    die_cuda(cudaMalloc((void **)&b.loss, sizeof(double)), "cudaMalloc loss");
    return b;
}

static void free_buffers(DeviceBuffers &b) {
    cudaFree(b.inputs); cudaFree(b.targets); cudaFree(b.parameter);
    cudaFree(b.outputs); cudaFree(b.gradient); cudaFree(b.loss);
    b = DeviceBuffers{};
}

static void upload(DeviceBuffers &b, const std::vector<double> &inputs,
                   const std::vector<double> &targets, const std::vector<double> &parameter) {
    die_cuda(cudaMemcpy(b.inputs, inputs.data(), inputs.size() * sizeof(double), cudaMemcpyHostToDevice), "copy inputs");
    die_cuda(cudaMemcpy(b.targets, targets.data(), targets.size() * sizeof(double), cudaMemcpyHostToDevice), "copy targets");
    die_cuda(cudaMemcpy(b.parameter, parameter.data(), parameter.size() * sizeof(double), cudaMemcpyHostToDevice), "copy parameter");
}

static void launch(const char *backend, const char *mode, const geo_batch_gp_cuda_plan_t &plan,
                   DeviceBuffers &b, size_t batch) {
    if (std::strcmp(mode, "inference") == 0) {
        if (std::strcmp(backend, "geo_cuda_reference") == 0) {
            die_geo(geo_batch_gp_cuda_reference_forward_f64(&plan, b.inputs, batch, b.parameter, 0, b.outputs, 0), "reference forward");
        } else {
            die_geo(geo_batch_gp_cuda_planned_forward_f64(&plan, b.inputs, batch, b.parameter, 0, b.outputs, 0), "planned forward");
        }
    } else {
        die_geo(geo_batch_gp_cuda_mse_sgd_step_f64(&plan, b.inputs, b.targets, batch, 0.0001, 0,
            b.parameter, b.outputs, b.gradient, b.loss, 0), "training step");
    }
}

static double checksum(DeviceBuffers &b, size_t values, size_t blades, const char *mode) {
    std::vector<double> host(std::strcmp(mode, "inference") == 0 ? values : blades);
    const double *source = std::strcmp(mode, "inference") == 0 ? b.outputs : b.parameter;
    die_cuda(cudaMemcpy(host.data(), source, host.size() * sizeof(double), cudaMemcpyDeviceToHost), "checksum copy");
    double sum = 0.0;
    for (size_t i = 0; i < host.size(); ++i) sum += host[i] * (double)(i + 1u);
    return sum;
}

static double resident_seconds(const char *backend, const char *mode,
                               const geo_batch_gp_cuda_plan_t &plan, DeviceBuffers &b,
                               size_t batch, int iterations) {
    for (int i = 0; i < 10; ++i) launch(backend, mode, plan, b, batch);
    die_cuda(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start, stop;
    die_cuda(cudaEventCreate(&start), "event start");
    die_cuda(cudaEventCreate(&stop), "event stop");
    die_cuda(cudaEventRecord(start), "record start");
    for (int i = 0; i < iterations; ++i) launch(backend, mode, plan, b, batch);
    die_cuda(cudaEventRecord(stop), "record stop");
    die_cuda(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    die_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return (double)ms / 1000.0;
}

static double transfer_seconds(const char *backend, const char *mode,
                               const geo_batch_gp_cuda_plan_t &plan, DeviceBuffers &b,
                               const std::vector<double> &inputs, const std::vector<double> &targets,
                               const std::vector<double> &parameter, size_t batch, int iterations) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        upload(b, inputs, targets, parameter);
        launch(backend, mode, plan, b, batch);
    }
    die_cuda(cudaDeviceSynchronize(), "transfer sync");
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static double end_to_end_seconds(const char *backend, const char *mode,
                                 const geo_batch_gp_cuda_plan_t &plan,
                                 const std::vector<double> &inputs, const std::vector<double> &targets,
                                 const std::vector<double> &parameter, size_t batch, int iterations) {
    const size_t values = inputs.size();
    const size_t blades = parameter.size();
    std::vector<double> result(std::strcmp(mode, "inference") == 0 ? values : blades);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        DeviceBuffers b = allocate_buffers(values, blades);
        upload(b, inputs, targets, parameter);
        launch(backend, mode, plan, b, batch);
        const double *source = std::strcmp(mode, "inference") == 0 ? b.outputs : b.parameter;
        die_cuda(cudaMemcpy(result.data(), source, result.size() * sizeof(double), cudaMemcpyDeviceToHost), "result copy");
        free_buffers(b);
    }
    die_cuda(cudaDeviceSynchronize(), "end-to-end sync");
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s OUTPUT.csv [iterations]\n", argv[0]);
        return 1;
    }
    const int iterations = argc >= 3 ? std::atoi(argv[2]) : 50;
    if (iterations <= 0) return 1;
    std::FILE *out = std::fopen(argv[1], "w");
    if (out == nullptr) return 1;
    std::fprintf(out, "backend,mode,timing_class,dimension,batch,samples_per_second,checksum\n");

    int device = 0;
    cudaDeviceProp props{};
    die_cuda(cudaGetDeviceProperties(&props, device), "device properties");
    std::printf("GEO_CUDA_DEVICE=%s capability=%d.%d\n", props.name, props.major, props.minor);

    const size_t batches[] = {1u, 16u, 64u, 256u, 1024u};
    const char *backends[] = {"geo_cuda_reference", "geo_cuda_planned"};
    const char *modes[] = {"inference", "training"};
    const char *timings[] = {"resident", "transfer_compute", "end_to_end"};

    for (uint8_t dimension = 2u; dimension <= 6u; ++dimension) {
        int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 1, 1, 1, 1};
        geo_batch_gp_plan_t host_plan;
        if (geo_batch_gp_plan_init(&host_plan, dimension, signature) != GEO_BATCH_GP_OK) return 4;
        geo_batch_gp_cuda_plan_t plan;
        die_geo(geo_batch_gp_cuda_plan_upload(&plan, &host_plan), "plan upload");
        const size_t blades = (size_t)1u << dimension;

        for (size_t batch : batches) {
            const size_t values = batch * blades;
            std::vector<double> inputs(values), targets(values), parameter(blades);
            for (size_t i = 0; i < values; ++i) {
                inputs[i] = deterministic_value(i, dimension);
                targets[i] = deterministic_value(i, dimension + 2);
            }
            for (size_t i = 0; i < blades; ++i) parameter[i] = deterministic_value(i, dimension + 5);
            DeviceBuffers b = allocate_buffers(values, blades);
            upload(b, inputs, targets, parameter);

            for (const char *backend : backends) {
                for (const char *mode : modes) {
                    for (const char *timing : timings) {
                        upload(b, inputs, targets, parameter);
                        double seconds = 0.0;
                        if (std::strcmp(timing, "resident") == 0) {
                            seconds = resident_seconds(backend, mode, plan, b, batch, iterations);
                        } else if (std::strcmp(timing, "transfer_compute") == 0) {
                            seconds = transfer_seconds(backend, mode, plan, b, inputs, targets, parameter, batch, iterations);
                        } else {
                            seconds = end_to_end_seconds(backend, mode, plan, inputs, targets, parameter, batch, iterations);
                        }
                        const double rate = (double)(batch * (size_t)iterations) / seconds;
                        const double sum = checksum(b, values, blades, mode);
                        std::fprintf(out, "%s,%s,%s,%u,%zu,%.17g,%.17g\n",
                            backend, mode, timing, (unsigned)dimension, batch, rate, sum);
                    }
                }
            }
            free_buffers(b);
        }
        geo_batch_gp_cuda_plan_destroy(&plan);
    }
    std::fclose(out);
    std::printf("GEO_V8_CUDA_BENCHMARK: PASS output=%s iterations=%d\n", argv[1], iterations);
    return 0;
}
