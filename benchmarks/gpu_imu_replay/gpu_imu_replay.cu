#include "geo_imu_generated_schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSampleRateHz = 200U;
constexpr uint32_t kSampleCount = 12000U;
constexpr float kGravityMps2 = 9.80665f;
constexpr float kDt = 1.0f / static_cast<float>(kSampleRateHz);
constexpr uint32_t kThreadsPerBlock = 256U;
constexpr uint32_t kDefaultBatch = 4096U;
constexpr uint32_t kDefaultRuns = 30U;
constexpr uint32_t kDefaultWarmup = 5U;
constexpr uint32_t kFNVOffset = UINT32_C(2166136261);
constexpr uint32_t kFNVPrime = UINT32_C(16777619);

struct ImuSample {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
};

struct Output {
    float qw;
    float qx;
    float qy;
    float qz;
    float gravity_x;
    float gravity_y;
    float gravity_z;
    float confidence;
    uint8_t motion_state;
};

struct FilterState {
    float qw;
    float qx;
    float qy;
    float qz;
    float integral_x;
    float integral_y;
    float integral_z;
};

static_assert(sizeof(FilterState) == 28U, "unexpected filter state size");

struct Options {
    int device = 0;
    uint32_t batch = kDefaultBatch;
    uint32_t runs = kDefaultRuns;
    uint32_t warmup = kDefaultWarmup;
    std::string csv_path;
};

struct Validation {
    double mean_error_deg = 0.0;
    double p95_error_deg = 0.0;
    double max_error_deg = 0.0;
    uint32_t nan_count = 0U;
    uint32_t trace_hash = kFNVOffset;
};

struct TimedRun {
    float kernel_us = 0.0f;
    uint32_t batch_hash = kFNVOffset;
};

bool cuda_ok(cudaError_t status, const char *stage) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "CUDA failure during %s: %s\n", stage,
                 cudaGetErrorString(status));
    return false;
}

uint32_t parse_u32(const char *text, const char *name) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        std::fprintf(stderr, "Invalid %s\n", name);
        std::exit(EXIT_FAILURE);
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value > UINT32_MAX) {
        std::fprintf(stderr, "Invalid %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint32_t>(value);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            std::printf(
                "Usage: %s [--device N] [--batch N] [--runs N] "
                "[--warmup N] [--csv PATH]\n",
                argv[0]
            );
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value after %s\n", argument.c_str());
            std::exit(EXIT_FAILURE);
        }
        const char *value = argv[++index];
        if (argument == "--device") {
            options.device = static_cast<int>(parse_u32(value, "device"));
        } else if (argument == "--batch") {
            options.batch = parse_u32(value, "batch");
            if (options.batch == 0U) {
                std::fprintf(stderr, "Batch must be nonzero\n");
                std::exit(EXIT_FAILURE);
            }
        } else if (argument == "--runs") {
            options.runs = parse_u32(value, "runs");
            if (options.runs == 0U) {
                std::fprintf(stderr, "Runs must be nonzero\n");
                std::exit(EXIT_FAILURE);
            }
        } else if (argument == "--warmup") {
            options.warmup = parse_u32(value, "warmup");
        } else if (argument == "--csv") {
            options.csv_path = value;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argument.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    return options;
}

float replay_noise(uint32_t value) {
    value ^= value << 13U;
    value ^= value >> 17U;
    value ^= value << 5U;
    return (static_cast<float>(value & UINT32_C(0xffff)) / 32767.5f) - 1.0f;
}

void euler_to_quaternion(float roll, float pitch, float yaw, Output *reference) {
    const float cr = std::cos(0.5f * roll);
    const float sr = std::sin(0.5f * roll);
    const float cp = std::cos(0.5f * pitch);
    const float sp = std::sin(0.5f * pitch);
    const float cy = std::cos(0.5f * yaw);
    const float sy = std::sin(0.5f * yaw);
    reference->qw = cr * cp * cy + sr * sp * sy;
    reference->qx = sr * cp * cy - cr * sp * sy;
    reference->qy = cr * sp * cy + sr * cp * sy;
    reference->qz = cr * cp * sy - sr * sp * cy;
}

void replay_trace_sample(uint32_t index, ImuSample *sample, Output *reference) {
    const float t = static_cast<float>(index) / static_cast<float>(kSampleRateHz);
    const float roll = 0.45f * std::sin(0.63f * t) +
        0.10f * std::sin(2.7f * t);
    const float pitch = 0.35f * std::sin(0.41f * t + 0.7f);
    const float yaw = 0.55f * std::sin(0.23f * t) + 0.012f * t;
    const float roll_rate = 0.45f * 0.63f * std::cos(0.63f * t) +
        0.10f * 2.7f * std::cos(2.7f * t);
    const float pitch_rate = 0.35f * 0.41f * std::cos(0.41f * t + 0.7f);
    const float yaw_rate = 0.55f * 0.23f * std::cos(0.23f * t) + 0.012f;

    euler_to_quaternion(roll, pitch, yaw, reference);

    const float qw = reference->qw;
    const float qx = reference->qx;
    const float qy = reference->qy;
    const float qz = reference->qz;
    const float gx_body = 2.0f * (qx * qz - qw * qy);
    const float gy_body = 2.0f * (qw * qx + qy * qz);
    const float gz_body = qw * qw - qx * qx - qy * qy + qz * qz;
    const float disturbance = ((index / 1200U) % 3U == 1U)
        ? 0.45f * std::sin(7.0f * t)
        : 0.0f;

    sample->ax = kGravityMps2 * gx_body + disturbance +
        0.025f * replay_noise(index * 17U + 1U);
    sample->ay = kGravityMps2 * gy_body + 0.5f * disturbance +
        0.025f * replay_noise(index * 17U + 2U);
    sample->az = kGravityMps2 * gz_body +
        0.025f * replay_noise(index * 17U + 3U);
    sample->gx = roll_rate + 0.0015f * replay_noise(index * 17U + 4U);
    sample->gy = pitch_rate + 0.0015f * replay_noise(index * 17U + 5U);
    sample->gz = yaw_rate + 0.0015f * replay_noise(index * 17U + 6U);

    reference->gravity_x = gx_body;
    reference->gravity_y = gy_body;
    reference->gravity_z = gz_body;
    reference->confidence = disturbance == 0.0f ? 1.0f : 0.75f;
    reference->motion_state =
        (std::fabs(roll_rate) + std::fabs(pitch_rate) + std::fabs(yaw_rate) < 0.08f)
        ? 0U
        : 1U;
}

__device__ __forceinline__ float device_clamp(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__device__ __forceinline__ void device_normalize(FilterState *state) {
    const float norm = sqrtf(
        state->qw * state->qw + state->qx * state->qx +
        state->qy * state->qy + state->qz * state->qz
    );
    if (norm > 1.0e-12f) {
        state->qw /= norm;
        state->qx /= norm;
        state->qy /= norm;
        state->qz /= norm;
    } else {
        state->qw = 1.0f;
        state->qx = 0.0f;
        state->qy = 0.0f;
        state->qz = 0.0f;
    }
}

__device__ __forceinline__ void conventional_gravity(
    float qw,
    float qx,
    float qy,
    float qz,
    float *gx,
    float *gy,
    float *gz
) {
    *gx = 2.0f * (qx * qz - qw * qy);
    *gy = 2.0f * (qw * qx + qy * qz);
    *gz = qw * qw - qx * qx - qy * qy + qz * qz;
}

__device__ __forceinline__ void conventional_q_times_vector(
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
) {
    *dw = -qx * wx - qy * wy - qz * wz;
    *dx = qw * wx + qy * wz - qz * wy;
    *dy = qw * wy - qx * wz + qz * wx;
    *dz = qw * wz + qx * wy - qy * wx;
}

template <bool Generated>
__device__ __forceinline__ Output filter_step(
    FilterState *state,
    const ImuSample sample
) {
    constexpr float kp = 2.2f;
    constexpr float ki = 0.035f;
    float ax = sample.ax;
    float ay = sample.ay;
    float az = sample.az;
    float wx = sample.gx;
    float wy = sample.gy;
    float wz = sample.gz;
    const float accel_norm = sqrtf(ax * ax + ay * ay + az * az);
    float confidence = 0.0f;

    if (accel_norm > 1.0e-6f) {
        float vx;
        float vy;
        float vz;
        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;
        confidence = device_clamp(
            1.0f - fabsf(accel_norm - kGravityMps2) / 2.0f,
            0.0f,
            1.0f
        );

        if constexpr (Generated) {
            geo_gpu_generated_float_gravity(
                state->qw, state->qx, state->qy, state->qz,
                &vx, &vy, &vz
            );
        } else {
            conventional_gravity(
                state->qw, state->qx, state->qy, state->qz,
                &vx, &vy, &vz
            );
        }

        const float ex = ay * vz - az * vy;
        const float ey = az * vx - ax * vz;
        const float ez = ax * vy - ay * vx;
        state->integral_x += ki * confidence * ex * kDt;
        state->integral_y += ki * confidence * ey * kDt;
        state->integral_z += ki * confidence * ez * kDt;
        wx += kp * confidence * ex + state->integral_x;
        wy += kp * confidence * ey + state->integral_y;
        wz += kp * confidence * ez + state->integral_z;
    }

    const float qw = state->qw;
    const float qx = state->qx;
    const float qy = state->qy;
    const float qz = state->qz;
    float dw;
    float dx;
    float dy;
    float dz;
    if constexpr (Generated) {
        geo_gpu_generated_float_q_times_vector_quaternion(
            qw, qx, qy, qz, wx, wy, wz, &dw, &dx, &dy, &dz
        );
    } else {
        conventional_q_times_vector(
            qw, qx, qy, qz, wx, wy, wz, &dw, &dx, &dy, &dz
        );
    }

    constexpr float half_dt = 0.5f * kDt;
    state->qw += dw * half_dt;
    state->qx += dx * half_dt;
    state->qy += dy * half_dt;
    state->qz += dz * half_dt;
    device_normalize(state);

    Output output{};
    output.qw = state->qw;
    output.qx = state->qx;
    output.qy = state->qy;
    output.qz = state->qz;
    if constexpr (Generated) {
        geo_gpu_generated_float_gravity(
            state->qw, state->qx, state->qy, state->qz,
            &output.gravity_x, &output.gravity_y, &output.gravity_z
        );
    } else {
        conventional_gravity(
            state->qw, state->qx, state->qy, state->qz,
            &output.gravity_x, &output.gravity_y, &output.gravity_z
        );
    }
    output.confidence = confidence;
    output.motion_state =
        (fabsf(wx) + fabsf(wy) + fabsf(wz) < 0.08f) ? 0U : 1U;
    return output;
}

template <bool Generated>
__global__ void resident_replay_kernel(
    const ImuSample *samples,
    uint32_t sample_count,
    uint32_t trajectories,
    Output *final_output,
    Output *capture_trace
) {
    const uint32_t trajectory = blockIdx.x * blockDim.x + threadIdx.x;
    if (trajectory >= trajectories) {
        return;
    }

    FilterState state{};
    state.qw = 1.0f;
    Output output{};
    for (uint32_t sample_index = 0U;
         sample_index < sample_count;
         ++sample_index) {
        output = filter_step<Generated>(&state, samples[sample_index]);
        if (capture_trace != nullptr && trajectory == 0U) {
            capture_trace[sample_index] = output;
        }
    }
    final_output[trajectory] = output;
}

uint32_t hash_bytes(uint32_t hash, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    for (std::size_t index = 0U; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFNVPrime;
    }
    return hash;
}

uint32_t hash_output(uint32_t hash, const Output &output) {
    hash = hash_bytes(hash, &output.qw, sizeof(output.qw));
    hash = hash_bytes(hash, &output.qx, sizeof(output.qx));
    hash = hash_bytes(hash, &output.qy, sizeof(output.qy));
    hash = hash_bytes(hash, &output.qz, sizeof(output.qz));
    hash = hash_bytes(hash, &output.gravity_x, sizeof(output.gravity_x));
    hash = hash_bytes(hash, &output.gravity_y, sizeof(output.gravity_y));
    hash = hash_bytes(hash, &output.gravity_z, sizeof(output.gravity_z));
    hash = hash_bytes(hash, &output.confidence, sizeof(output.confidence));
    return hash_bytes(hash, &output.motion_state, sizeof(output.motion_state));
}

uint32_t hash_outputs(const std::vector<Output> &outputs) {
    uint32_t hash = kFNVOffset;
    for (const Output &output : outputs) {
        hash = hash_output(hash, output);
    }
    return hash;
}

double quaternion_error_deg(const Output &actual, const Output &reference) {
    if (!std::isfinite(actual.qw) || !std::isfinite(actual.qx) ||
        !std::isfinite(actual.qy) || !std::isfinite(actual.qz)) {
        return 180.0;
    }
    double dot = std::fabs(
        static_cast<double>(actual.qw) * reference.qw +
        static_cast<double>(actual.qx) * reference.qx +
        static_cast<double>(actual.qy) * reference.qy +
        static_cast<double>(actual.qz) * reference.qz
    );
    dot = std::max(0.0, std::min(1.0, dot));
    return 2.0 * std::acos(dot) * 57.29577951308232;
}

template <bool Generated>
bool validate_implementation(
    const ImuSample *device_samples,
    const std::vector<Output> &references,
    Validation *validation
) {
    Output *device_final = nullptr;
    Output *device_trace = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_final), sizeof(Output)),
                 "validation final allocation") ||
        !cuda_ok(cudaMalloc(
            reinterpret_cast<void **>(&device_trace),
            sizeof(Output) * kSampleCount),
            "validation trace allocation")) {
        if (device_trace != nullptr) cudaFree(device_trace);
        if (device_final != nullptr) cudaFree(device_final);
        return false;
    }

    resident_replay_kernel<Generated><<<1, 1>>>(
        device_samples, kSampleCount, 1U, device_final, device_trace
    );
    if (!cuda_ok(cudaGetLastError(), "validation kernel launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "validation kernel synchronization")) {
        cudaFree(device_trace);
        cudaFree(device_final);
        return false;
    }

    std::vector<Output> trace(kSampleCount);
    const bool copied = cuda_ok(cudaMemcpy(
        trace.data(), device_trace, sizeof(Output) * kSampleCount,
        cudaMemcpyDeviceToHost), "validation trace download");
    cudaFree(device_trace);
    cudaFree(device_final);
    if (!copied) {
        return false;
    }

    std::vector<double> errors;
    errors.reserve(kSampleCount);
    double sum = 0.0;
    validation->nan_count = 0U;
    validation->trace_hash = kFNVOffset;
    for (uint32_t index = 0U; index < kSampleCount; ++index) {
        const Output &output = trace[index];
        const double error = quaternion_error_deg(output, references[index]);
        errors.push_back(error);
        sum += error;
        if (!std::isfinite(output.qw) || !std::isfinite(output.qx) ||
            !std::isfinite(output.qy) || !std::isfinite(output.qz)) {
            ++validation->nan_count;
        }
        validation->trace_hash = hash_output(validation->trace_hash, output);
    }
    std::sort(errors.begin(), errors.end());
    validation->mean_error_deg = sum / static_cast<double>(kSampleCount);
    validation->p95_error_deg = errors[(kSampleCount * 95U) / 100U];
    validation->max_error_deg = errors.back();
    return true;
}

template <bool Generated>
bool time_one(
    const ImuSample *device_samples,
    uint32_t batch,
    Output *device_final,
    cudaEvent_t start,
    cudaEvent_t stop,
    TimedRun *result
) {
    const uint32_t blocks = (batch + kThreadsPerBlock - 1U) / kThreadsPerBlock;
    if (!cuda_ok(cudaEventRecord(start), "event record start")) {
        return false;
    }
    resident_replay_kernel<Generated><<<blocks, kThreadsPerBlock>>>(
        device_samples, kSampleCount, batch, device_final, nullptr
    );
    if (!cuda_ok(cudaGetLastError(), "timed kernel launch") ||
        !cuda_ok(cudaEventRecord(stop), "event record stop") ||
        !cuda_ok(cudaEventSynchronize(stop), "event synchronize stop")) {
        return false;
    }
    float elapsed_ms = 0.0f;
    if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms, start, stop),
                 "event elapsed time")) {
        return false;
    }
    result->kernel_us = elapsed_ms * 1000.0f;

    std::vector<Output> final_output(batch);
    if (!cuda_ok(cudaMemcpy(
        final_output.data(), device_final, sizeof(Output) * batch,
        cudaMemcpyDeviceToHost), "timed final download")) {
        return false;
    }
    result->batch_hash = hash_outputs(final_output);
    return true;
}

void print_csv_row(
    FILE *stream,
    const char *implementation,
    uint32_t run,
    uint32_t batch,
    const TimedRun &timing,
    const Validation &validation,
    std::size_t device_bytes
) {
    const double seconds = static_cast<double>(timing.kernel_us) * 1.0e-6;
    const double work_items = static_cast<double>(batch) * kSampleCount;
    const double ns_per_sample_trajectory =
        static_cast<double>(timing.kernel_us) * 1000.0 / work_items;
    const double trajectories_per_second = static_cast<double>(batch) / seconds;
    const double samples_per_second = work_items / seconds;

    std::fprintf(
        stream,
        "CSV_GPU,%s,resident_replay,%u,%u,%zu,%.6f,%.9f,%.3f,%.3f,"
        "%.6f,%.6f,%.6f,%u,%08x,%08x,%zu\n",
        implementation,
        run,
        batch,
        sizeof(FilterState),
        static_cast<double>(timing.kernel_us),
        ns_per_sample_trajectory,
        trajectories_per_second,
        samples_per_second,
        validation.mean_error_deg,
        validation.p95_error_deg,
        validation.max_error_deg,
        validation.nan_count,
        validation.trace_hash,
        timing.batch_hash,
        device_bytes
    );
}

}  // namespace

int main(int argc, char **argv) {
    const Options options = parse_options(argc, argv);
    if (!cuda_ok(cudaSetDevice(options.device), "device selection")) {
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, options.device),
                 "device properties")) {
        return EXIT_FAILURE;
    }
    int runtime_version = 0;
    int driver_version = 0;
    if (!cuda_ok(cudaRuntimeGetVersion(&runtime_version), "runtime version") ||
        !cuda_ok(cudaDriverGetVersion(&driver_version), "driver version")) {
        return EXIT_FAILURE;
    }

    std::vector<ImuSample> samples(kSampleCount);
    std::vector<Output> references(kSampleCount);
    for (uint32_t index = 0U; index < kSampleCount; ++index) {
        replay_trace_sample(index, &samples[index], &references[index]);
    }

    ImuSample *device_samples = nullptr;
    Output *device_final = nullptr;
    const std::size_t sample_bytes = sizeof(ImuSample) * samples.size();
    const std::size_t final_bytes = sizeof(Output) * options.batch;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_samples), sample_bytes),
                 "sample allocation") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&device_final), final_bytes),
                 "final output allocation") ||
        !cuda_ok(cudaMemcpy(
            device_samples, samples.data(), sample_bytes,
            cudaMemcpyHostToDevice), "sample upload")) {
        if (device_final != nullptr) cudaFree(device_final);
        if (device_samples != nullptr) cudaFree(device_samples);
        return EXIT_FAILURE;
    }

    Validation generated_validation;
    Validation conventional_validation;
    if (!validate_implementation<true>(
            device_samples, references, &generated_validation) ||
        !validate_implementation<false>(
            device_samples, references, &conventional_validation)) {
        cudaFree(device_final);
        cudaFree(device_samples);
        return EXIT_FAILURE;
    }

    std::printf(
        "GEO_GPU_IMU_REPLAY,device=%d,name=%s,cc=%d.%d,runtime=%d,driver=%d,"
        "samples=%u,runs=%u,warmup=%u,batch=%u,fmad=off\n",
        options.device,
        properties.name,
        properties.major,
        properties.minor,
        runtime_version,
        driver_version,
        kSampleCount,
        options.runs,
        options.warmup,
        options.batch
    );
    std::printf(
        "GEO_GPU_PARITY,a1_hash=%08x,c_hash=%08x,status=%s\n",
        generated_validation.trace_hash,
        conventional_validation.trace_hash,
        generated_validation.trace_hash == conventional_validation.trace_hash
            ? "pass"
            : "fail"
    );
    std::printf(
        "CSV_GPU,implementation,mode,run,batch,state_bytes,kernel_us,"
        "ns_per_sample_trajectory,trajectories_per_second,samples_per_second,"
        "mean_error_deg,p95_error_deg,max_error_deg,nan_count,trace_hash,"
        "batch_hash,device_bytes\n"
    );

    FILE *csv = nullptr;
    if (!options.csv_path.empty()) {
        csv = std::fopen(options.csv_path.c_str(), "wb");
        if (csv == nullptr) {
            std::fprintf(stderr, "Unable to open CSV path: %s\n",
                         options.csv_path.c_str());
            cudaFree(device_final);
            cudaFree(device_samples);
            return EXIT_FAILURE;
        }
        std::fprintf(
            csv,
            "CSV_GPU,implementation,mode,run,batch,state_bytes,kernel_us,"
            "ns_per_sample_trajectory,trajectories_per_second,samples_per_second,"
            "mean_error_deg,p95_error_deg,max_error_deg,nan_count,trace_hash,"
            "batch_hash,device_bytes\n"
        );
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "event create start") ||
        !cuda_ok(cudaEventCreate(&stop), "event create stop")) {
        if (csv != nullptr) std::fclose(csv);
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        cudaFree(device_final);
        cudaFree(device_samples);
        return EXIT_FAILURE;
    }

    TimedRun ignored;
    for (uint32_t warmup = 0U; warmup < options.warmup; ++warmup) {
        const bool generated_first = (warmup % 2U) == 0U;
        if (generated_first) {
            if (!time_one<true>(device_samples, options.batch, device_final,
                                start, stop, &ignored) ||
                !time_one<false>(device_samples, options.batch, device_final,
                                 start, stop, &ignored)) {
                return EXIT_FAILURE;
            }
        } else {
            if (!time_one<false>(device_samples, options.batch, device_final,
                                 start, stop, &ignored) ||
                !time_one<true>(device_samples, options.batch, device_final,
                                start, stop, &ignored)) {
                return EXIT_FAILURE;
            }
        }
    }

    const std::size_t device_bytes = sample_bytes + final_bytes;
    for (uint32_t run = 0U; run < options.runs; ++run) {
        TimedRun generated_timing;
        TimedRun conventional_timing;
        const bool generated_first = (run % 2U) == 0U;
        bool ok = true;
        if (generated_first) {
            ok = time_one<true>(device_samples, options.batch, device_final,
                                start, stop, &generated_timing) &&
                time_one<false>(device_samples, options.batch, device_final,
                                start, stop, &conventional_timing);
        } else {
            ok = time_one<false>(device_samples, options.batch, device_final,
                                 start, stop, &conventional_timing) &&
                time_one<true>(device_samples, options.batch, device_final,
                               start, stop, &generated_timing);
        }
        if (!ok) {
            if (csv != nullptr) std::fclose(csv);
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            cudaFree(device_final);
            cudaFree(device_samples);
            return EXIT_FAILURE;
        }

        print_csv_row(
            stdout,
            "A1_geo_float_generated_gpu",
            run,
            options.batch,
            generated_timing,
            generated_validation,
            device_bytes
        );
        print_csv_row(
            stdout,
            "C_conventional_quaternion_gpu",
            run,
            options.batch,
            conventional_timing,
            conventional_validation,
            device_bytes
        );
        if (csv != nullptr) {
            print_csv_row(
                csv,
                "A1_geo_float_generated_gpu",
                run,
                options.batch,
                generated_timing,
                generated_validation,
                device_bytes
            );
            print_csv_row(
                csv,
                "C_conventional_quaternion_gpu",
                run,
                options.batch,
                conventional_timing,
                conventional_validation,
                device_bytes
            );
        }
    }

    if (csv != nullptr) {
        std::fclose(csv);
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_final);
    cudaFree(device_samples);

    const bool parity =
        generated_validation.trace_hash == conventional_validation.trace_hash &&
        generated_validation.nan_count == 0U &&
        conventional_validation.nan_count == 0U;
    std::printf("GEO_GPU_IMU_REPLAY,status=%s\n", parity ? "complete" : "fail");
    return parity ? EXIT_SUCCESS : EXIT_FAILURE;
}
