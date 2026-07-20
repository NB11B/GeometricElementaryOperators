#include "geo/batch_gp.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_vector(double *values, size_t count, uint64_t seed) {
    size_t i;
    for (i = 0u; i < count; ++i) {
        uint64_t x = seed + UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(i + 1u);
        x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
        x ^= x >> 31;
        values[i] = ((double)(int64_t)(x % 2001u) - 1000.0) / 1000.0;
    }
}

static int run_case(uint8_t n, size_t batch, size_t iterations, FILE *out) {
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 1, 1, 1, 1};
    geo_batch_gp_plan_t plan;
    const size_t blades = (size_t)1u << n;
    const size_t values = batch * blades;
    double *inputs = (double *)malloc(values * sizeof(double));
    double *targets = (double *)malloc(values * sizeof(double));
    double *outputs = (double *)malloc(values * sizeof(double));
    double truth[GEO_OPERATOR_MAX_BLADES];
    double parameter[GEO_OPERATOR_MAX_BLADES];
    size_t iteration;
    size_t sample;
    double start;
    double elapsed;
    double checksum = 0.0;

    if (inputs == NULL || targets == NULL || outputs == NULL) return 0;
    if (geo_batch_gp_plan_init(&plan, n, signature) != GEO_BATCH_GP_OK) return 0;
    fill_vector(truth, blades, 3u);
    memcpy(parameter, truth, blades * sizeof(double));

    for (iteration = 0u; iteration < 32u; ++iteration) {
        for (sample = 0u; sample < batch; ++sample) {
            fill_vector(inputs + sample * blades, blades, iteration * batch + sample + 11u);
        }
        if (geo_batch_gp_right_forward_f64(&plan, inputs, batch, parameter, outputs) != GEO_BATCH_GP_OK) return 0;
    }

    start = now_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        size_t index;
        for (sample = 0u; sample < batch; ++sample) {
            fill_vector(inputs + sample * blades, blades, iteration * batch + sample + 101u);
        }
        if (geo_batch_gp_right_forward_f64(&plan, inputs, batch, parameter, outputs) != GEO_BATCH_GP_OK) return 0;
        for (index = 0u; index < values; ++index) checksum += 0.5 * outputs[index] * outputs[index];
    }
    elapsed = now_seconds() - start;
    fprintf(out, "geo_v8_batch,inference,%u,%zu,%.3f,%.9g\n", (unsigned)n, batch,
        (double)(iterations * batch) / elapsed, checksum);

    checksum = 0.0;
    memcpy(parameter, truth, blades * sizeof(double));
    start = now_seconds();
    for (iteration = 0u; iteration < iterations; ++iteration) {
        double loss;
        for (sample = 0u; sample < batch; ++sample) {
            fill_vector(inputs + sample * blades, blades, iteration * batch + sample + 1001u);
        }
        if (geo_batch_gp_right_forward_f64(&plan, inputs, batch, truth, targets) != GEO_BATCH_GP_OK) return 0;
        if (geo_batch_gp_right_mse_sgd_step_f64(
                &plan,
                inputs,
                targets,
                batch,
                1e-6,
                parameter,
                &loss
            ) != GEO_BATCH_GP_OK) return 0;
        checksum += loss;
    }
    elapsed = now_seconds() - start;
    fprintf(out, "geo_v8_batch,train_step,%u,%zu,%.3f,%.9g\n", (unsigned)n, batch,
        (double)(iterations * batch) / elapsed, checksum);

    free(inputs);
    free(targets);
    free(outputs);
    return 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "geo_v8_batch_bench.csv";
    FILE *out = fopen(path, "wb");
    const size_t batches[] = {1u, 16u, 64u};
    uint8_t n;
    size_t batch_index;
    if (out == NULL) return EXIT_FAILURE;
    fputs("backend,mode,dimension,batch,samples_per_second,checksum\n", out);
    for (n = 2u; n <= 6u; ++n) {
        for (batch_index = 0u; batch_index < sizeof(batches) / sizeof(batches[0]); ++batch_index) {
            const size_t iterations = n >= 6u ? 50u : 200u;
            if (!run_case(n, batches[batch_index], iterations, out)) {
                fclose(out);
                return EXIT_FAILURE;
            }
        }
    }
    fclose(out);
    return EXIT_SUCCESS;
}
