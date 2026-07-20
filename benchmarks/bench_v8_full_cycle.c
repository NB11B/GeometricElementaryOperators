#include "geo/full_cycle_v8.h"

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

static void init_mv(geo_operator_mv_f64_t *v, uint8_t n, const int8_t *sig) {
    memset(v, 0, sizeof(*v));
    v->dimension = n;
    memcpy(v->signature, sig, n);
}

static void fill_mv(geo_operator_mv_f64_t *v, uint8_t n, const int8_t *sig, uint64_t seed) {
    size_t i;
    size_t count = (size_t)1u << n;
    init_mv(v, n, sig);
    for (i = 0; i < count; ++i) {
        uint64_t x = seed + UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(i + 1u);
        x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
        x ^= x >> 31;
        v->coefficients[i] = ((double)(int64_t)(x % 2001u) - 1000.0) / 1000.0;
    }
}

static int build_program(
    geo_v8_program_t *p,
    uint8_t n,
    const int8_t *sig,
    geo_v8_node_id_t *input,
    geo_v8_node_id_t *target,
    geo_v8_node_id_t *parameter,
    geo_v8_node_id_t *loss
) {
    geo_operator_mv_f64_t zero;
    geo_v8_node_id_t prediction, neg_target, residual;
    init_mv(&zero, n, sig);
    if (geo_v8_program_init(p, n, sig, GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN, 8u) != GEO_V8_OK) return 0;
    if (geo_v8_add_leaf(p, GEO_V8_NODE_INPUT, "x", &zero, 0, GEO_V8_CONSTRAINT_NONE, input) != GEO_V8_OK) return 0;
    if (geo_v8_add_leaf(p, GEO_V8_NODE_TARGET, "y", &zero, 0, GEO_V8_CONSTRAINT_NONE, target) != GEO_V8_OK) return 0;
    if (geo_v8_add_leaf(p, GEO_V8_NODE_PARAMETER, "w", &zero, 1, GEO_V8_CONSTRAINT_NONE, parameter) != GEO_V8_OK) return 0;
    if (geo_v8_add_binary(p, GEO_V8_NODE_GEOMETRIC_PRODUCT, "pred", *input, *parameter, &prediction) != GEO_V8_OK) return 0;
    if (geo_v8_add_unary(p, GEO_V8_NODE_SCALE, "neg_y", *target, -1.0, 0u, &neg_target) != GEO_V8_OK) return 0;
    if (geo_v8_add_binary(p, GEO_V8_NODE_ADD, "residual", prediction, neg_target, &residual) != GEO_V8_OK) return 0;
    if (geo_v8_add_unary(p, GEO_V8_NODE_SQUARED_NORM, "loss", residual, 0.0, 0u, loss) != GEO_V8_OK) return 0;
    return geo_v8_compile(p, *loss) == GEO_V8_OK;
}

static int run_case(uint8_t n, size_t batch, size_t iterations, FILE *out) {
    int8_t sig[GEO_OPERATOR_MAX_DIMENSION] = {1,1,1,1,1,1};
    geo_v8_program_t p;
    geo_v8_node_id_t input, target, parameter, loss;
    geo_operator_mv_f64_t x, y, w;
    size_t i, j;
    double start, elapsed, checksum = 0.0;

    if (!build_program(&p, n, sig, &input, &target, &parameter, &loss)) return 0;
    fill_mv(&w, n, sig, 3u);
    if (geo_v8_set_value(&p, parameter, &w) != GEO_V8_OK) return 0;

    for (i = 0; i < 32u; ++i) {
        fill_mv(&x, n, sig, i + 11u);
        if (geo_operator_gp_f64(&x, &w, &y) != GEO_OPERATOR_OK) return 0;
        if (geo_v8_set_value(&p, input, &x) != GEO_V8_OK || geo_v8_set_value(&p, target, &y) != GEO_V8_OK) return 0;
        if (geo_v8_forward(&p) != GEO_V8_OK) return 0;
    }

    start = now_seconds();
    for (i = 0; i < iterations; ++i) {
        for (j = 0; j < batch; ++j) {
            fill_mv(&x, n, sig, (uint64_t)(i * batch + j + 101u));
            if (geo_v8_set_value(&p, input, &x) != GEO_V8_OK) return 0;
            if (geo_v8_forward(&p) != GEO_V8_OK) return 0;
            checksum += geo_v8_value(&p, loss)->coefficients[0];
        }
    }
    elapsed = now_seconds() - start;
    fprintf(out, "geo_v8,inference,%u,%zu,%.3f,%.9g\n", (unsigned)n, batch,
        (double)(iterations * batch) / elapsed, checksum);

    start = now_seconds();
    checksum = 0.0;
    for (i = 0; i < iterations; ++i) {
        if (geo_v8_zero_grad(&p) != GEO_V8_OK) return 0;
        for (j = 0; j < batch; ++j) {
            fill_mv(&x, n, sig, (uint64_t)(i * batch + j + 1001u));
            if (geo_operator_gp_f64(&x, &w, &y) != GEO_OPERATOR_OK) return 0;
            if (geo_v8_set_value(&p, input, &x) != GEO_V8_OK || geo_v8_set_value(&p, target, &y) != GEO_V8_OK) return 0;
            if (geo_v8_forward(&p) != GEO_V8_OK || geo_v8_backward(&p, j != 0u) != GEO_V8_OK) return 0;
            checksum += geo_v8_value(&p, loss)->coefficients[0];
        }
        if (geo_v8_optimizer_step(&p, GEO_V8_OPTIMIZER_SGD, 1e-6, 0.9, 0.999, 1e-8, 1.0 / (double)batch) != GEO_V8_OK) return 0;
    }
    elapsed = now_seconds() - start;
    fprintf(out, "geo_v8,train_step,%u,%zu,%.3f,%.9g\n", (unsigned)n, batch,
        (double)(iterations * batch) / elapsed, checksum);
    geo_v8_program_free(&p);
    return 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "geo_v8_bench.csv";
    FILE *out = fopen(path, "wb");
    uint8_t n;
    const size_t batches[] = {1u, 16u, 64u};
    size_t b;
    if (out == NULL) return EXIT_FAILURE;
    fputs("backend,mode,dimension,batch,samples_per_second,checksum\n", out);
    for (n = 2u; n <= 6u; ++n) {
        for (b = 0u; b < sizeof(batches)/sizeof(batches[0]); ++b) {
            size_t iterations = n >= 6u ? 50u : 200u;
            if (!run_case(n, batches[b], iterations, out)) {
                fclose(out);
                return EXIT_FAILURE;
            }
        }
    }
    fclose(out);
    return EXIT_SUCCESS;
}
