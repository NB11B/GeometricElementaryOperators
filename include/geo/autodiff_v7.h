#ifndef GEO_AUTODIFF_V7_H
#define GEO_AUTODIFF_V7_H

#include "geo/operator_kernel.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_V7_ABI_VERSION 0x00070000u
#define GEO_V7_MAX_NODES 128u
#define GEO_V7_INVALID_NODE UINT16_MAX

typedef uint16_t geo_v7_node_id_t;

typedef enum {
    GEO_V7_OK = 0,
    GEO_V7_INVALID_ARGUMENT = 1,
    GEO_V7_UNSUPPORTED_DIMENSION = 2,
    GEO_V7_CAPACITY_EXCEEDED = 3,
    GEO_V7_BAD_PROGRAM = 4,
    GEO_V7_BAD_NODE = 5,
    GEO_V7_NOT_COMPILED = 6,
    GEO_V7_FORWARD_REQUIRED = 7,
    GEO_V7_NON_SCALAR_LOSS = 8,
    GEO_V7_NUMERIC_FAILURE = 9
} geo_v7_status_t;

typedef enum {
    GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN = 1
} geo_v7_pairing_t;

typedef enum {
    GEO_V7_NODE_INPUT = 1,
    GEO_V7_NODE_PARAMETER = 2,
    GEO_V7_NODE_CONSTANT = 3,
    GEO_V7_NODE_ADD = 4,
    GEO_V7_NODE_SCALE = 5,
    GEO_V7_NODE_GEOMETRIC_PRODUCT = 6,
    GEO_V7_NODE_REVERSE = 7,
    GEO_V7_NODE_GRADE_PROJECT = 8,
    GEO_V7_NODE_SQUARED_NORM = 9
} geo_v7_node_kind_t;

typedef struct {
    uint8_t kind;
    uint8_t requires_grad;
    uint8_t grade;
    uint8_t reserved;
    geo_v7_node_id_t left;
    geo_v7_node_id_t right;
    double scalar;
    geo_operator_mv_f64_t value;
    geo_operator_mv_f64_t cotangent;
    double first_moment[GEO_OPERATOR_MAX_BLADES];
    double second_moment[GEO_OPERATOR_MAX_BLADES];
} geo_v7_node_t;

typedef struct {
    uint32_t abi_version;
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    uint8_t pairing;
    uint8_t compiled;
    uint8_t forward_valid;
    uint8_t reserved;
    geo_v7_node_id_t node_count;
    geo_v7_node_id_t loss_node;
    uint64_t optimizer_step;
    geo_v7_node_t nodes[GEO_V7_MAX_NODES];
} geo_v7_program_t;

uint32_t geo_v7_abi_version(void);
size_t geo_v7_program_bytes(void);

geo_v7_status_t geo_v7_program_init(
    geo_v7_program_t *program,
    uint8_t dimension,
    const int8_t *signature,
    geo_v7_pairing_t pairing
);

geo_v7_status_t geo_v7_add_input(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *initial_value,
    int requires_grad,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_parameter(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *initial_value,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_constant(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *value,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_add(
    geo_v7_program_t *program,
    geo_v7_node_id_t left,
    geo_v7_node_id_t right,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_scale(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    double scalar,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_geometric_product(
    geo_v7_program_t *program,
    geo_v7_node_id_t left,
    geo_v7_node_id_t right,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_reverse(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_grade_project(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    uint8_t grade,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_add_squared_norm(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    geo_v7_node_id_t *node_id
);

geo_v7_status_t geo_v7_compile(
    geo_v7_program_t *program,
    geo_v7_node_id_t loss_node
);

geo_v7_status_t geo_v7_set_value(
    geo_v7_program_t *program,
    geo_v7_node_id_t node_id,
    const geo_operator_mv_f64_t *value
);

geo_v7_status_t geo_v7_forward(geo_v7_program_t *program);
geo_v7_status_t geo_v7_zero_grad(geo_v7_program_t *program);
geo_v7_status_t geo_v7_backward(geo_v7_program_t *program);

geo_v7_status_t geo_v7_sgd_step(
    geo_v7_program_t *program,
    double learning_rate
);

geo_v7_status_t geo_v7_adam_step(
    geo_v7_program_t *program,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
);

const geo_operator_mv_f64_t *geo_v7_value(
    const geo_v7_program_t *program,
    geo_v7_node_id_t node_id
);

const geo_operator_mv_f64_t *geo_v7_gradient(
    const geo_v7_program_t *program,
    geo_v7_node_id_t node_id
);

#ifdef __cplusplus
}
#endif

#endif
