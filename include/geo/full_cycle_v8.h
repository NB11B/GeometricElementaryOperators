#ifndef GEO_FULL_CYCLE_V8_H
#define GEO_FULL_CYCLE_V8_H

#include "geo/operator_kernel.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEO_V8_ABI_VERSION 0x00080000u
#define GEO_V8_INVALID_NODE UINT32_MAX
#define GEO_V8_NAME_MAX 64u

typedef uint32_t geo_v8_node_id_t;

typedef enum {
    GEO_V8_OK = 0,
    GEO_V8_INVALID_ARGUMENT = 1,
    GEO_V8_UNSUPPORTED_DIMENSION = 2,
    GEO_V8_ALLOCATION_FAILURE = 3,
    GEO_V8_BAD_PROGRAM = 4,
    GEO_V8_BAD_NODE = 5,
    GEO_V8_DUPLICATE_NAME = 6,
    GEO_V8_NOT_COMPILED = 7,
    GEO_V8_FORWARD_REQUIRED = 8,
    GEO_V8_BACKWARD_REQUIRED = 9,
    GEO_V8_NON_SCALAR_LOSS = 10,
    GEO_V8_NUMERIC_FAILURE = 11,
    GEO_V8_CONSTRAINT_FAILURE = 12,
    GEO_V8_STATE_UPDATE_REQUIRED = 13
} geo_v8_status_t;

typedef enum {
    GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN = 1
} geo_v8_pairing_t;

typedef enum {
    GEO_V8_NODE_INPUT = 1,
    GEO_V8_NODE_TARGET = 2,
    GEO_V8_NODE_PARAMETER = 3,
    GEO_V8_NODE_CONSTANT = 4,
    GEO_V8_NODE_STATE = 5,
    GEO_V8_NODE_ADD = 6,
    GEO_V8_NODE_SCALE = 7,
    GEO_V8_NODE_GEOMETRIC_PRODUCT = 8,
    GEO_V8_NODE_REVERSE = 9,
    GEO_V8_NODE_GRADE_PROJECT = 10,
    GEO_V8_NODE_GRADE_INVOLUTION = 11,
    GEO_V8_NODE_CLIFFORD_CONJUGATE = 12,
    GEO_V8_NODE_HADAMARD = 13,
    GEO_V8_NODE_TANH = 14,
    GEO_V8_NODE_SIGMOID = 15,
    GEO_V8_NODE_EUCLIDEAN_NORMALIZE = 16,
    GEO_V8_NODE_SQUARED_NORM = 17
} geo_v8_node_kind_t;

typedef enum {
    GEO_V8_CONSTRAINT_NONE = 0,
    GEO_V8_CONSTRAINT_UNIT_EUCLIDEAN = 1,
    GEO_V8_CONSTRAINT_UNIT_VECTOR_METRIC = 2,
    GEO_V8_CONSTRAINT_EVEN_VERSOR = 3
} geo_v8_constraint_t;

typedef enum {
    GEO_V8_OPTIMIZER_SGD = 1,
    GEO_V8_OPTIMIZER_ADAM = 2
} geo_v8_optimizer_t;

typedef struct {
    uint8_t kind;
    uint8_t requires_grad;
    uint8_t grade;
    uint8_t constraint;
    geo_v8_node_id_t left;
    geo_v8_node_id_t right;
    geo_v8_node_id_t state_update;
    double scalar;
    char name[GEO_V8_NAME_MAX];
    geo_operator_mv_f64_t value;
    geo_operator_mv_f64_t cotangent;
    geo_operator_mv_f64_t gradient;
    double first_moment[GEO_OPERATOR_MAX_BLADES];
    double second_moment[GEO_OPERATOR_MAX_BLADES];
} geo_v8_node_t;

typedef struct {
    uint32_t abi_version;
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    uint8_t pairing;
    uint8_t compiled;
    uint8_t forward_valid;
    uint8_t backward_valid;
    uint8_t reserved;
    geo_v8_node_id_t node_count;
    geo_v8_node_id_t node_capacity;
    geo_v8_node_id_t loss_node;
    uint64_t optimizer_step;
    uint64_t gradient_samples;
    geo_v8_node_t *nodes;
} geo_v8_program_t;

uint32_t geo_v8_abi_version(void);

geo_v8_status_t geo_v8_program_init(
    geo_v8_program_t *program,
    uint8_t dimension,
    const int8_t *signature,
    geo_v8_pairing_t pairing,
    size_t initial_capacity
);

void geo_v8_program_free(geo_v8_program_t *program);

geo_v8_status_t geo_v8_program_reserve(
    geo_v8_program_t *program,
    size_t node_capacity
);

geo_v8_status_t geo_v8_add_leaf(
    geo_v8_program_t *program,
    geo_v8_node_kind_t kind,
    const char *name,
    const geo_operator_mv_f64_t *initial_value,
    int requires_grad,
    geo_v8_constraint_t constraint,
    geo_v8_node_id_t *node_id
);

geo_v8_status_t geo_v8_add_unary(
    geo_v8_program_t *program,
    geo_v8_node_kind_t kind,
    const char *name,
    geo_v8_node_id_t input,
    double scalar,
    uint8_t grade,
    geo_v8_node_id_t *node_id
);

geo_v8_status_t geo_v8_add_binary(
    geo_v8_program_t *program,
    geo_v8_node_kind_t kind,
    const char *name,
    geo_v8_node_id_t left,
    geo_v8_node_id_t right,
    geo_v8_node_id_t *node_id
);

geo_v8_status_t geo_v8_bind_state_update(
    geo_v8_program_t *program,
    geo_v8_node_id_t state_node,
    geo_v8_node_id_t source_node
);

geo_v8_status_t geo_v8_compile(
    geo_v8_program_t *program,
    geo_v8_node_id_t loss_node
);

geo_v8_node_id_t geo_v8_find_node(
    const geo_v8_program_t *program,
    const char *name
);

geo_v8_status_t geo_v8_set_value(
    geo_v8_program_t *program,
    geo_v8_node_id_t node_id,
    const geo_operator_mv_f64_t *value
);

geo_v8_status_t geo_v8_forward(geo_v8_program_t *program);
geo_v8_status_t geo_v8_zero_grad(geo_v8_program_t *program);

geo_v8_status_t geo_v8_backward(
    geo_v8_program_t *program,
    int accumulate
);

geo_v8_status_t geo_v8_optimizer_step(
    geo_v8_program_t *program,
    geo_v8_optimizer_t optimizer,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double gradient_scale
);

geo_v8_status_t geo_v8_commit_states(geo_v8_program_t *program);

const geo_operator_mv_f64_t *geo_v8_value(
    const geo_v8_program_t *program,
    geo_v8_node_id_t node_id
);

const geo_operator_mv_f64_t *geo_v8_gradient(
    const geo_v8_program_t *program,
    geo_v8_node_id_t node_id
);

#ifdef __cplusplus
}
#endif

#endif
