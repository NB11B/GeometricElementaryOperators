#include "geo/full_cycle_v8.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int v8_dimension_valid(uint8_t n) {
    return n >= 1u && n <= GEO_OPERATOR_MAX_DIMENSION;
}

static int v8_signature_valid(const int8_t *signature, uint8_t n) {
    uint8_t i;
    if (signature == NULL || !v8_dimension_valid(n)) return 0;
    for (i = 0u; i < n; ++i) {
        if (signature[i] != 1 && signature[i] != -1) return 0;
    }
    return 1;
}

static size_t v8_blade_count(uint8_t n) {
    return (size_t)1u << n;
}

static unsigned v8_popcount(uint8_t value) {
    unsigned count = 0u;
    while (value != 0u) {
        value = (uint8_t)(value & (uint8_t)(value - 1u));
        ++count;
    }
    return count;
}

static int v8_same_signature(uint8_t n, const int8_t *a, const int8_t *b) {
    uint8_t i;
    if (a == NULL || b == NULL) return 0;
    for (i = 0u; i < n; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static void v8_mv_zero(geo_operator_mv_f64_t *value, uint8_t n, const int8_t *signature) {
    memset(value, 0, sizeof(*value));
    value->dimension = n;
    memcpy(value->signature, signature, (size_t)n * sizeof(signature[0]));
}

static int v8_mv_valid(const geo_operator_mv_f64_t *value, uint8_t n, const int8_t *signature) {
    size_t blade;
    const size_t count = v8_blade_count(n);
    if (value == NULL || value->dimension != n ||
        !v8_same_signature(n, value->signature, signature)) {
        return 0;
    }
    for (blade = 0u; blade < count; ++blade) {
        if (!isfinite(value->coefficients[blade])) return 0;
    }
    return 1;
}

static int v8_program_valid(const geo_v8_program_t *program) {
    return program != NULL &&
        program->abi_version == GEO_V8_ABI_VERSION &&
        v8_signature_valid(program->signature, program->dimension) &&
        program->pairing == GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN &&
        program->node_count <= program->node_capacity &&
        (program->node_capacity == 0u || program->nodes != NULL);
}

static int v8_node_exists(const geo_v8_program_t *program, geo_v8_node_id_t node_id) {
    return v8_program_valid(program) && node_id < program->node_count;
}

static int v8_name_valid(const char *name) {
    const size_t length = name == NULL ? 0u : strlen(name);
    return length > 0u && length < GEO_V8_NAME_MAX;
}

static int v8_leaf_kind(geo_v8_node_kind_t kind) {
    return kind == GEO_V8_NODE_INPUT || kind == GEO_V8_NODE_TARGET ||
        kind == GEO_V8_NODE_PARAMETER || kind == GEO_V8_NODE_CONSTANT ||
        kind == GEO_V8_NODE_STATE;
}

static int v8_unary_kind(geo_v8_node_kind_t kind) {
    return kind == GEO_V8_NODE_SCALE || kind == GEO_V8_NODE_REVERSE ||
        kind == GEO_V8_NODE_GRADE_PROJECT ||
        kind == GEO_V8_NODE_GRADE_INVOLUTION ||
        kind == GEO_V8_NODE_CLIFFORD_CONJUGATE ||
        kind == GEO_V8_NODE_TANH || kind == GEO_V8_NODE_SIGMOID ||
        kind == GEO_V8_NODE_EUCLIDEAN_NORMALIZE ||
        kind == GEO_V8_NODE_SQUARED_NORM;
}

static int v8_binary_kind(geo_v8_node_kind_t kind) {
    return kind == GEO_V8_NODE_ADD ||
        kind == GEO_V8_NODE_GEOMETRIC_PRODUCT ||
        kind == GEO_V8_NODE_HADAMARD;
}

static int v8_constraint_valid(geo_v8_constraint_t constraint) {
    return constraint >= GEO_V8_CONSTRAINT_NONE &&
        constraint <= GEO_V8_CONSTRAINT_EVEN_VERSOR;
}

static int v8_node_name_unique(const geo_v8_program_t *program, const char *name) {
    geo_v8_node_id_t i;
    for (i = 0u; i < program->node_count; ++i) {
        if (strcmp(program->nodes[i].name, name) == 0) return 0;
    }
    return 1;
}

static geo_v8_status_t v8_accumulate(
    geo_operator_mv_f64_t *destination,
    const geo_operator_mv_f64_t *source,
    double scale,
    size_t count
) {
    size_t blade;
    for (blade = 0u; blade < count; ++blade) {
        const double next = destination->coefficients[blade] +
            source->coefficients[blade] * scale;
        if (!isfinite(next)) return GEO_V8_NUMERIC_FAILURE;
        destination->coefficients[blade] = next;
    }
    return GEO_V8_OK;
}

static int v8_reverse_sign(uint8_t blade) {
    const unsigned grade = v8_popcount(blade);
    return (((grade * (grade - 1u) / 2u) & 1u) != 0u) ? -1 : 1;
}

static int v8_involution_sign(uint8_t blade) {
    return (v8_popcount(blade) & 1u) != 0u ? -1 : 1;
}

static int v8_conjugate_sign(uint8_t blade) {
    const unsigned grade = v8_popcount(blade);
    return (((grade * (grade + 1u) / 2u) & 1u) != 0u) ? -1 : 1;
}

static geo_v8_status_t v8_apply_sign_map(
    const geo_operator_mv_f64_t *input,
    geo_operator_mv_f64_t *output,
    uint8_t n,
    const int8_t *signature,
    int mode
) {
    size_t blade;
    const size_t count = v8_blade_count(n);
    v8_mv_zero(output, n, signature);
    for (blade = 0u; blade < count; ++blade) {
        int sign;
        if (mode == 0) sign = v8_reverse_sign((uint8_t)blade);
        else if (mode == 1) sign = v8_involution_sign((uint8_t)blade);
        else sign = v8_conjugate_sign((uint8_t)blade);
        output->coefficients[blade] = input->coefficients[blade] * (double)sign;
    }
    return GEO_V8_OK;
}

static void v8_apply_grade_project(
    const geo_operator_mv_f64_t *input,
    uint8_t grade,
    geo_operator_mv_f64_t *output,
    uint8_t n,
    const int8_t *signature
) {
    size_t blade;
    const size_t count = v8_blade_count(n);
    v8_mv_zero(output, n, signature);
    for (blade = 0u; blade < count; ++blade) {
        if (v8_popcount((uint8_t)blade) == (unsigned)grade) {
            output->coefficients[blade] = input->coefficients[blade];
        }
    }
}

static double v8_sigmoid(double value) {
    if (value >= 0.0) {
        const double z = exp(-value);
        return 1.0 / (1.0 + z);
    }
    {
        const double z = exp(value);
        return z / (1.0 + z);
    }
}

static geo_v8_status_t v8_project_constraint(
    const geo_v8_program_t *program,
    geo_v8_node_t *node
) {
    size_t blade;
    const size_t count = v8_blade_count(program->dimension);
    const double tiny = 1e-14;

    if (node->constraint == GEO_V8_CONSTRAINT_NONE) return GEO_V8_OK;

    if (node->constraint == GEO_V8_CONSTRAINT_UNIT_EUCLIDEAN) {
        double norm2 = 0.0;
        for (blade = 0u; blade < count; ++blade) {
            const double c = node->value.coefficients[blade];
            norm2 += c * c;
        }
        if (!isfinite(norm2) || norm2 <= tiny) return GEO_V8_CONSTRAINT_FAILURE;
        {
            const double inverse = 1.0 / sqrt(norm2);
            for (blade = 0u; blade < count; ++blade) {
                node->value.coefficients[blade] *= inverse;
            }
        }
        return GEO_V8_OK;
    }

    if (node->constraint == GEO_V8_CONSTRAINT_UNIT_VECTOR_METRIC) {
        double metric_norm = 0.0;
        uint8_t axis;
        for (blade = 0u; blade < count; ++blade) {
            if (v8_popcount((uint8_t)blade) != 1u) {
                node->value.coefficients[blade] = 0.0;
            }
        }
        for (axis = 0u; axis < program->dimension; ++axis) {
            const size_t index = (size_t)1u << axis;
            const double c = node->value.coefficients[index];
            metric_norm += (double)program->signature[axis] * c * c;
        }
        if (!isfinite(metric_norm) || fabs(metric_norm) <= tiny) {
            return GEO_V8_CONSTRAINT_FAILURE;
        }
        {
            const double inverse = 1.0 / sqrt(fabs(metric_norm));
            for (axis = 0u; axis < program->dimension; ++axis) {
                node->value.coefficients[(size_t)1u << axis] *= inverse;
            }
        }
        return GEO_V8_OK;
    }

    if (node->constraint == GEO_V8_CONSTRAINT_EVEN_VERSOR) {
        geo_operator_mv_f64_t reverse;
        geo_operator_mv_f64_t product;
        double residual2 = 0.0;
        double norm_scalar;
        for (blade = 0u; blade < count; ++blade) {
            if ((v8_popcount((uint8_t)blade) & 1u) != 0u) {
                node->value.coefficients[blade] = 0.0;
            }
        }
        (void)v8_apply_sign_map(
            &node->value,
            &reverse,
            program->dimension,
            program->signature,
            0
        );
        if (geo_operator_gp_f64(&node->value, &reverse, &product) != GEO_OPERATOR_OK) {
            return GEO_V8_CONSTRAINT_FAILURE;
        }
        norm_scalar = product.coefficients[0];
        for (blade = 1u; blade < count; ++blade) {
            residual2 += product.coefficients[blade] * product.coefficients[blade];
        }
        if (!isfinite(norm_scalar) || !isfinite(residual2) || fabs(norm_scalar) <= tiny ||
            residual2 > 1e-12 * (1.0 + norm_scalar * norm_scalar)) {
            return GEO_V8_CONSTRAINT_FAILURE;
        }
        {
            const double inverse = 1.0 / sqrt(fabs(norm_scalar));
            for (blade = 0u; blade < count; ++blade) {
                node->value.coefficients[blade] *= inverse;
            }
        }
        return GEO_V8_OK;
    }

    return GEO_V8_CONSTRAINT_FAILURE;
}

uint32_t geo_v8_abi_version(void) {
    return GEO_V8_ABI_VERSION;
}

geo_v8_status_t geo_v8_program_init(
    geo_v8_program_t *program,
    uint8_t dimension,
    const int8_t *signature,
    geo_v8_pairing_t pairing,
    size_t initial_capacity
) {
    if (program == NULL || signature == NULL) return GEO_V8_INVALID_ARGUMENT;
    if (!v8_dimension_valid(dimension)) return GEO_V8_UNSUPPORTED_DIMENSION;
    if (!v8_signature_valid(signature, dimension) ||
        pairing != GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN ||
        initial_capacity > (size_t)UINT32_MAX) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    memset(program, 0, sizeof(*program));
    program->abi_version = GEO_V8_ABI_VERSION;
    program->dimension = dimension;
    memcpy(program->signature, signature, (size_t)dimension * sizeof(signature[0]));
    program->pairing = (uint8_t)pairing;
    program->loss_node = GEO_V8_INVALID_NODE;
    if (initial_capacity != 0u) {
        return geo_v8_program_reserve(program, initial_capacity);
    }
    return GEO_V8_OK;
}

void geo_v8_program_free(geo_v8_program_t *program) {
    if (program == NULL) return;
    free(program->nodes);
    memset(program, 0, sizeof(*program));
}

geo_v8_status_t geo_v8_program_reserve(
    geo_v8_program_t *program,
    size_t node_capacity
) {
    geo_v8_node_t *replacement;
    size_t old_capacity;
    if (!v8_program_valid(program) || node_capacity > (size_t)UINT32_MAX) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    if (program->compiled != 0u) return GEO_V8_BAD_PROGRAM;
    if (node_capacity <= (size_t)program->node_capacity) return GEO_V8_OK;
    if (node_capacity > SIZE_MAX / sizeof(*replacement)) return GEO_V8_ALLOCATION_FAILURE;
    old_capacity = (size_t)program->node_capacity;
    replacement = (geo_v8_node_t *)realloc(
        program->nodes,
        node_capacity * sizeof(*replacement)
    );
    if (replacement == NULL) return GEO_V8_ALLOCATION_FAILURE;
    memset(
        replacement + old_capacity,
        0,
        (node_capacity - old_capacity) * sizeof(*replacement)
    );
    program->nodes = replacement;
    program->node_capacity = (geo_v8_node_id_t)node_capacity;
    return GEO_V8_OK;
}

static geo_v8_status_t v8_reserve_one(
    geo_v8_program_t *program,
    const char *name,
    geo_v8_node_id_t *node_id
) {
    size_t next_capacity;
    geo_v8_status_t status;
    if (!v8_program_valid(program) || node_id == NULL || !v8_name_valid(name)) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    if (program->compiled != 0u) return GEO_V8_BAD_PROGRAM;
    if (!v8_node_name_unique(program, name)) return GEO_V8_DUPLICATE_NAME;
    if (program->node_count == program->node_capacity) {
        next_capacity = program->node_capacity == 0u ? 16u :
            (size_t)program->node_capacity * 2u;
        if (next_capacity <= (size_t)program->node_capacity ||
            next_capacity > (size_t)UINT32_MAX) {
            return GEO_V8_ALLOCATION_FAILURE;
        }
        status = geo_v8_program_reserve(program, next_capacity);
        if (status != GEO_V8_OK) return status;
    }
    *node_id = program->node_count;
    ++program->node_count;
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_add_leaf(
    geo_v8_program_t *program,
    geo_v8_node_kind_t kind,
    const char *name,
    const geo_operator_mv_f64_t *initial_value,
    int requires_grad,
    geo_v8_constraint_t constraint,
    geo_v8_node_id_t *node_id
) {
    geo_v8_node_t *node;
    geo_v8_status_t status;
    if (!v8_program_valid(program) || !v8_leaf_kind(kind) ||
        !v8_constraint_valid(constraint) ||
        !v8_mv_valid(initial_value, program->dimension, program->signature)) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    if (kind != GEO_V8_NODE_PARAMETER && constraint != GEO_V8_CONSTRAINT_NONE) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    status = v8_reserve_one(program, name, node_id);
    if (status != GEO_V8_OK) return status;
    node = &program->nodes[*node_id];
    memset(node, 0, sizeof(*node));
    node->kind = (uint8_t)kind;
    node->requires_grad = requires_grad != 0 ? 1u : 0u;
    if (kind == GEO_V8_NODE_PARAMETER) node->requires_grad = 1u;
    node->constraint = (uint8_t)constraint;
    node->left = GEO_V8_INVALID_NODE;
    node->right = GEO_V8_INVALID_NODE;
    node->state_update = GEO_V8_INVALID_NODE;
    memcpy(node->name, name, strlen(name) + 1u);
    node->value = *initial_value;
    v8_mv_zero(&node->cotangent, program->dimension, program->signature);
    v8_mv_zero(&node->gradient, program->dimension, program->signature);
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_add_unary(
    geo_v8_program_t *program,
    geo_v8_node_kind_t kind,
    const char *name,
    geo_v8_node_id_t input,
    double scalar,
    uint8_t grade,
    geo_v8_node_id_t *node_id
) {
    geo_v8_node_t *node;
    geo_v8_status_t status;
    if (!v8_program_valid(program) || !v8_unary_kind(kind) ||
        !v8_node_exists(program, input) || !isfinite(scalar) ||
        (kind == GEO_V8_NODE_GRADE_PROJECT && grade > program->dimension)) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    status = v8_reserve_one(program, name, node_id);
    if (status != GEO_V8_OK) return status;
    node = &program->nodes[*node_id];
    memset(node, 0, sizeof(*node));
    node->kind = (uint8_t)kind;
    node->requires_grad = program->nodes[input].requires_grad;
    node->grade = grade;
    node->left = input;
    node->right = GEO_V8_INVALID_NODE;
    node->state_update = GEO_V8_INVALID_NODE;
    node->scalar = scalar;
    memcpy(node->name, name, strlen(name) + 1u);
    v8_mv_zero(&node->value, program->dimension, program->signature);
    v8_mv_zero(&node->cotangent, program->dimension, program->signature);
    v8_mv_zero(&node->gradient, program->dimension, program->signature);
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_add_binary(
    geo_v8_program_t *program,
    geo_v8_node_kind_t kind,
    const char *name,
    geo_v8_node_id_t left,
    geo_v8_node_id_t right,
    geo_v8_node_id_t *node_id
) {
    geo_v8_node_t *node;
    geo_v8_status_t status;
    if (!v8_program_valid(program) || !v8_binary_kind(kind) ||
        !v8_node_exists(program, left) || !v8_node_exists(program, right)) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    status = v8_reserve_one(program, name, node_id);
    if (status != GEO_V8_OK) return status;
    node = &program->nodes[*node_id];
    memset(node, 0, sizeof(*node));
    node->kind = (uint8_t)kind;
    node->requires_grad =
        (program->nodes[left].requires_grad != 0u ||
         program->nodes[right].requires_grad != 0u) ? 1u : 0u;
    node->left = left;
    node->right = right;
    node->state_update = GEO_V8_INVALID_NODE;
    memcpy(node->name, name, strlen(name) + 1u);
    v8_mv_zero(&node->value, program->dimension, program->signature);
    v8_mv_zero(&node->cotangent, program->dimension, program->signature);
    v8_mv_zero(&node->gradient, program->dimension, program->signature);
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_bind_state_update(
    geo_v8_program_t *program,
    geo_v8_node_id_t state_node,
    geo_v8_node_id_t source_node
) {
    if (!v8_node_exists(program, state_node) || !v8_node_exists(program, source_node)) {
        return GEO_V8_BAD_NODE;
    }
    if (program->compiled != 0u ||
        program->nodes[state_node].kind != GEO_V8_NODE_STATE) {
        return GEO_V8_BAD_PROGRAM;
    }
    program->nodes[state_node].state_update = source_node;
    return GEO_V8_OK;
}

geo_v8_node_id_t geo_v8_find_node(
    const geo_v8_program_t *program,
    const char *name
) {
    geo_v8_node_id_t i;
    if (!v8_program_valid(program) || !v8_name_valid(name)) {
        return GEO_V8_INVALID_NODE;
    }
    for (i = 0u; i < program->node_count; ++i) {
        if (strcmp(program->nodes[i].name, name) == 0) return i;
    }
    return GEO_V8_INVALID_NODE;
}

geo_v8_status_t geo_v8_compile(
    geo_v8_program_t *program,
    geo_v8_node_id_t loss_node
) {
    geo_v8_node_id_t i;
    if (!v8_program_valid(program)) return GEO_V8_BAD_PROGRAM;
    if (program->node_count == 0u || loss_node >= program->node_count) {
        return GEO_V8_BAD_NODE;
    }
    for (i = 0u; i < program->node_count; ++i) {
        const geo_v8_node_t *node = &program->nodes[i];
        const geo_v8_node_kind_t kind = (geo_v8_node_kind_t)node->kind;
        if (!v8_name_valid(node->name)) return GEO_V8_BAD_PROGRAM;
        if (v8_leaf_kind(kind)) {
            if (node->left != GEO_V8_INVALID_NODE || node->right != GEO_V8_INVALID_NODE ||
                !v8_mv_valid(&node->value, program->dimension, program->signature)) {
                return GEO_V8_BAD_PROGRAM;
            }
            if (kind == GEO_V8_NODE_STATE && node->state_update != GEO_V8_INVALID_NODE &&
                node->state_update >= program->node_count) {
                return GEO_V8_BAD_PROGRAM;
            }
        } else if (v8_unary_kind(kind)) {
            if (node->left >= i || node->right != GEO_V8_INVALID_NODE) {
                return GEO_V8_BAD_PROGRAM;
            }
        } else if (v8_binary_kind(kind)) {
            if (node->left >= i || node->right >= i) return GEO_V8_BAD_PROGRAM;
        } else {
            return GEO_V8_BAD_PROGRAM;
        }
    }
    program->compiled = 1u;
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    program->loss_node = loss_node;
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_set_value(
    geo_v8_program_t *program,
    geo_v8_node_id_t node_id,
    const geo_operator_mv_f64_t *value
) {
    geo_v8_node_t *node;
    if (!v8_node_exists(program, node_id)) return GEO_V8_BAD_NODE;
    if (!v8_mv_valid(value, program->dimension, program->signature)) {
        return GEO_V8_INVALID_ARGUMENT;
    }
    node = &program->nodes[node_id];
    if (!v8_leaf_kind((geo_v8_node_kind_t)node->kind)) return GEO_V8_BAD_NODE;
    node->value = *value;
    if (node->kind == GEO_V8_NODE_PARAMETER) {
        memset(node->first_moment, 0, sizeof(node->first_moment));
        memset(node->second_moment, 0, sizeof(node->second_moment));
        program->optimizer_step = 0u;
    }
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_forward(geo_v8_program_t *program) {
    geo_v8_node_id_t i;
    const size_t count = program == NULL ? 0u : v8_blade_count(program->dimension);
    if (!v8_program_valid(program)) return GEO_V8_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V8_NOT_COMPILED;
    program->forward_valid = 0u;
    program->backward_valid = 0u;

    for (i = 0u; i < program->node_count; ++i) {
        geo_v8_node_t *node = &program->nodes[i];
        const geo_v8_node_kind_t kind = (geo_v8_node_kind_t)node->kind;
        geo_operator_mv_f64_t result;
        size_t blade;
        v8_mv_zero(&result, program->dimension, program->signature);

        if (v8_leaf_kind(kind)) {
            if (!v8_mv_valid(&node->value, program->dimension, program->signature)) {
                return GEO_V8_NUMERIC_FAILURE;
            }
            continue;
        }

        if (kind == GEO_V8_NODE_ADD) {
            for (blade = 0u; blade < count; ++blade) {
                result.coefficients[blade] =
                    program->nodes[node->left].value.coefficients[blade] +
                    program->nodes[node->right].value.coefficients[blade];
            }
        } else if (kind == GEO_V8_NODE_SCALE) {
            for (blade = 0u; blade < count; ++blade) {
                result.coefficients[blade] =
                    program->nodes[node->left].value.coefficients[blade] * node->scalar;
            }
        } else if (kind == GEO_V8_NODE_GEOMETRIC_PRODUCT) {
            if (geo_operator_gp_f64(
                    &program->nodes[node->left].value,
                    &program->nodes[node->right].value,
                    &result
                ) != GEO_OPERATOR_OK) {
                return GEO_V8_BAD_PROGRAM;
            }
        } else if (kind == GEO_V8_NODE_REVERSE) {
            (void)v8_apply_sign_map(
                &program->nodes[node->left].value,
                &result,
                program->dimension,
                program->signature,
                0
            );
        } else if (kind == GEO_V8_NODE_GRADE_PROJECT) {
            v8_apply_grade_project(
                &program->nodes[node->left].value,
                node->grade,
                &result,
                program->dimension,
                program->signature
            );
        } else if (kind == GEO_V8_NODE_GRADE_INVOLUTION) {
            (void)v8_apply_sign_map(
                &program->nodes[node->left].value,
                &result,
                program->dimension,
                program->signature,
                1
            );
        } else if (kind == GEO_V8_NODE_CLIFFORD_CONJUGATE) {
            (void)v8_apply_sign_map(
                &program->nodes[node->left].value,
                &result,
                program->dimension,
                program->signature,
                2
            );
        } else if (kind == GEO_V8_NODE_HADAMARD) {
            for (blade = 0u; blade < count; ++blade) {
                result.coefficients[blade] =
                    program->nodes[node->left].value.coefficients[blade] *
                    program->nodes[node->right].value.coefficients[blade];
            }
        } else if (kind == GEO_V8_NODE_TANH) {
            for (blade = 0u; blade < count; ++blade) {
                result.coefficients[blade] =
                    tanh(program->nodes[node->left].value.coefficients[blade]);
            }
        } else if (kind == GEO_V8_NODE_SIGMOID) {
            for (blade = 0u; blade < count; ++blade) {
                result.coefficients[blade] =
                    v8_sigmoid(program->nodes[node->left].value.coefficients[blade]);
            }
        } else if (kind == GEO_V8_NODE_EUCLIDEAN_NORMALIZE) {
            double norm2 = 0.0;
            const double floor_value = node->scalar > 0.0 ? node->scalar : 1e-12;
            for (blade = 0u; blade < count; ++blade) {
                const double c = program->nodes[node->left].value.coefficients[blade];
                norm2 += c * c;
            }
            if (!isfinite(norm2) || norm2 <= floor_value * floor_value) {
                return GEO_V8_NUMERIC_FAILURE;
            }
            {
                const double inverse = 1.0 / sqrt(norm2);
                for (blade = 0u; blade < count; ++blade) {
                    result.coefficients[blade] =
                        program->nodes[node->left].value.coefficients[blade] * inverse;
                }
            }
        } else if (kind == GEO_V8_NODE_SQUARED_NORM) {
            for (blade = 0u; blade < count; ++blade) {
                const double c = program->nodes[node->left].value.coefficients[blade];
                result.coefficients[0] += 0.5 * c * c;
            }
        } else {
            return GEO_V8_BAD_PROGRAM;
        }

        if (!v8_mv_valid(&result, program->dimension, program->signature)) {
            return GEO_V8_NUMERIC_FAILURE;
        }
        node->value = result;
    }

    program->forward_valid = 1u;
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_zero_grad(geo_v8_program_t *program) {
    geo_v8_node_id_t i;
    if (!v8_program_valid(program)) return GEO_V8_BAD_PROGRAM;
    for (i = 0u; i < program->node_count; ++i) {
        v8_mv_zero(&program->nodes[i].cotangent, program->dimension, program->signature);
        v8_mv_zero(&program->nodes[i].gradient, program->dimension, program->signature);
    }
    program->gradient_samples = 0u;
    program->backward_valid = 0u;
    return GEO_V8_OK;
}

static int v8_loss_scalar(const geo_v8_program_t *program) {
    size_t blade;
    const size_t count = v8_blade_count(program->dimension);
    const geo_operator_mv_f64_t *loss = &program->nodes[program->loss_node].value;
    if (!isfinite(loss->coefficients[0])) return 0;
    for (blade = 1u; blade < count; ++blade) {
        if (loss->coefficients[blade] != 0.0) return 0;
    }
    return 1;
}

geo_v8_status_t geo_v8_backward(
    geo_v8_program_t *program,
    int accumulate
) {
    geo_v8_node_id_t cursor;
    geo_v8_node_id_t i;
    geo_v8_status_t status;
    const size_t count = program == NULL ? 0u : v8_blade_count(program->dimension);

    if (!v8_program_valid(program)) return GEO_V8_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V8_NOT_COMPILED;
    if (program->forward_valid == 0u) return GEO_V8_FORWARD_REQUIRED;
    if (!v8_loss_scalar(program)) return GEO_V8_NON_SCALAR_LOSS;

    if (accumulate == 0) {
        for (i = 0u; i < program->node_count; ++i) {
            v8_mv_zero(&program->nodes[i].gradient, program->dimension, program->signature);
        }
        program->gradient_samples = 0u;
    }
    for (i = 0u; i < program->node_count; ++i) {
        v8_mv_zero(&program->nodes[i].cotangent, program->dimension, program->signature);
    }

    program->nodes[program->loss_node].cotangent.coefficients[0] = 1.0;
    cursor = program->node_count;
    while (cursor > 0u) {
        geo_v8_node_t *node;
        geo_v8_node_kind_t kind;
        --cursor;
        node = &program->nodes[cursor];
        kind = (geo_v8_node_kind_t)node->kind;
        if (node->requires_grad == 0u) continue;

        if (v8_leaf_kind(kind)) {
            continue;
        } else if (kind == GEO_V8_NODE_ADD) {
            if (program->nodes[node->left].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->left].cotangent,
                    &node->cotangent,
                    1.0,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
            if (program->nodes[node->right].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->right].cotangent,
                    &node->cotangent,
                    1.0,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
        } else if (kind == GEO_V8_NODE_SCALE) {
            if (program->nodes[node->left].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->left].cotangent,
                    &node->cotangent,
                    node->scalar,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
        } else if (kind == GEO_V8_NODE_GEOMETRIC_PRODUCT) {
            geo_operator_mv_f64_t left_bar;
            geo_operator_mv_f64_t right_bar;
            if (geo_operator_gp_f64_vjp(
                    &program->nodes[node->left].value,
                    &program->nodes[node->right].value,
                    &node->cotangent,
                    &left_bar,
                    &right_bar
                ) != GEO_OPERATOR_OK) {
                return GEO_V8_BAD_PROGRAM;
            }
            if (program->nodes[node->left].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->left].cotangent,
                    &left_bar,
                    1.0,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
            if (program->nodes[node->right].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->right].cotangent,
                    &right_bar,
                    1.0,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
        } else if (kind == GEO_V8_NODE_REVERSE ||
                   kind == GEO_V8_NODE_GRADE_INVOLUTION ||
                   kind == GEO_V8_NODE_CLIFFORD_CONJUGATE) {
            geo_operator_mv_f64_t mapped;
            const int mode = kind == GEO_V8_NODE_REVERSE ? 0 :
                (kind == GEO_V8_NODE_GRADE_INVOLUTION ? 1 : 2);
            (void)v8_apply_sign_map(
                &node->cotangent,
                &mapped,
                program->dimension,
                program->signature,
                mode
            );
            if (program->nodes[node->left].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->left].cotangent,
                    &mapped,
                    1.0,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
        } else if (kind == GEO_V8_NODE_GRADE_PROJECT) {
            geo_operator_mv_f64_t mapped;
            v8_apply_grade_project(
                &node->cotangent,
                node->grade,
                &mapped,
                program->dimension,
                program->signature
            );
            if (program->nodes[node->left].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->left].cotangent,
                    &mapped,
                    1.0,
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
        } else if (kind == GEO_V8_NODE_HADAMARD) {
            size_t blade;
            if (program->nodes[node->left].requires_grad != 0u) {
                for (blade = 0u; blade < count; ++blade) {
                    const double next = program->nodes[node->left].cotangent.coefficients[blade] +
                        node->cotangent.coefficients[blade] *
                        program->nodes[node->right].value.coefficients[blade];
                    if (!isfinite(next)) return GEO_V8_NUMERIC_FAILURE;
                    program->nodes[node->left].cotangent.coefficients[blade] = next;
                }
            }
            if (program->nodes[node->right].requires_grad != 0u) {
                for (blade = 0u; blade < count; ++blade) {
                    const double next = program->nodes[node->right].cotangent.coefficients[blade] +
                        node->cotangent.coefficients[blade] *
                        program->nodes[node->left].value.coefficients[blade];
                    if (!isfinite(next)) return GEO_V8_NUMERIC_FAILURE;
                    program->nodes[node->right].cotangent.coefficients[blade] = next;
                }
            }
        } else if (kind == GEO_V8_NODE_TANH || kind == GEO_V8_NODE_SIGMOID) {
            size_t blade;
            if (program->nodes[node->left].requires_grad != 0u) {
                for (blade = 0u; blade < count; ++blade) {
                    const double output = node->value.coefficients[blade];
                    const double derivative = kind == GEO_V8_NODE_TANH ?
                        1.0 - output * output : output * (1.0 - output);
                    const double next = program->nodes[node->left].cotangent.coefficients[blade] +
                        node->cotangent.coefficients[blade] * derivative;
                    if (!isfinite(next)) return GEO_V8_NUMERIC_FAILURE;
                    program->nodes[node->left].cotangent.coefficients[blade] = next;
                }
            }
        } else if (kind == GEO_V8_NODE_EUCLIDEAN_NORMALIZE) {
            size_t blade;
            double norm2 = 0.0;
            double dot = 0.0;
            const geo_operator_mv_f64_t *input = &program->nodes[node->left].value;
            for (blade = 0u; blade < count; ++blade) {
                norm2 += input->coefficients[blade] * input->coefficients[blade];
                dot += node->value.coefficients[blade] * node->cotangent.coefficients[blade];
            }
            if (!isfinite(norm2) || norm2 <= 0.0 || !isfinite(dot)) {
                return GEO_V8_NUMERIC_FAILURE;
            }
            if (program->nodes[node->left].requires_grad != 0u) {
                const double inverse = 1.0 / sqrt(norm2);
                for (blade = 0u; blade < count; ++blade) {
                    const double contribution = inverse *
                        (node->cotangent.coefficients[blade] -
                         node->value.coefficients[blade] * dot);
                    const double next = program->nodes[node->left].cotangent.coefficients[blade] +
                        contribution;
                    if (!isfinite(next)) return GEO_V8_NUMERIC_FAILURE;
                    program->nodes[node->left].cotangent.coefficients[blade] = next;
                }
            }
        } else if (kind == GEO_V8_NODE_SQUARED_NORM) {
            size_t blade;
            for (blade = 1u; blade < count; ++blade) {
                if (node->cotangent.coefficients[blade] != 0.0) {
                    return GEO_V8_NON_SCALAR_LOSS;
                }
            }
            if (program->nodes[node->left].requires_grad != 0u) {
                status = v8_accumulate(
                    &program->nodes[node->left].cotangent,
                    &program->nodes[node->left].value,
                    node->cotangent.coefficients[0],
                    count
                );
                if (status != GEO_V8_OK) return status;
            }
        } else {
            return GEO_V8_BAD_PROGRAM;
        }
    }

    for (i = 0u; i < program->node_count; ++i) {
        geo_v8_node_t *node = &program->nodes[i];
        if (v8_leaf_kind((geo_v8_node_kind_t)node->kind) && node->requires_grad != 0u) {
            status = v8_accumulate(&node->gradient, &node->cotangent, 1.0, count);
            if (status != GEO_V8_OK) return status;
        }
    }
    ++program->gradient_samples;
    program->backward_valid = 1u;
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_optimizer_step(
    geo_v8_program_t *program,
    geo_v8_optimizer_t optimizer,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double gradient_scale
) {
    geo_v8_node_t *candidate;
    geo_v8_node_id_t i;
    uint64_t step;
    double bias1 = 1.0;
    double bias2 = 1.0;
    const size_t count = program == NULL ? 0u : v8_blade_count(program->dimension);

    if (!v8_program_valid(program)) return GEO_V8_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V8_NOT_COMPILED;
    if (program->backward_valid == 0u || program->gradient_samples == 0u) {
        return GEO_V8_BACKWARD_REQUIRED;
    }
    if ((optimizer != GEO_V8_OPTIMIZER_SGD && optimizer != GEO_V8_OPTIMIZER_ADAM) ||
        !isfinite(learning_rate) || learning_rate <= 0.0 ||
        !isfinite(gradient_scale) || gradient_scale <= 0.0 ||
        !isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0 ||
        !isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0 ||
        !isfinite(epsilon) || epsilon <= 0.0) {
        return GEO_V8_INVALID_ARGUMENT;
    }

    candidate = (geo_v8_node_t *)malloc((size_t)program->node_count * sizeof(*candidate));
    if (candidate == NULL) return GEO_V8_ALLOCATION_FAILURE;
    memcpy(candidate, program->nodes, (size_t)program->node_count * sizeof(*candidate));

    step = program->optimizer_step + 1u;
    if (optimizer == GEO_V8_OPTIMIZER_ADAM) {
        bias1 = 1.0 - pow(beta1, (double)step);
        bias2 = 1.0 - pow(beta2, (double)step);
        if (!isfinite(bias1) || !isfinite(bias2) || bias1 <= 0.0 || bias2 <= 0.0) {
            free(candidate);
            return GEO_V8_NUMERIC_FAILURE;
        }
    }

    for (i = 0u; i < program->node_count; ++i) {
        geo_v8_node_t *node = &candidate[i];
        size_t blade;
        if (node->kind != GEO_V8_NODE_PARAMETER) continue;
        for (blade = 0u; blade < count; ++blade) {
            const double gradient = program->nodes[i].gradient.coefficients[blade] * gradient_scale;
            double update;
            if (optimizer == GEO_V8_OPTIMIZER_SGD) {
                update = gradient;
            } else {
                const double first = beta1 * node->first_moment[blade] +
                    (1.0 - beta1) * gradient;
                const double second = beta2 * node->second_moment[blade] +
                    (1.0 - beta2) * gradient * gradient;
                if (!isfinite(first) || !isfinite(second)) {
                    free(candidate);
                    return GEO_V8_NUMERIC_FAILURE;
                }
                node->first_moment[blade] = first;
                node->second_moment[blade] = second;
                update = (first / bias1) / (sqrt(second / bias2) + epsilon);
            }
            node->value.coefficients[blade] -= learning_rate * update;
            if (!isfinite(node->value.coefficients[blade])) {
                free(candidate);
                return GEO_V8_NUMERIC_FAILURE;
            }
        }
        if (v8_project_constraint(program, node) != GEO_V8_OK) {
            free(candidate);
            return GEO_V8_CONSTRAINT_FAILURE;
        }
    }

    for (i = 0u; i < program->node_count; ++i) {
        if (program->nodes[i].kind == GEO_V8_NODE_PARAMETER) {
            program->nodes[i].value = candidate[i].value;
            memcpy(
                program->nodes[i].first_moment,
                candidate[i].first_moment,
                sizeof(program->nodes[i].first_moment)
            );
            memcpy(
                program->nodes[i].second_moment,
                candidate[i].second_moment,
                sizeof(program->nodes[i].second_moment)
            );
        }
    }
    free(candidate);
    program->optimizer_step = step;
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    (void)geo_v8_zero_grad(program);
    return GEO_V8_OK;
}

geo_v8_status_t geo_v8_commit_states(geo_v8_program_t *program) {
    geo_operator_mv_f64_t *updates;
    uint8_t *present;
    geo_v8_node_id_t i;
    if (!v8_program_valid(program)) return GEO_V8_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V8_NOT_COMPILED;
    if (program->forward_valid == 0u) return GEO_V8_FORWARD_REQUIRED;

    updates = (geo_operator_mv_f64_t *)calloc(
        (size_t)program->node_count,
        sizeof(*updates)
    );
    present = (uint8_t *)calloc((size_t)program->node_count, sizeof(*present));
    if (updates == NULL || present == NULL) {
        free(updates);
        free(present);
        return GEO_V8_ALLOCATION_FAILURE;
    }

    for (i = 0u; i < program->node_count; ++i) {
        const geo_v8_node_t *node = &program->nodes[i];
        if (node->kind != GEO_V8_NODE_STATE) continue;
        if (node->state_update == GEO_V8_INVALID_NODE) {
            free(updates);
            free(present);
            return GEO_V8_STATE_UPDATE_REQUIRED;
        }
        updates[i] = program->nodes[node->state_update].value;
        present[i] = 1u;
    }
    for (i = 0u; i < program->node_count; ++i) {
        if (present[i] != 0u) program->nodes[i].value = updates[i];
    }
    free(updates);
    free(present);
    program->forward_valid = 0u;
    return GEO_V8_OK;
}

const geo_operator_mv_f64_t *geo_v8_value(
    const geo_v8_program_t *program,
    geo_v8_node_id_t node_id
) {
    if (!v8_node_exists(program, node_id)) return NULL;
    return &program->nodes[node_id].value;
}

const geo_operator_mv_f64_t *geo_v8_gradient(
    const geo_v8_program_t *program,
    geo_v8_node_id_t node_id
) {
    if (!v8_node_exists(program, node_id) || program->gradient_samples == 0u) {
        return NULL;
    }
    return &program->nodes[node_id].gradient;
}
