#include "geo/autodiff_v7.h"

#include <math.h>
#include <string.h>

static int geo_v7_dimension_valid(uint8_t dimension) {
    return dimension >= 1u && dimension <= GEO_OPERATOR_MAX_DIMENSION;
}

static int geo_v7_signature_valid(const int8_t *signature, uint8_t dimension) {
    uint8_t index;
    if (signature == NULL || !geo_v7_dimension_valid(dimension)) return 0;
    for (index = 0u; index < dimension; ++index) {
        if (signature[index] != 1 && signature[index] != -1) return 0;
    }
    return 1;
}

static int geo_v7_same_metric(
    uint8_t dimension,
    const int8_t *left,
    const int8_t *right
) {
    uint8_t index;
    if (left == NULL || right == NULL) return 0;
    for (index = 0u; index < dimension; ++index) {
        if (left[index] != right[index]) return 0;
    }
    return 1;
}

static size_t geo_v7_blade_count(uint8_t dimension) {
    return (size_t)1u << dimension;
}

static void geo_v7_mv_zero(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, dimension * sizeof(signature[0]));
}

static int geo_v7_mv_valid(
    const geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    size_t blade;
    const size_t blade_count = geo_v7_blade_count(dimension);
    if (value == NULL || value->dimension != dimension ||
        !geo_v7_same_metric(dimension, value->signature, signature)) {
        return 0;
    }
    for (blade = 0u; blade < blade_count; ++blade) {
        if (!isfinite(value->coefficients[blade])) return 0;
    }
    return 1;
}

static int geo_v7_program_header_valid(const geo_v7_program_t *program) {
    return program != NULL &&
        program->abi_version == GEO_V7_ABI_VERSION &&
        geo_v7_signature_valid(program->signature, program->dimension) &&
        program->pairing == GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN &&
        program->node_count <= GEO_V7_MAX_NODES;
}

static int geo_v7_node_exists(
    const geo_v7_program_t *program,
    geo_v7_node_id_t node_id
) {
    return geo_v7_program_header_valid(program) && node_id < program->node_count;
}

static uint8_t geo_v7_popcount(uint8_t value) {
    uint8_t count = 0u;
    while (value != 0u) {
        value = (uint8_t)(value & (uint8_t)(value - 1u));
        ++count;
    }
    return count;
}

static int geo_v7_reverse_sign(uint8_t blade) {
    const uint8_t grade = geo_v7_popcount(blade);
    return ((((uint8_t)(grade * (uint8_t)(grade - 1u) / 2u)) & 1u) != 0u) ? -1 : 1;
}

static geo_v7_status_t geo_v7_reserve_node(
    geo_v7_program_t *program,
    geo_v7_node_kind_t kind,
    geo_v7_node_id_t *node_id
) {
    geo_v7_node_t *node;
    if (program == NULL || node_id == NULL) return GEO_V7_INVALID_ARGUMENT;
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (program->compiled != 0u) return GEO_V7_BAD_PROGRAM;
    if (program->node_count >= GEO_V7_MAX_NODES) return GEO_V7_CAPACITY_EXCEEDED;
    *node_id = program->node_count;
    node = &program->nodes[program->node_count];
    memset(node, 0, sizeof(*node));
    node->kind = (uint8_t)kind;
    node->left = GEO_V7_INVALID_NODE;
    node->right = GEO_V7_INVALID_NODE;
    geo_v7_mv_zero(&node->value, program->dimension, program->signature);
    geo_v7_mv_zero(&node->cotangent, program->dimension, program->signature);
    ++program->node_count;
    return GEO_V7_OK;
}

static geo_v7_status_t geo_v7_add_leaf(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *value,
    geo_v7_node_kind_t kind,
    int requires_grad,
    geo_v7_node_id_t *node_id
) {
    geo_v7_status_t status;
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (!geo_v7_mv_valid(value, program->dimension, program->signature)) {
        return GEO_V7_INVALID_ARGUMENT;
    }
    status = geo_v7_reserve_node(program, kind, node_id);
    if (status != GEO_V7_OK) return status;
    program->nodes[*node_id].requires_grad = requires_grad != 0 ? 1u : 0u;
    program->nodes[*node_id].value = *value;
    return GEO_V7_OK;
}

static geo_v7_status_t geo_v7_add_unary(
    geo_v7_program_t *program,
    geo_v7_node_kind_t kind,
    geo_v7_node_id_t input,
    uint8_t grade,
    double scalar,
    geo_v7_node_id_t *node_id
) {
    geo_v7_status_t status;
    if (!geo_v7_node_exists(program, input) || node_id == NULL) return GEO_V7_BAD_NODE;
    status = geo_v7_reserve_node(program, kind, node_id);
    if (status != GEO_V7_OK) return status;
    program->nodes[*node_id].left = input;
    program->nodes[*node_id].grade = grade;
    program->nodes[*node_id].scalar = scalar;
    program->nodes[*node_id].requires_grad = program->nodes[input].requires_grad;
    return GEO_V7_OK;
}

static geo_v7_status_t geo_v7_add_binary(
    geo_v7_program_t *program,
    geo_v7_node_kind_t kind,
    geo_v7_node_id_t left,
    geo_v7_node_id_t right,
    geo_v7_node_id_t *node_id
) {
    geo_v7_status_t status;
    if (!geo_v7_node_exists(program, left) || !geo_v7_node_exists(program, right) ||
        node_id == NULL) {
        return GEO_V7_BAD_NODE;
    }
    status = geo_v7_reserve_node(program, kind, node_id);
    if (status != GEO_V7_OK) return status;
    program->nodes[*node_id].left = left;
    program->nodes[*node_id].right = right;
    program->nodes[*node_id].requires_grad =
        (program->nodes[left].requires_grad != 0u || program->nodes[right].requires_grad != 0u) ? 1u : 0u;
    return GEO_V7_OK;
}

static int geo_v7_loss_is_scalar(const geo_v7_program_t *program) {
    size_t blade;
    const geo_operator_mv_f64_t *loss;
    const size_t blade_count = geo_v7_blade_count(program->dimension);
    if (!geo_v7_node_exists(program, program->loss_node)) return 0;
    loss = &program->nodes[program->loss_node].value;
    for (blade = 1u; blade < blade_count; ++blade) {
        if (loss->coefficients[blade] != 0.0) return 0;
    }
    return isfinite(loss->coefficients[0]);
}

static geo_v7_status_t geo_v7_accumulate_scaled(
    geo_operator_mv_f64_t *destination,
    const geo_operator_mv_f64_t *source,
    double scale,
    size_t blade_count
) {
    size_t blade;
    for (blade = 0u; blade < blade_count; ++blade) {
        const double next = destination->coefficients[blade] + source->coefficients[blade] * scale;
        if (!isfinite(next)) return GEO_V7_NUMERIC_FAILURE;
        destination->coefficients[blade] = next;
    }
    return GEO_V7_OK;
}

static geo_v7_status_t geo_v7_reverse_apply(
    const geo_operator_mv_f64_t *input,
    geo_operator_mv_f64_t *output,
    uint8_t dimension,
    const int8_t *signature
) {
    size_t blade;
    const size_t blade_count = geo_v7_blade_count(dimension);
    geo_v7_mv_zero(output, dimension, signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        output->coefficients[blade] = input->coefficients[blade] * (double)geo_v7_reverse_sign((uint8_t)blade);
        if (!isfinite(output->coefficients[blade])) return GEO_V7_NUMERIC_FAILURE;
    }
    return GEO_V7_OK;
}

static geo_v7_status_t geo_v7_grade_project_apply(
    const geo_operator_mv_f64_t *input,
    uint8_t grade,
    geo_operator_mv_f64_t *output,
    uint8_t dimension,
    const int8_t *signature
) {
    size_t blade;
    const size_t blade_count = geo_v7_blade_count(dimension);
    geo_v7_mv_zero(output, dimension, signature);
    for (blade = 0u; blade < blade_count; ++blade) {
        if (geo_v7_popcount((uint8_t)blade) == grade) {
            output->coefficients[blade] = input->coefficients[blade];
        }
    }
    return GEO_V7_OK;
}

uint32_t geo_v7_abi_version(void) {
    return GEO_V7_ABI_VERSION;
}

size_t geo_v7_program_bytes(void) {
    return sizeof(geo_v7_program_t);
}

geo_v7_status_t geo_v7_program_init(
    geo_v7_program_t *program,
    uint8_t dimension,
    const int8_t *signature,
    geo_v7_pairing_t pairing
) {
    if (program == NULL || signature == NULL) return GEO_V7_INVALID_ARGUMENT;
    if (!geo_v7_dimension_valid(dimension)) return GEO_V7_UNSUPPORTED_DIMENSION;
    if (!geo_v7_signature_valid(signature, dimension) ||
        pairing != GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN) {
        return GEO_V7_INVALID_ARGUMENT;
    }
    memset(program, 0, sizeof(*program));
    program->abi_version = GEO_V7_ABI_VERSION;
    program->dimension = dimension;
    memcpy(program->signature, signature, dimension * sizeof(signature[0]));
    program->pairing = (uint8_t)pairing;
    program->loss_node = GEO_V7_INVALID_NODE;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_add_input(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *initial_value,
    int requires_grad,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_leaf(
        program,
        initial_value,
        GEO_V7_NODE_INPUT,
        requires_grad,
        node_id
    );
}

geo_v7_status_t geo_v7_add_parameter(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *initial_value,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_leaf(program, initial_value, GEO_V7_NODE_PARAMETER, 1, node_id);
}

geo_v7_status_t geo_v7_add_constant(
    geo_v7_program_t *program,
    const geo_operator_mv_f64_t *value,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_leaf(program, value, GEO_V7_NODE_CONSTANT, 0, node_id);
}

geo_v7_status_t geo_v7_add_add(
    geo_v7_program_t *program,
    geo_v7_node_id_t left,
    geo_v7_node_id_t right,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_binary(program, GEO_V7_NODE_ADD, left, right, node_id);
}

geo_v7_status_t geo_v7_add_scale(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    double scalar,
    geo_v7_node_id_t *node_id
) {
    if (!isfinite(scalar)) return GEO_V7_INVALID_ARGUMENT;
    return geo_v7_add_unary(program, GEO_V7_NODE_SCALE, input, 0u, scalar, node_id);
}

geo_v7_status_t geo_v7_add_geometric_product(
    geo_v7_program_t *program,
    geo_v7_node_id_t left,
    geo_v7_node_id_t right,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_binary(
        program,
        GEO_V7_NODE_GEOMETRIC_PRODUCT,
        left,
        right,
        node_id
    );
}

geo_v7_status_t geo_v7_add_reverse(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_unary(program, GEO_V7_NODE_REVERSE, input, 0u, 0.0, node_id);
}

geo_v7_status_t geo_v7_add_grade_project(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    uint8_t grade,
    geo_v7_node_id_t *node_id
) {
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (grade > program->dimension) return GEO_V7_INVALID_ARGUMENT;
    return geo_v7_add_unary(
        program,
        GEO_V7_NODE_GRADE_PROJECT,
        input,
        grade,
        0.0,
        node_id
    );
}

geo_v7_status_t geo_v7_add_squared_norm(
    geo_v7_program_t *program,
    geo_v7_node_id_t input,
    geo_v7_node_id_t *node_id
) {
    return geo_v7_add_unary(
        program,
        GEO_V7_NODE_SQUARED_NORM,
        input,
        0u,
        0.0,
        node_id
    );
}

geo_v7_status_t geo_v7_compile(
    geo_v7_program_t *program,
    geo_v7_node_id_t loss_node
) {
    geo_v7_node_id_t index;
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (program->node_count == 0u || loss_node >= program->node_count) return GEO_V7_BAD_NODE;
    for (index = 0u; index < program->node_count; ++index) {
        const geo_v7_node_t *node = &program->nodes[index];
        switch ((geo_v7_node_kind_t)node->kind) {
            case GEO_V7_NODE_INPUT:
            case GEO_V7_NODE_PARAMETER:
            case GEO_V7_NODE_CONSTANT:
                if (node->left != GEO_V7_INVALID_NODE || node->right != GEO_V7_INVALID_NODE ||
                    !geo_v7_mv_valid(&node->value, program->dimension, program->signature)) {
                    return GEO_V7_BAD_PROGRAM;
                }
                break;
            case GEO_V7_NODE_ADD:
            case GEO_V7_NODE_GEOMETRIC_PRODUCT:
                if (node->left >= index || node->right >= index) return GEO_V7_BAD_PROGRAM;
                break;
            case GEO_V7_NODE_SCALE:
                if (node->left >= index || node->right != GEO_V7_INVALID_NODE ||
                    !isfinite(node->scalar)) {
                    return GEO_V7_BAD_PROGRAM;
                }
                break;
            case GEO_V7_NODE_REVERSE:
            case GEO_V7_NODE_SQUARED_NORM:
                if (node->left >= index || node->right != GEO_V7_INVALID_NODE) {
                    return GEO_V7_BAD_PROGRAM;
                }
                break;
            case GEO_V7_NODE_GRADE_PROJECT:
                if (node->left >= index || node->right != GEO_V7_INVALID_NODE ||
                    node->grade > program->dimension) {
                    return GEO_V7_BAD_PROGRAM;
                }
                break;
            default:
                return GEO_V7_BAD_PROGRAM;
        }
    }
    program->compiled = 1u;
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    program->loss_node = loss_node;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_set_value(
    geo_v7_program_t *program,
    geo_v7_node_id_t node_id,
    const geo_operator_mv_f64_t *value
) {
    geo_v7_node_t *node;
    if (!geo_v7_node_exists(program, node_id)) return GEO_V7_BAD_NODE;
    if (!geo_v7_mv_valid(value, program->dimension, program->signature)) {
        return GEO_V7_INVALID_ARGUMENT;
    }
    node = &program->nodes[node_id];
    if (node->kind != GEO_V7_NODE_INPUT && node->kind != GEO_V7_NODE_PARAMETER &&
        node->kind != GEO_V7_NODE_CONSTANT) {
        return GEO_V7_BAD_NODE;
    }
    node->value = *value;
    if (node->kind == GEO_V7_NODE_PARAMETER) {
        memset(node->first_moment, 0, sizeof(node->first_moment));
        memset(node->second_moment, 0, sizeof(node->second_moment));
        program->optimizer_step = 0u;
    }
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_forward(geo_v7_program_t *program) {
    geo_v7_node_id_t index;
    const size_t blade_count = program == NULL ? 0u : geo_v7_blade_count(program->dimension);
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V7_NOT_COMPILED;
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    for (index = 0u; index < program->node_count; ++index) {
        geo_v7_node_t *node = &program->nodes[index];
        geo_operator_mv_f64_t result;
        size_t blade;
        geo_v7_mv_zero(&result, program->dimension, program->signature);
        switch ((geo_v7_node_kind_t)node->kind) {
            case GEO_V7_NODE_INPUT:
            case GEO_V7_NODE_PARAMETER:
            case GEO_V7_NODE_CONSTANT:
                if (!geo_v7_mv_valid(&node->value, program->dimension, program->signature)) {
                    return GEO_V7_NUMERIC_FAILURE;
                }
                continue;
            case GEO_V7_NODE_ADD:
                for (blade = 0u; blade < blade_count; ++blade) {
                    result.coefficients[blade] =
                        program->nodes[node->left].value.coefficients[blade] +
                        program->nodes[node->right].value.coefficients[blade];
                }
                break;
            case GEO_V7_NODE_SCALE:
                for (blade = 0u; blade < blade_count; ++blade) {
                    result.coefficients[blade] =
                        program->nodes[node->left].value.coefficients[blade] * node->scalar;
                }
                break;
            case GEO_V7_NODE_GEOMETRIC_PRODUCT:
                if (geo_operator_gp_f64(
                        &program->nodes[node->left].value,
                        &program->nodes[node->right].value,
                        &result
                    ) != GEO_OPERATOR_OK) {
                    return GEO_V7_BAD_PROGRAM;
                }
                break;
            case GEO_V7_NODE_REVERSE:
                if (geo_v7_reverse_apply(
                        &program->nodes[node->left].value,
                        &result,
                        program->dimension,
                        program->signature
                    ) != GEO_V7_OK) {
                    return GEO_V7_NUMERIC_FAILURE;
                }
                break;
            case GEO_V7_NODE_GRADE_PROJECT:
                (void)geo_v7_grade_project_apply(
                    &program->nodes[node->left].value,
                    node->grade,
                    &result,
                    program->dimension,
                    program->signature
                );
                break;
            case GEO_V7_NODE_SQUARED_NORM:
                for (blade = 0u; blade < blade_count; ++blade) {
                    const double coefficient = program->nodes[node->left].value.coefficients[blade];
                    result.coefficients[0] += 0.5 * coefficient * coefficient;
                }
                break;
            default:
                return GEO_V7_BAD_PROGRAM;
        }
        if (!geo_v7_mv_valid(&result, program->dimension, program->signature)) {
            return GEO_V7_NUMERIC_FAILURE;
        }
        node->value = result;
    }
    program->forward_valid = 1u;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_zero_grad(geo_v7_program_t *program) {
    geo_v7_node_id_t index;
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    for (index = 0u; index < program->node_count; ++index) {
        geo_v7_mv_zero(
            &program->nodes[index].cotangent,
            program->dimension,
            program->signature
        );
    }
    program->backward_valid = 0u;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_backward(geo_v7_program_t *program) {
    geo_v7_node_id_t cursor;
    const size_t blade_count = program == NULL ? 0u : geo_v7_blade_count(program->dimension);
    geo_v7_status_t status;
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V7_NOT_COMPILED;
    if (program->forward_valid == 0u) return GEO_V7_FORWARD_REQUIRED;
    if (!geo_v7_loss_is_scalar(program)) return GEO_V7_NON_SCALAR_LOSS;
    status = geo_v7_zero_grad(program);
    if (status != GEO_V7_OK) return status;
    program->nodes[program->loss_node].cotangent.coefficients[0] = 1.0;
    cursor = program->node_count;
    while (cursor > 0u) {
        geo_v7_node_t *node;
        --cursor;
        node = &program->nodes[cursor];
        if (node->requires_grad == 0u) continue;
        if (!geo_v7_mv_valid(&node->cotangent, program->dimension, program->signature)) {
            return GEO_V7_NUMERIC_FAILURE;
        }
        switch ((geo_v7_node_kind_t)node->kind) {
            case GEO_V7_NODE_INPUT:
            case GEO_V7_NODE_PARAMETER:
            case GEO_V7_NODE_CONSTANT:
                break;
            case GEO_V7_NODE_ADD:
                if (program->nodes[node->left].requires_grad != 0u) {
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->left].cotangent,
                        &node->cotangent,
                        1.0,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                if (program->nodes[node->right].requires_grad != 0u) {
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->right].cotangent,
                        &node->cotangent,
                        1.0,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                break;
            case GEO_V7_NODE_SCALE:
                if (program->nodes[node->left].requires_grad != 0u) {
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->left].cotangent,
                        &node->cotangent,
                        node->scalar,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                break;
            case GEO_V7_NODE_GEOMETRIC_PRODUCT: {
                geo_operator_mv_f64_t left_cotangent;
                geo_operator_mv_f64_t right_cotangent;
                if (geo_operator_gp_f64_vjp(
                        &program->nodes[node->left].value,
                        &program->nodes[node->right].value,
                        &node->cotangent,
                        &left_cotangent,
                        &right_cotangent
                    ) != GEO_OPERATOR_OK) {
                    return GEO_V7_BAD_PROGRAM;
                }
                if (program->nodes[node->left].requires_grad != 0u) {
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->left].cotangent,
                        &left_cotangent,
                        1.0,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                if (program->nodes[node->right].requires_grad != 0u) {
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->right].cotangent,
                        &right_cotangent,
                        1.0,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                break;
            }
            case GEO_V7_NODE_REVERSE:
                if (program->nodes[node->left].requires_grad != 0u) {
                    geo_operator_mv_f64_t reversed;
                    status = geo_v7_reverse_apply(
                        &node->cotangent,
                        &reversed,
                        program->dimension,
                        program->signature
                    );
                    if (status != GEO_V7_OK) return status;
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->left].cotangent,
                        &reversed,
                        1.0,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                break;
            case GEO_V7_NODE_GRADE_PROJECT:
                if (program->nodes[node->left].requires_grad != 0u) {
                    geo_operator_mv_f64_t projected;
                    status = geo_v7_grade_project_apply(
                        &node->cotangent,
                        node->grade,
                        &projected,
                        program->dimension,
                        program->signature
                    );
                    if (status != GEO_V7_OK) return status;
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->left].cotangent,
                        &projected,
                        1.0,
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                break;
            case GEO_V7_NODE_SQUARED_NORM:
                if (program->nodes[node->left].requires_grad != 0u) {
                    size_t blade;
                    for (blade = 1u; blade < blade_count; ++blade) {
                        if (node->cotangent.coefficients[blade] != 0.0) {
                            return GEO_V7_NON_SCALAR_LOSS;
                        }
                    }
                    status = geo_v7_accumulate_scaled(
                        &program->nodes[node->left].cotangent,
                        &program->nodes[node->left].value,
                        node->cotangent.coefficients[0],
                        blade_count
                    );
                    if (status != GEO_V7_OK) return status;
                }
                break;
            default:
                return GEO_V7_BAD_PROGRAM;
        }
    }
    program->backward_valid = 1u;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_sgd_step(
    geo_v7_program_t *program,
    double learning_rate
) {
    geo_v7_node_id_t index;
    const size_t blade_count = program == NULL ? 0u : geo_v7_blade_count(program->dimension);
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V7_NOT_COMPILED;
    if (program->backward_valid == 0u) return GEO_V7_BACKWARD_REQUIRED;
    if (!isfinite(learning_rate) || learning_rate <= 0.0) return GEO_V7_INVALID_ARGUMENT;
    for (index = 0u; index < program->node_count; ++index) {
        const geo_v7_node_t *node = &program->nodes[index];
        size_t blade;
        if (node->kind != GEO_V7_NODE_PARAMETER) continue;
        for (blade = 0u; blade < blade_count; ++blade) {
            const double candidate =
                node->value.coefficients[blade] - learning_rate * node->cotangent.coefficients[blade];
            if (!isfinite(candidate)) return GEO_V7_NUMERIC_FAILURE;
        }
    }
    for (index = 0u; index < program->node_count; ++index) {
        geo_v7_node_t *node = &program->nodes[index];
        size_t blade;
        if (node->kind != GEO_V7_NODE_PARAMETER) continue;
        for (blade = 0u; blade < blade_count; ++blade) {
            node->value.coefficients[blade] -= learning_rate * node->cotangent.coefficients[blade];
        }
    }
    ++program->optimizer_step;
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    return GEO_V7_OK;
}

geo_v7_status_t geo_v7_adam_step(
    geo_v7_program_t *program,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
) {
    geo_v7_node_id_t index;
    uint64_t step;
    double beta1_power;
    double beta2_power;
    double bias1;
    double bias2;
    const size_t blade_count = program == NULL ? 0u : geo_v7_blade_count(program->dimension);
    if (!geo_v7_program_header_valid(program)) return GEO_V7_BAD_PROGRAM;
    if (program->compiled == 0u) return GEO_V7_NOT_COMPILED;
    if (program->backward_valid == 0u) return GEO_V7_BACKWARD_REQUIRED;
    if (!isfinite(learning_rate) || learning_rate <= 0.0 ||
        !isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0 ||
        !isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0 ||
        !isfinite(epsilon) || epsilon <= 0.0) {
        return GEO_V7_INVALID_ARGUMENT;
    }
    step = program->optimizer_step + 1u;
    beta1_power = pow(beta1, (double)step);
    beta2_power = pow(beta2, (double)step);
    bias1 = 1.0 - beta1_power;
    bias2 = 1.0 - beta2_power;
    if (!isfinite(bias1) || !isfinite(bias2) || bias1 <= 0.0 || bias2 <= 0.0) {
        return GEO_V7_NUMERIC_FAILURE;
    }
    for (index = 0u; index < program->node_count; ++index) {
        const geo_v7_node_t *node = &program->nodes[index];
        size_t blade;
        if (node->kind != GEO_V7_NODE_PARAMETER) continue;
        for (blade = 0u; blade < blade_count; ++blade) {
            const double gradient = node->cotangent.coefficients[blade];
            const double first = beta1 * node->first_moment[blade] + (1.0 - beta1) * gradient;
            const double second = beta2 * node->second_moment[blade] +
                (1.0 - beta2) * gradient * gradient;
            const double update = (first / bias1) / (sqrt(second / bias2) + epsilon);
            const double candidate = node->value.coefficients[blade] - learning_rate * update;
            if (!isfinite(first) || !isfinite(second) || !isfinite(update) ||
                !isfinite(candidate)) {
                return GEO_V7_NUMERIC_FAILURE;
            }
        }
    }
    for (index = 0u; index < program->node_count; ++index) {
        geo_v7_node_t *node = &program->nodes[index];
        size_t blade;
        if (node->kind != GEO_V7_NODE_PARAMETER) continue;
        for (blade = 0u; blade < blade_count; ++blade) {
            const double gradient = node->cotangent.coefficients[blade];
            const double first = beta1 * node->first_moment[blade] + (1.0 - beta1) * gradient;
            const double second = beta2 * node->second_moment[blade] +
                (1.0 - beta2) * gradient * gradient;
            const double update = (first / bias1) / (sqrt(second / bias2) + epsilon);
            node->first_moment[blade] = first;
            node->second_moment[blade] = second;
            node->value.coefficients[blade] -= learning_rate * update;
        }
    }
    program->optimizer_step = step;
    program->forward_valid = 0u;
    program->backward_valid = 0u;
    return GEO_V7_OK;
}

const geo_operator_mv_f64_t *geo_v7_value(
    const geo_v7_program_t *program,
    geo_v7_node_id_t node_id
) {
    if (!geo_v7_node_exists(program, node_id)) return NULL;
    return &program->nodes[node_id].value;
}

const geo_operator_mv_f64_t *geo_v7_gradient(
    const geo_v7_program_t *program,
    geo_v7_node_id_t node_id
) {
    if (!geo_v7_node_exists(program, node_id) || program->backward_valid == 0u) return NULL;
    return &program->nodes[node_id].cotangent;
}
