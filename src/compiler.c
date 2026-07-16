#include "geo/compiler.h"

#include <limits.h>

static size_t geo_witness_instruction_count(const geo_witness_tree_t *tree) {
    size_t count = 0;
    size_t index;

    for (index = 0; index < tree->node_count; ++index) {
        if ((geo_witness_kind_t)tree->nodes[index].kind == GEO_WITNESS_OMEGA) {
            ++count;
        }
    }

    return count;
}

geo_status_t geo_witness_validate(const geo_witness_tree_t *tree) {
    size_t index;

    if (tree == NULL || tree->nodes == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (tree->node_count == 0 || tree->root >= tree->node_count) {
        return GEO_STATUS_BAD_TREE;
    }

    if (tree->terminal_count > (size_t)UINT8_MAX + 1u) {
        return GEO_STATUS_REGISTER_RANGE;
    }

    for (index = 0; index < tree->node_count; ++index) {
        const geo_witness_node_t node = tree->nodes[index];

        switch ((geo_witness_kind_t)node.kind) {
            case GEO_WITNESS_TERMINAL:
                if (node.terminal_index >= tree->terminal_count) {
                    return GEO_STATUS_BAD_TREE;
                }
                break;

            case GEO_WITNESS_OMEGA:
                if (node.left >= index || node.right >= index) {
                    return GEO_STATUS_BAD_TREE;
                }
                if ((node.requested_lanes & (uint8_t)(~GEO_LANE_ALL)) != 0u ||
                    node.requested_lanes == GEO_LANE_NONE) {
                    return GEO_STATUS_BAD_TREE;
                }
                break;

            default:
                return GEO_STATUS_BAD_TREE;
        }
    }

    return GEO_STATUS_OK;
}

geo_status_t geo_witness_compile(
    const geo_witness_tree_t *tree,
    geo_compile_workspace_t *workspace,
    geo_compiled_witness_t *output
) {
    size_t index;
    size_t instruction_index = 0;
    size_t next_register;
    const size_t instruction_count =
        tree == NULL ? 0u : geo_witness_instruction_count(tree);
    geo_status_t status;

    if (tree == NULL || workspace == NULL || output == NULL ||
        workspace->instructions == NULL || workspace->node_registers == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    status = geo_witness_validate(tree);
    if (status != GEO_STATUS_OK) {
        return status;
    }

    if (workspace->instruction_capacity < instruction_count ||
        workspace->node_register_capacity < tree->node_count) {
        return GEO_STATUS_BUFFER_CAPACITY;
    }

    if (tree->terminal_count + instruction_count > (size_t)UINT8_MAX + 1u) {
        return GEO_STATUS_REGISTER_RANGE;
    }

    next_register = tree->terminal_count;

    for (index = 0; index < tree->node_count; ++index) {
        const geo_witness_node_t node = tree->nodes[index];

        if ((geo_witness_kind_t)node.kind == GEO_WITNESS_TERMINAL) {
            workspace->node_registers[index] = (uint8_t)node.terminal_index;
        } else {
            geo_instruction_t *instruction =
                &workspace->instructions[instruction_index];

            instruction->opcode = (uint8_t)GEO_OPCODE_OMEGA;
            instruction->destination = (uint8_t)next_register;
            instruction->left = workspace->node_registers[node.left];
            instruction->right = workspace->node_registers[node.right];
            instruction->requested_lanes = node.requested_lanes;

            workspace->node_registers[index] = (uint8_t)next_register;
            ++next_register;
            ++instruction_index;
        }
    }

    output->program.instructions = workspace->instructions;
    output->program.instruction_count = instruction_count;
    output->program.register_count = next_register;
    output->root_register = workspace->node_registers[tree->root];

    return GEO_STATUS_OK;
}
