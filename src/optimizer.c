#include "geo/optimizer.h"

#include <limits.h>
#include <string.h>

static size_t geo_count_omega_nodes(const geo_witness_tree_t *tree) {
    size_t count = 0u;
    size_t index;
    for (index = 0u; index < tree->node_count; ++index) {
        if ((geo_witness_kind_t)tree->nodes[index].kind == GEO_WITNESS_OMEGA) ++count;
    }
    return count;
}

static geo_status_t geo_mark_live_lanes(const geo_witness_tree_t *tree, uint8_t *live_lanes) {
    size_t reverse_index;
    memset(live_lanes, 0, tree->node_count * sizeof(live_lanes[0]));
    live_lanes[tree->root] =
        (geo_witness_kind_t)tree->nodes[tree->root].kind == GEO_WITNESS_TERMINAL
            ? GEO_LANE_ALL
            : tree->nodes[tree->root].requested_lanes;

    reverse_index = tree->node_count;
    while (reverse_index > 0u) {
        const size_t index = reverse_index - 1u;
        const geo_witness_node_t node = tree->nodes[index];
        const uint8_t required = live_lanes[index];
        if (required != GEO_LANE_NONE &&
            (geo_witness_kind_t)node.kind == GEO_WITNESS_OMEGA) {
            const uint8_t propagated = (uint8_t)(required & node.requested_lanes);
            live_lanes[node.left] = (uint8_t)(live_lanes[node.left] | propagated);
            live_lanes[node.right] = (uint8_t)(live_lanes[node.right] | propagated);
        }
        --reverse_index;
    }
    return GEO_STATUS_OK;
}

static int geo_nodes_equivalent(
    const geo_witness_tree_t *tree,
    const geo_optimizer_workspace_t *workspace,
    size_t left_index,
    size_t right_index
) {
    const geo_witness_node_t left = tree->nodes[left_index];
    const geo_witness_node_t right = tree->nodes[right_index];
    const uint8_t left_lanes = (uint8_t)(left.requested_lanes & workspace->live_lanes[left_index]);
    const uint8_t right_lanes = (uint8_t)(right.requested_lanes & workspace->live_lanes[right_index]);

    if ((geo_witness_kind_t)left.kind != GEO_WITNESS_OMEGA ||
        (geo_witness_kind_t)right.kind != GEO_WITNESS_OMEGA) return 0;

    return left_lanes == right_lanes &&
        workspace->representatives[left.left] == workspace->representatives[right.left] &&
        workspace->representatives[left.right] == workspace->representatives[right.right];
}

geo_status_t geo_witness_compile_optimized(
    const geo_witness_tree_t *tree,
    geo_optimizer_workspace_t *workspace,
    geo_optimized_witness_t *output
) {
    size_t index;
    size_t instruction_index = 0u;
    size_t next_register;
    size_t dead_nodes = 0u;
    size_t duplicate_nodes = 0u;
    const size_t original_instruction_count = tree == NULL ? 0u : geo_count_omega_nodes(tree);
    geo_status_t status;

    if (tree == NULL || workspace == NULL || output == NULL ||
        workspace->instructions == NULL || workspace->node_registers == NULL ||
        workspace->live_lanes == NULL || workspace->representatives == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    status = geo_witness_validate(tree);
    if (status != GEO_STATUS_OK) return status;

    if (workspace->node_register_capacity < tree->node_count ||
        workspace->live_lane_capacity < tree->node_count ||
        workspace->representative_capacity < tree->node_count) {
        return GEO_STATUS_BUFFER_CAPACITY;
    }

    status = geo_mark_live_lanes(tree, workspace->live_lanes);
    if (status != GEO_STATUS_OK) return status;

    next_register = tree->terminal_count;
    for (index = 0u; index < tree->node_count; ++index) {
        const geo_witness_node_t node = tree->nodes[index];
        workspace->representatives[index] = (uint16_t)index;

        if (workspace->live_lanes[index] == GEO_LANE_NONE) {
            workspace->node_registers[index] = 0u;
            ++dead_nodes;
            continue;
        }

        if ((geo_witness_kind_t)node.kind == GEO_WITNESS_TERMINAL) {
            workspace->node_registers[index] = (uint8_t)node.terminal_index;
            continue;
        }

        {
            size_t prior;
            int duplicate_found = 0;
            for (prior = 0u; prior < index; ++prior) {
                if (workspace->live_lanes[prior] != GEO_LANE_NONE &&
                    geo_nodes_equivalent(tree, workspace, prior, index)) {
                    workspace->representatives[index] = workspace->representatives[prior];
                    workspace->node_registers[index] = workspace->node_registers[prior];
                    duplicate_found = 1;
                    ++duplicate_nodes;
                    break;
                }
            }
            if (duplicate_found) continue;
        }

        if (instruction_index >= workspace->instruction_capacity) return GEO_STATUS_BUFFER_CAPACITY;
        if (next_register > (size_t)UINT8_MAX) return GEO_STATUS_REGISTER_RANGE;

        {
            geo_instruction_t *instruction = &workspace->instructions[instruction_index];
            const uint8_t lanes = (uint8_t)(node.requested_lanes & workspace->live_lanes[index]);
            instruction->opcode = (uint8_t)GEO_OPCODE_OMEGA;
            instruction->destination = (uint8_t)next_register;
            instruction->left = workspace->node_registers[node.left];
            instruction->right = workspace->node_registers[node.right];
            instruction->requested_lanes = lanes;
            workspace->node_registers[index] = (uint8_t)next_register;
            ++instruction_index;
            ++next_register;
        }
    }

    if (next_register > (size_t)UINT8_MAX + 1u) return GEO_STATUS_REGISTER_RANGE;

    output->program.instructions = workspace->instructions;
    output->program.instruction_count = instruction_index;
    output->program.register_count = next_register;
    output->root_register = workspace->node_registers[tree->root];
    output->original_instruction_count = original_instruction_count;
    output->optimized_instruction_count = instruction_index;
    output->eliminated_dead_nodes = dead_nodes;
    output->eliminated_duplicate_nodes = duplicate_nodes;
    return GEO_STATUS_OK;
}
