#include "geo_cycle_internal.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void geo_cycle_usage(FILE *stream) {
    fprintf(
        stream,
        "geo_cycle %s\n"
        "GEO V8 arbitrary-graph full-cycle runner.\n\n"
        "Commands:\n"
        "  geo_cycle check <model.geoir>\n"
        "  geo_cycle train <model.geoir> <train.csv> <checkpoint> [resume-checkpoint]\n"
        "  geo_cycle predict <model.geoir> <checkpoint> <input.csv> <output.csv>\n"
        "  geo_cycle export-c <model.geoir> <checkpoint> <header.h> <symbol>\n",
        GEO_CYCLE_VERSION
    );
}

char *geo_cycle_trim(char *text) {
    char *end;
    while (*text != '\0' && isspace((unsigned char)*text)) ++text;
    end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return text;
}

int geo_cycle_read_line(FILE *file, geo_cycle_line_t *line) {
    size_t length = 0u;
    int ch;
    if (line->data == NULL) {
        line->capacity = 4096u;
        line->data = (char *)malloc(line->capacity);
        if (line->data == NULL) return -1;
    }
    while ((ch = fgetc(file)) != EOF) {
        if (length + 1u >= line->capacity) {
            char *replacement;
            const size_t next = line->capacity * 2u;
            if (next <= line->capacity) return -1;
            replacement = (char *)realloc(line->data, next);
            if (replacement == NULL) return -1;
            line->data = replacement;
            line->capacity = next;
        }
        if (ch == '\n') break;
        line->data[length++] = (char)ch;
    }
    if (ch == EOF && length == 0u) return 0;
    line->data[length] = '\0';
    return 1;
}

void geo_cycle_line_free(geo_cycle_line_t *line) {
    free(line->data);
    memset(line, 0, sizeof(*line));
}

int geo_cycle_parse_size(const char *text, size_t *output) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *geo_cycle_trim(end) != '\0' ||
        value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *output = (size_t)value;
    return 1;
}

int geo_cycle_parse_u64(const char *text, uint64_t *output) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *geo_cycle_trim(end) != '\0') return 0;
    *output = (uint64_t)value;
    return 1;
}

int geo_cycle_parse_double(const char *text, double *output) {
    char *end = NULL;
    double value;
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *geo_cycle_trim(end) != '\0' || !isfinite(value)) {
        return 0;
    }
    *output = value;
    return 1;
}

int geo_cycle_parse_bool(const char *text, int *output) {
    if (strcmp(text, "0") == 0) {
        *output = 0;
        return 1;
    }
    if (strcmp(text, "1") == 0) {
        *output = 1;
        return 1;
    }
    return 0;
}

int geo_cycle_parse_signature(const char *text, uint8_t dimension, int8_t *signature) {
    char buffer[128];
    char *token;
    uint8_t index = 0u;
    if (strlen(text) >= sizeof(buffer)) return 0;
    strcpy(buffer, text);
    token = strtok(buffer, ",");
    while (token != NULL) {
        char *trimmed = geo_cycle_trim(token);
        long value;
        char *end = NULL;
        if (index >= dimension) return 0;
        errno = 0;
        value = strtol(trimmed, &end, 10);
        if (errno != 0 || end == trimmed || *geo_cycle_trim(end) != '\0' ||
            (value != 1 && value != -1)) {
            return 0;
        }
        signature[index++] = (int8_t)value;
        token = strtok(NULL, ",");
    }
    return index == dimension;
}

int geo_cycle_parse_vector(const char *text, double *output, size_t count) {
    const char *cursor = text;
    size_t index = 0u;
    while (*cursor != '\0') {
        char *end = NULL;
        double value;
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) ++cursor;
        errno = 0;
        value = strtod(cursor, &end);
        if (errno != 0 || end == cursor || !isfinite(value) || index >= count) return 0;
        output[index++] = value;
        cursor = end;
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == ',') ++cursor;
        else if (*cursor != '\0') return 0;
    }
    return index == count;
}

int geo_cycle_parse_csv_row(char *line, double *values, size_t expected) {
    char *cursor = line;
    size_t count = 0u;
    while (*cursor != '\0') {
        char *end = NULL;
        double value;
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == '\0' || *cursor == '#') break;
        errno = 0;
        value = strtod(cursor, &end);
        if (errno != 0 || end == cursor || !isfinite(value) || count >= expected) return 0;
        values[count++] = value;
        cursor = end;
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor == ',') ++cursor;
        else if (*cursor != '\0' && *cursor != '#') return 0;
    }
    return count == expected;
}

void geo_cycle_mv_init(
    geo_operator_mv_f64_t *value,
    uint8_t dimension,
    const int8_t *signature
) {
    memset(value, 0, sizeof(*value));
    value->dimension = dimension;
    memcpy(value->signature, signature, (size_t)dimension * sizeof(signature[0]));
}

static int cycle_split_node(
    char *text,
    size_t *index,
    int *kind,
    char **name,
    geo_v8_node_id_t *left,
    geo_v8_node_id_t *right,
    double *scalar,
    uint8_t *grade,
    int *constraint,
    int *requires_grad
) {
    char *fields[9];
    size_t field_count = 0u;
    char *token = strtok(text, ",");
    size_t parsed;
    while (token != NULL && field_count < 9u) {
        fields[field_count++] = geo_cycle_trim(token);
        token = strtok(NULL, ",");
    }
    if (field_count != 9u || token != NULL) return 0;
    if (!geo_cycle_parse_size(fields[0], index)) return 0;
    if (!geo_cycle_parse_size(fields[1], &parsed) || parsed > 255u) return 0;
    *kind = (int)parsed;
    *name = fields[2];
    if (!geo_cycle_parse_size(fields[3], &parsed) || parsed > (size_t)UINT32_MAX) return 0;
    *left = (geo_v8_node_id_t)parsed;
    if (!geo_cycle_parse_size(fields[4], &parsed) || parsed > (size_t)UINT32_MAX) return 0;
    *right = (geo_v8_node_id_t)parsed;
    if (!geo_cycle_parse_double(fields[5], scalar)) return 0;
    if (!geo_cycle_parse_size(fields[6], &parsed) || parsed > 255u) return 0;
    *grade = (uint8_t)parsed;
    if (!geo_cycle_parse_size(fields[7], &parsed) || parsed > 255u) return 0;
    *constraint = (int)parsed;
    if (!geo_cycle_parse_size(fields[8], &parsed) || parsed > 1u) return 0;
    *requires_grad = (int)parsed;
    return 1;
}

static int cycle_split_pair(
    char *text,
    geo_v8_node_id_t *left,
    geo_v8_node_id_t *right
) {
    char *comma = strchr(text, ',');
    size_t parsed;
    if (comma == NULL) return 0;
    *comma = '\0';
    if (!geo_cycle_parse_size(geo_cycle_trim(text), &parsed) ||
        parsed > (size_t)UINT32_MAX) return 0;
    *left = (geo_v8_node_id_t)parsed;
    if (!geo_cycle_parse_size(geo_cycle_trim(comma + 1), &parsed) ||
        parsed > (size_t)UINT32_MAX) return 0;
    *right = (geo_v8_node_id_t)parsed;
    return 1;
}

void geo_cycle_model_free(geo_cycle_model_t *model) {
    geo_v8_program_free(&model->program);
    free(model->inputs);
    free(model->targets);
    free(model->parameters);
    free(model->states);
    free(model->initial_states);
    memset(model, 0, sizeof(*model));
}

static int cycle_collect_nodes(geo_cycle_model_t *model) {
    geo_v8_node_id_t i;
    size_t input = 0u;
    size_t target = 0u;
    size_t parameter = 0u;
    size_t state = 0u;
    for (i = 0u; i < model->program.node_count; ++i) {
        const uint8_t kind = model->program.nodes[i].kind;
        if (kind == GEO_V8_NODE_INPUT) ++input;
        else if (kind == GEO_V8_NODE_TARGET) ++target;
        else if (kind == GEO_V8_NODE_PARAMETER) ++parameter;
        else if (kind == GEO_V8_NODE_STATE) ++state;
    }
    model->inputs = input == 0u ? NULL :
        (geo_v8_node_id_t *)malloc(input * sizeof(*model->inputs));
    model->targets = target == 0u ? NULL :
        (geo_v8_node_id_t *)malloc(target * sizeof(*model->targets));
    model->parameters = parameter == 0u ? NULL :
        (geo_v8_node_id_t *)malloc(parameter * sizeof(*model->parameters));
    model->states = state == 0u ? NULL :
        (geo_v8_node_id_t *)malloc(state * sizeof(*model->states));
    model->initial_states = state == 0u ? NULL :
        (geo_operator_mv_f64_t *)malloc(state * sizeof(*model->initial_states));
    if ((input != 0u && model->inputs == NULL) ||
        (target != 0u && model->targets == NULL) ||
        (parameter != 0u && model->parameters == NULL) ||
        (state != 0u && (model->states == NULL || model->initial_states == NULL))) {
        return 0;
    }
    for (i = 0u; i < model->program.node_count; ++i) {
        const uint8_t kind = model->program.nodes[i].kind;
        if (kind == GEO_V8_NODE_INPUT) model->inputs[model->input_count++] = i;
        else if (kind == GEO_V8_NODE_TARGET) model->targets[model->target_count++] = i;
        else if (kind == GEO_V8_NODE_PARAMETER) model->parameters[model->parameter_count++] = i;
        else if (kind == GEO_V8_NODE_STATE) {
            model->states[model->state_count] = i;
            model->initial_states[model->state_count] = model->program.nodes[i].value;
            ++model->state_count;
        }
    }
    return model->input_count != 0u;
}

int geo_cycle_load_model(const char *path, geo_cycle_model_t *model) {
    FILE *file;
    geo_cycle_line_t line = {0};
    uint8_t dimension = 0u;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {0};
    size_t node_count = 0u;
    geo_v8_node_id_t loss = GEO_V8_INVALID_NODE;
    geo_v8_node_id_t output = GEO_V8_INVALID_NODE;
    int header_seen = 0;
    int dimension_seen = 0;
    int signature_seen = 0;
    int optimizer_seen = 0;
    int node_count_seen = 0;
    size_t nodes_read = 0u;
    int result;

    memset(model, 0, sizeof(*model));
    model->optimizer = GEO_V8_OPTIMIZER_ADAM;
    model->learning_rate = 0.01;
    model->beta1 = 0.9;
    model->beta2 = 0.999;
    model->epsilon = 1e-8;
    model->epochs = 1u;
    model->batch_size = 1u;
    model->reset_state_each_epoch = 1;

    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "geo_cycle: cannot open model '%s'\n", path);
        return 0;
    }

    while ((result = geo_cycle_read_line(file, &line)) > 0) {
        char *text = geo_cycle_trim(line.data);
        char *equals;
        char *key;
        char *value;
        if (*text == '\0' || *text == '#') continue;
        equals = strchr(text, '=');
        if (equals == NULL) goto invalid;
        *equals = '\0';
        key = geo_cycle_trim(text);
        value = geo_cycle_trim(equals + 1);

        if (strcmp(key, "geo_model_ir") == 0) {
            if (strcmp(value, "8") != 0 || header_seen) goto invalid;
            header_seen = 1;
        } else if (strcmp(key, "dimension") == 0) {
            size_t parsed;
            if (!geo_cycle_parse_size(value, &parsed) || parsed < 1u || parsed > 6u || dimension_seen) goto invalid;
            dimension = (uint8_t)parsed;
            dimension_seen = 1;
        } else if (strcmp(key, "signature") == 0) {
            if (!dimension_seen || !geo_cycle_parse_signature(value, dimension, signature) || signature_seen) goto invalid;
            signature_seen = 1;
        } else if (strcmp(key, "optimizer") == 0) {
            size_t parsed;
            if (!geo_cycle_parse_size(value, &parsed) || (parsed != 1u && parsed != 2u)) goto invalid;
            model->optimizer = (geo_v8_optimizer_t)parsed;
            optimizer_seen = 1;
        } else if (strcmp(key, "learning_rate") == 0) {
            if (!geo_cycle_parse_double(value, &model->learning_rate) || model->learning_rate <= 0.0) goto invalid;
        } else if (strcmp(key, "beta1") == 0) {
            if (!geo_cycle_parse_double(value, &model->beta1) || model->beta1 < 0.0 || model->beta1 >= 1.0) goto invalid;
        } else if (strcmp(key, "beta2") == 0) {
            if (!geo_cycle_parse_double(value, &model->beta2) || model->beta2 < 0.0 || model->beta2 >= 1.0) goto invalid;
        } else if (strcmp(key, "epsilon") == 0) {
            if (!geo_cycle_parse_double(value, &model->epsilon) || model->epsilon <= 0.0) goto invalid;
        } else if (strcmp(key, "epochs") == 0) {
            if (!geo_cycle_parse_size(value, &model->epochs) || model->epochs == 0u) goto invalid;
        } else if (strcmp(key, "batch_size") == 0) {
            if (!geo_cycle_parse_size(value, &model->batch_size) || model->batch_size == 0u) goto invalid;
        } else if (strcmp(key, "reset_state_each_epoch") == 0) {
            if (!geo_cycle_parse_bool(value, &model->reset_state_each_epoch)) goto invalid;
        } else if (strcmp(key, "reset_column") == 0) {
            if (!geo_cycle_parse_bool(value, &model->reset_column)) goto invalid;
        } else if (strcmp(key, "node_count") == 0) {
            if (!header_seen || !signature_seen || !optimizer_seen ||
                !geo_cycle_parse_size(value, &node_count) || node_count == 0u ||
                node_count > (size_t)UINT32_MAX || node_count_seen) goto invalid;
            if (geo_v8_program_init(
                    &model->program,
                    dimension,
                    signature,
                    GEO_V8_PAIRING_COEFFICIENT_EUCLIDEAN,
                    node_count
                ) != GEO_V8_OK) goto invalid;
            node_count_seen = 1;
        } else if (strcmp(key, "loss") == 0) {
            size_t parsed;
            if (!geo_cycle_parse_size(value, &parsed) || parsed > (size_t)UINT32_MAX) goto invalid;
            loss = (geo_v8_node_id_t)parsed;
        } else if (strcmp(key, "output") == 0) {
            size_t parsed;
            if (!geo_cycle_parse_size(value, &parsed) || parsed > (size_t)UINT32_MAX) goto invalid;
            output = (geo_v8_node_id_t)parsed;
        } else if (strcmp(key, "node") == 0) {
            size_t index;
            int kind;
            char *name;
            geo_v8_node_id_t left;
            geo_v8_node_id_t right;
            double scalar;
            uint8_t grade;
            int constraint;
            int requires_grad;
            geo_operator_mv_f64_t initial;
            geo_v8_node_id_t created;
            int next_result;
            char *next_text;
            char *next_equals;

            if (!node_count_seen || nodes_read >= node_count ||
                !cycle_split_node(value, &index, &kind, &name, &left, &right,
                    &scalar, &grade, &constraint, &requires_grad) ||
                index != nodes_read || kind < 1 || kind > 17 ||
                constraint < 0 || constraint > 3) goto invalid;
            next_result = geo_cycle_read_line(file, &line);
            if (next_result <= 0) goto invalid;
            next_text = geo_cycle_trim(line.data);
            next_equals = strchr(next_text, '=');
            if (next_equals == NULL) goto invalid;
            *next_equals = '\0';
            if (strcmp(geo_cycle_trim(next_text), "value") != 0) goto invalid;
            geo_cycle_mv_init(&initial, dimension, signature);
            if (!geo_cycle_parse_vector(
                    geo_cycle_trim(next_equals + 1),
                    initial.coefficients,
                    (size_t)1u << dimension
                )) goto invalid;

            if (kind >= GEO_V8_NODE_INPUT && kind <= GEO_V8_NODE_STATE) {
                if (geo_v8_add_leaf(
                        &model->program,
                        (geo_v8_node_kind_t)kind,
                        name,
                        &initial,
                        requires_grad,
                        (geo_v8_constraint_t)constraint,
                        &created
                    ) != GEO_V8_OK) goto invalid;
            } else if (kind == GEO_V8_NODE_ADD ||
                       kind == GEO_V8_NODE_GEOMETRIC_PRODUCT ||
                       kind == GEO_V8_NODE_HADAMARD) {
                if (left >= nodes_read || right >= nodes_read ||
                    geo_v8_add_binary(
                        &model->program,
                        (geo_v8_node_kind_t)kind,
                        name,
                        left,
                        right,
                        &created
                    ) != GEO_V8_OK) goto invalid;
            } else {
                if (left >= nodes_read || right != GEO_V8_INVALID_NODE ||
                    geo_v8_add_unary(
                        &model->program,
                        (geo_v8_node_kind_t)kind,
                        name,
                        left,
                        scalar,
                        grade,
                        &created
                    ) != GEO_V8_OK) goto invalid;
            }
            if ((size_t)created != index) goto invalid;
            ++nodes_read;
        } else if (strcmp(key, "state_update") == 0) {
            geo_v8_node_id_t state;
            geo_v8_node_id_t source;
            if (!cycle_split_pair(value, &state, &source) ||
                geo_v8_bind_state_update(&model->program, state, source) != GEO_V8_OK) goto invalid;
        } else if (strcmp(key, "end") == 0) {
            if (strcmp(value, "1") != 0) goto invalid;
        } else goto invalid;
    }

    if (result < 0 || ferror(file) || !header_seen || !dimension_seen || !signature_seen ||
        !node_count_seen || nodes_read != node_count || loss >= node_count || output >= node_count ||
        geo_v8_compile(&model->program, loss) != GEO_V8_OK) goto invalid;
    model->output_node = output;
    fclose(file);
    geo_cycle_line_free(&line);
    if (!cycle_collect_nodes(model)) {
        geo_cycle_model_free(model);
        return 0;
    }
    return 1;

invalid:
    fprintf(stderr, "geo_cycle: invalid model IR '%s'\n", path);
    fclose(file);
    geo_cycle_line_free(&line);
    geo_cycle_model_free(model);
    return 0;
}

int geo_cycle_reset_states(geo_cycle_model_t *model) {
    size_t i;
    for (i = 0u; i < model->state_count; ++i) {
        if (geo_v8_set_value(
                &model->program,
                model->states[i],
                &model->initial_states[i]
            ) != GEO_V8_OK) return 0;
    }
    return 1;
}

int geo_cycle_set_leaf_coefficients(
    geo_cycle_model_t *model,
    const geo_v8_node_id_t *nodes,
    size_t node_count,
    const double *values
) {
    size_t i;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    for (i = 0u; i < node_count; ++i) {
        geo_operator_mv_f64_t value = model->program.nodes[nodes[i]].value;
        memcpy(
            value.coefficients,
            values + i * blade_count,
            blade_count * sizeof(double)
        );
        if (geo_v8_set_value(&model->program, nodes[i], &value) != GEO_V8_OK) return 0;
    }
    return 1;
}

int geo_cycle_atomic_begin(
    const char *path,
    char *temporary,
    size_t capacity,
    FILE **file
) {
    const int written = snprintf(temporary, capacity, "%s.tmp", path);
    if (written < 0 || (size_t)written >= capacity) return 0;
    *file = fopen(temporary, "wb");
    return *file != NULL;
}

int geo_cycle_atomic_finish(FILE *file, const char *temporary, const char *path) {
    if (fflush(file) != 0 || fclose(file) != 0) {
        remove(temporary);
        return 0;
    }
    remove(path);
    if (rename(temporary, path) != 0) {
        remove(temporary);
        return 0;
    }
    return 1;
}
