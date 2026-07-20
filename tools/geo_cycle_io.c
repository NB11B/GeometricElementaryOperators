#include "geo_cycle_internal.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int geo_cycle_save_checkpoint(const char *path, const geo_cycle_model_t *model) {
    char temporary[GEO_CYCLE_PATH_MAX];
    FILE *file;
    size_t i;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    if (!geo_cycle_atomic_begin(path, temporary, sizeof(temporary), &file)) return 0;
    fprintf(file, "geo_cycle_checkpoint=8\n");
    fprintf(file, "dimension=%u\n", (unsigned)model->program.dimension);
    fprintf(file, "signature=");
    for (i = 0u; i < model->program.dimension; ++i) {
        fprintf(file, "%s%d", i == 0u ? "" : ",", (int)model->program.signature[i]);
    }
    fprintf(file, "\noptimizer_step=%llu\n", (unsigned long long)model->program.optimizer_step);
    for (i = 0u; i < model->parameter_count; ++i) {
        const geo_v8_node_t *node = &model->program.nodes[model->parameters[i]];
        size_t blade;
        fprintf(file, "parameter=%s,%u\nvalue=", node->name, (unsigned)node->constraint);
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(file, "%s%.17g", blade == 0u ? "" : ",", node->value.coefficients[blade]);
        }
        fprintf(file, "\nfirst=");
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(file, "%s%.17g", blade == 0u ? "" : ",", node->first_moment[blade]);
        }
        fprintf(file, "\nsecond=");
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(file, "%s%.17g", blade == 0u ? "" : ",", node->second_moment[blade]);
        }
        fprintf(file, "\nend_parameter=1\n");
    }
    for (i = 0u; i < model->state_count; ++i) {
        const geo_v8_node_t *node = &model->program.nodes[model->states[i]];
        size_t blade;
        fprintf(file, "state=%s\nvalue=", node->name);
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(file, "%s%.17g", blade == 0u ? "" : ",", node->value.coefficients[blade]);
        }
        fprintf(file, "\nend_state=1\n");
    }
    fprintf(file, "end=1\n");
    return geo_cycle_atomic_finish(file, temporary, path);
}

static int cycle_checkpoint_name_constraint(
    char *text,
    char **name,
    int *constraint
) {
    char *comma = strchr(text, ',');
    size_t parsed;
    if (comma == NULL) return 0;
    *comma = '\0';
    *name = geo_cycle_trim(text);
    if (!geo_cycle_parse_size(geo_cycle_trim(comma + 1), &parsed) || parsed > 3u) return 0;
    *constraint = (int)parsed;
    return 1;
}

static size_t cycle_id_position(
    const geo_v8_node_id_t *ids,
    size_t count,
    geo_v8_node_id_t id
) {
    size_t i;
    for (i = 0u; i < count; ++i) {
        if (ids[i] == id) return i;
    }
    return SIZE_MAX;
}

int geo_cycle_load_checkpoint(const char *path, geo_cycle_model_t *model) {
    FILE *file = fopen(path, "rb");
    geo_cycle_line_t line = {0};
    uint8_t *parameter_seen = NULL;
    uint8_t *state_seen = NULL;
    int result;
    int header = 0;
    int dimension = 0;
    int signature = 0;
    int step = 0;
    size_t parameters = 0u;
    size_t states = 0u;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    if (file == NULL) return 0;
    if (model->parameter_count != 0u) {
        parameter_seen = (uint8_t *)calloc(model->parameter_count, sizeof(*parameter_seen));
    }
    if (model->state_count != 0u) {
        state_seen = (uint8_t *)calloc(model->state_count, sizeof(*state_seen));
    }
    if ((model->parameter_count != 0u && parameter_seen == NULL) ||
        (model->state_count != 0u && state_seen == NULL)) goto invalid;

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
        if (strcmp(key, "geo_cycle_checkpoint") == 0) {
            if (strcmp(value, "8") != 0 || header) goto invalid;
            header = 1;
        } else if (strcmp(key, "dimension") == 0) {
            size_t parsed;
            if (!geo_cycle_parse_size(value, &parsed) || parsed != model->program.dimension) goto invalid;
            dimension = 1;
        } else if (strcmp(key, "signature") == 0) {
            int8_t parsed[GEO_OPERATOR_MAX_DIMENSION] = {0};
            if (!geo_cycle_parse_signature(value, model->program.dimension, parsed) ||
                memcmp(parsed, model->program.signature, model->program.dimension) != 0) goto invalid;
            signature = 1;
        } else if (strcmp(key, "optimizer_step") == 0) {
            if (!geo_cycle_parse_u64(value, &model->program.optimizer_step)) goto invalid;
            step = 1;
        } else if (strcmp(key, "parameter") == 0) {
            char *name;
            int constraint;
            geo_v8_node_id_t id;
            geo_v8_node_t *node;
            size_t position;
            int r;
            if (!cycle_checkpoint_name_constraint(value, &name, &constraint)) goto invalid;
            id = geo_v8_find_node(&model->program, name);
            if (id == GEO_V8_INVALID_NODE || model->program.nodes[id].kind != GEO_V8_NODE_PARAMETER ||
                model->program.nodes[id].constraint != (uint8_t)constraint) goto invalid;
            position = cycle_id_position(model->parameters, model->parameter_count, id);
            if (position == SIZE_MAX || parameter_seen[position] != 0u) goto invalid;
            parameter_seen[position] = 1u;
            node = &model->program.nodes[id];
            r = geo_cycle_read_line(file, &line);
            if (r <= 0 || strncmp(geo_cycle_trim(line.data), "value=", 6) != 0 ||
                !geo_cycle_parse_vector(geo_cycle_trim(line.data) + 6, node->value.coefficients, blade_count)) goto invalid;
            r = geo_cycle_read_line(file, &line);
            if (r <= 0 || strncmp(geo_cycle_trim(line.data), "first=", 6) != 0 ||
                !geo_cycle_parse_vector(geo_cycle_trim(line.data) + 6, node->first_moment, blade_count)) goto invalid;
            r = geo_cycle_read_line(file, &line);
            if (r <= 0 || strncmp(geo_cycle_trim(line.data), "second=", 7) != 0 ||
                !geo_cycle_parse_vector(geo_cycle_trim(line.data) + 7, node->second_moment, blade_count)) goto invalid;
            r = geo_cycle_read_line(file, &line);
            if (r <= 0 || strcmp(geo_cycle_trim(line.data), "end_parameter=1") != 0) goto invalid;
            ++parameters;
        } else if (strcmp(key, "state") == 0) {
            geo_v8_node_id_t id = geo_v8_find_node(&model->program, value);
            geo_v8_node_t *node;
            size_t position;
            int r;
            if (id == GEO_V8_INVALID_NODE || model->program.nodes[id].kind != GEO_V8_NODE_STATE) goto invalid;
            position = cycle_id_position(model->states, model->state_count, id);
            if (position == SIZE_MAX || state_seen[position] != 0u) goto invalid;
            state_seen[position] = 1u;
            node = &model->program.nodes[id];
            r = geo_cycle_read_line(file, &line);
            if (r <= 0 || strncmp(geo_cycle_trim(line.data), "value=", 6) != 0 ||
                !geo_cycle_parse_vector(geo_cycle_trim(line.data) + 6, node->value.coefficients, blade_count)) goto invalid;
            r = geo_cycle_read_line(file, &line);
            if (r <= 0 || strcmp(geo_cycle_trim(line.data), "end_state=1") != 0) goto invalid;
            ++states;
        } else if (strcmp(key, "end") == 0) {
            if (strcmp(value, "1") != 0) goto invalid;
        } else goto invalid;
    }
    if (result < 0 || ferror(file) || !header || !dimension || !signature || !step ||
        parameters != model->parameter_count || states != model->state_count) goto invalid;
    fclose(file);
    geo_cycle_line_free(&line);
    free(parameter_seen);
    free(state_seen);
    model->program.forward_valid = 0u;
    model->program.backward_valid = 0u;
    model->program.gradient_samples = 0u;
    return 1;

invalid:
    fclose(file);
    geo_cycle_line_free(&line);
    free(parameter_seen);
    free(state_seen);
    fprintf(stderr, "geo_cycle: invalid or mismatched checkpoint '%s'\n", path);
    return 0;
}

int geo_cycle_train(
    const char *data_path,
    const char *checkpoint_path,
    const char *resume,
    geo_cycle_model_t *model
) {
    size_t epoch;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    const size_t expected = (size_t)model->reset_column +
        (model->input_count + model->target_count) * blade_count;
    double *values = (double *)malloc(expected * sizeof(*values));
    double initial_loss = NAN;
    double final_loss = NAN;
    uint64_t rows_total = 0u;
    if (values == NULL || model->target_count == 0u || model->parameter_count == 0u) {
        free(values);
        return 0;
    }
    if (resume != NULL && !geo_cycle_load_checkpoint(resume, model)) {
        free(values);
        return 0;
    }
    for (epoch = 0u; epoch < model->epochs; ++epoch) {
        FILE *file;
        geo_cycle_line_t line = {0};
        int read_result;
        size_t batch_count = 0u;
        double epoch_loss = 0.0;
        uint64_t rows = 0u;
        if (model->reset_state_each_epoch && !geo_cycle_reset_states(model)) {
            free(values);
            return 0;
        }
        if (geo_v8_zero_grad(&model->program) != GEO_V8_OK) {
            free(values);
            return 0;
        }
        file = fopen(data_path, "rb");
        if (file == NULL) {
            free(values);
            return 0;
        }
        while ((read_result = geo_cycle_read_line(file, &line)) > 0) {
            char *text = geo_cycle_trim(line.data);
            size_t offset = 0u;
            const geo_operator_mv_f64_t *loss;
            if (*text == '\0' || *text == '#') continue;
            if (!geo_cycle_parse_csv_row(text, values, expected)) goto train_invalid;
            if (model->reset_column) {
                if (values[0] != 0.0 && values[0] != 1.0) goto train_invalid;
                if (values[0] == 1.0 && !geo_cycle_reset_states(model)) goto train_invalid;
                offset = 1u;
            }
            if (!geo_cycle_set_leaf_coefficients(
                    model,
                    model->inputs,
                    model->input_count,
                    values + offset
                )) goto train_invalid;
            offset += model->input_count * blade_count;
            if (!geo_cycle_set_leaf_coefficients(
                    model,
                    model->targets,
                    model->target_count,
                    values + offset
                )) goto train_invalid;
            if (geo_v8_forward(&model->program) != GEO_V8_OK) goto train_invalid;
            loss = geo_v8_value(&model->program, model->program.loss_node);
            if (loss == NULL || !isfinite(loss->coefficients[0])) goto train_invalid;
            epoch_loss += loss->coefficients[0];
            if (geo_v8_backward(&model->program, batch_count != 0u) != GEO_V8_OK) goto train_invalid;
            if (model->state_count != 0u &&
                geo_v8_commit_states(&model->program) != GEO_V8_OK) goto train_invalid;
            ++batch_count;
            ++rows;
            ++rows_total;
            if (batch_count == model->batch_size) {
                if (geo_v8_optimizer_step(
                        &model->program,
                        model->optimizer,
                        model->learning_rate,
                        model->beta1,
                        model->beta2,
                        model->epsilon,
                        1.0 / (double)batch_count
                    ) != GEO_V8_OK) goto train_invalid;
                batch_count = 0u;
            }
        }
        if (read_result < 0 || ferror(file) || rows == 0u) goto train_invalid;
        if (batch_count != 0u) {
            if (geo_v8_optimizer_step(
                    &model->program,
                    model->optimizer,
                    model->learning_rate,
                    model->beta1,
                    model->beta2,
                    model->epsilon,
                    1.0 / (double)batch_count
                ) != GEO_V8_OK) goto train_invalid;
        }
        fclose(file);
        geo_cycle_line_free(&line);
        epoch_loss /= (double)rows;
        if (epoch == 0u) initial_loss = epoch_loss;
        final_loss = epoch_loss;
        continue;

train_invalid:
        fclose(file);
        geo_cycle_line_free(&line);
        free(values);
        fprintf(stderr, "geo_cycle: invalid training data or runtime failure\n");
        return 0;
    }
    free(values);
    if (!geo_cycle_save_checkpoint(checkpoint_path, model)) return 0;
    printf(
        "GEO_CYCLE_TRAIN: PASS epochs=%zu rows=%llu batch_size=%zu parameters=%zu states=%zu initial_loss=%.17g final_loss=%.17g checkpoint=%s\n",
        model->epochs,
        (unsigned long long)rows_total,
        model->batch_size,
        model->parameter_count,
        model->state_count,
        initial_loss,
        final_loss,
        checkpoint_path
    );
    return 1;
}

int geo_cycle_predict(
    const char *input_path,
    const char *output_path,
    geo_cycle_model_t *model
) {
    FILE *input;
    FILE *output;
    char temporary[GEO_CYCLE_PATH_MAX];
    geo_cycle_line_t line = {0};
    int read_result;
    uint64_t rows = 0u;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    const size_t expected = (size_t)model->reset_column +
        model->input_count * blade_count;
    double *values = (double *)malloc(expected * sizeof(*values));
    if (values == NULL) return 0;
    input = fopen(input_path, "rb");
    if (input == NULL ||
        !geo_cycle_atomic_begin(output_path, temporary, sizeof(temporary), &output)) {
        if (input != NULL) fclose(input);
        free(values);
        return 0;
    }
    while ((read_result = geo_cycle_read_line(input, &line)) > 0) {
        char *text = geo_cycle_trim(line.data);
        size_t offset = 0u;
        const geo_operator_mv_f64_t *prediction;
        size_t blade;
        if (*text == '\0' || *text == '#') continue;
        if (!geo_cycle_parse_csv_row(text, values, expected)) goto predict_invalid;
        if (model->reset_column) {
            if (values[0] != 0.0 && values[0] != 1.0) goto predict_invalid;
            if (values[0] == 1.0 && !geo_cycle_reset_states(model)) goto predict_invalid;
            offset = 1u;
        }
        if (!geo_cycle_set_leaf_coefficients(
                model,
                model->inputs,
                model->input_count,
                values + offset
            )) goto predict_invalid;
        if (geo_v8_forward(&model->program) != GEO_V8_OK) goto predict_invalid;
        prediction = geo_v8_value(&model->program, model->output_node);
        if (prediction == NULL) goto predict_invalid;
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(output, "%s%.17g", blade == 0u ? "" : ",", prediction->coefficients[blade]);
        }
        fprintf(output, "\n");
        if (model->state_count != 0u &&
            geo_v8_commit_states(&model->program) != GEO_V8_OK) goto predict_invalid;
        ++rows;
    }
    if (read_result < 0 || ferror(input) || rows == 0u) goto predict_invalid;
    fclose(input);
    geo_cycle_line_free(&line);
    free(values);
    if (!geo_cycle_atomic_finish(output, temporary, output_path)) return 0;
    printf(
        "GEO_CYCLE_PREDICT: PASS rows=%llu output=%s\n",
        (unsigned long long)rows,
        output_path
    );
    return 1;

predict_invalid:
    fclose(input);
    fclose(output);
    remove(temporary);
    geo_cycle_line_free(&line);
    free(values);
    fprintf(stderr, "geo_cycle: invalid prediction input or runtime failure\n");
    return 0;
}

static int cycle_symbol_valid(const char *symbol) {
    const unsigned char *cursor = (const unsigned char *)symbol;
    if (symbol == NULL || !(isalpha(*cursor) || *cursor == '_')) return 0;
    ++cursor;
    while (*cursor != '\0') {
        if (!(isalnum(*cursor) || *cursor == '_')) return 0;
        ++cursor;
    }
    return 1;
}

int geo_cycle_export_c(
    const char *path,
    const char *symbol,
    const geo_cycle_model_t *model
) {
    char temporary[GEO_CYCLE_PATH_MAX];
    FILE *file;
    size_t i;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    if (!cycle_symbol_valid(symbol) ||
        !geo_cycle_atomic_begin(path, temporary, sizeof(temporary), &file)) return 0;
    fprintf(
        file,
        "#ifndef %s_GEO_V8_MODEL_H\n#define %s_GEO_V8_MODEL_H\n\n",
        symbol,
        symbol
    );
    fprintf(file, "#include <stddef.h>\n#include <stdint.h>\n\n");
    fprintf(
        file,
        "#define %s_DIMENSION %uu\n#define %s_BLADE_COUNT %zuu\n#define %s_PARAMETER_COUNT %zuu\n\n",
        symbol,
        (unsigned)model->program.dimension,
        symbol,
        blade_count,
        symbol,
        model->parameter_count
    );
    fprintf(
        file,
        "static const int8_t %s_signature[%u] = {",
        symbol,
        (unsigned)model->program.dimension
    );
    for (i = 0u; i < model->program.dimension; ++i) {
        fprintf(file, "%s%d", i == 0u ? "" : ", ", (int)model->program.signature[i]);
    }
    fprintf(file, "};\n");
    for (i = 0u; i < model->parameter_count; ++i) {
        const geo_v8_node_t *node = &model->program.nodes[model->parameters[i]];
        size_t blade;
        fprintf(
            file,
            "static const double %s_%s[%zu] = {\n",
            symbol,
            node->name,
            blade_count
        );
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(
                file,
                "    %.17g%s\n",
                node->value.coefficients[blade],
                blade + 1u == blade_count ? "" : ","
            );
        }
        fprintf(file, "};\n");
    }
    fprintf(file, "\n#endif\n");
    if (!geo_cycle_atomic_finish(file, temporary, path)) return 0;
    printf(
        "GEO_CYCLE_EXPORT_C: PASS parameters=%zu header=%s\n",
        model->parameter_count,
        path
    );
    return 1;
}

int geo_cycle_check(geo_cycle_model_t *model) {
    size_t i;
    const size_t blade_count = (size_t)1u << model->program.dimension;
    for (i = 0u; i < model->input_count; ++i) {
        geo_operator_mv_f64_t probe = model->program.nodes[model->inputs[i]].value;
        size_t blade;
        for (blade = 0u; blade < blade_count; ++blade) {
            probe.coefficients[blade] =
                (double)((int)((blade + 1u) * (i + 3u) % 17u) - 8) / 19.0;
        }
        if (geo_v8_set_value(&model->program, model->inputs[i], &probe) != GEO_V8_OK) return 0;
    }
    if (geo_v8_forward(&model->program) != GEO_V8_OK) return 0;
    if (geo_v8_backward(&model->program, 0) != GEO_V8_OK) return 0;
    printf(
        "GEO_CYCLE_CHECK: PASS nodes=%u parameters=%zu inputs=%zu targets=%zu states=%zu abi=0x%08x external_autograd=NONE\n",
        (unsigned)model->program.node_count,
        model->parameter_count,
        model->input_count,
        model->target_count,
        model->state_count,
        geo_v8_abi_version()
    );
    return 1;
}
