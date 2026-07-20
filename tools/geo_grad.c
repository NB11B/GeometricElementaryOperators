#include "geo/autodiff_v7.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEO_GRAD_TOOL_VERSION "7.1"
#define GEO_GRAD_LINE_MAX 8192u
#define GEO_GRAD_PATH_MAX 1024u

typedef enum {
    GEO_GRAD_OPTIMIZER_SGD = 1,
    GEO_GRAD_OPTIMIZER_ADAM = 2
} geo_grad_optimizer_t;

typedef struct {
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    geo_operator_side_t side;
    geo_grad_optimizer_t optimizer;
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    size_t epochs;
    uint64_t seed;
} geo_grad_config_t;

typedef struct {
    size_t row_count;
    size_t blade_count;
    double *inputs;
    double *targets;
} geo_grad_dataset_t;

typedef struct {
    geo_v7_program_t program;
    geo_v7_node_id_t input_node;
    geo_v7_node_id_t target_node;
    geo_v7_node_id_t parameter_node;
    geo_v7_node_id_t prediction_node;
    geo_v7_node_id_t loss_node;
} geo_grad_model_t;

typedef struct {
    uint8_t dimension;
    int8_t signature[GEO_OPERATOR_MAX_DIMENSION];
    geo_operator_side_t side;
    uint64_t optimizer_step;
    double coefficients[GEO_OPERATOR_MAX_BLADES];
    double first_moment[GEO_OPERATOR_MAX_BLADES];
    double second_moment[GEO_OPERATOR_MAX_BLADES];
} geo_grad_checkpoint_t;

static void geo_grad_usage(FILE *stream) {
    fprintf(
        stream,
        "geo_grad %s\n"
        "Standalone GEO-native gradient compiler and trainer.\n\n"
        "Commands:\n"
        "  geo_grad init-example <model.geo> <train.csv>\n"
        "  geo_grad check <model.geo>\n"
        "  geo_grad train <model.geo> <train.csv> <checkpoint.txt>\n"
        "  geo_grad predict <model.geo> <checkpoint.txt> <input.csv> <output.csv>\n"
        "  geo_grad export-c <model.geo> <checkpoint.txt> <header.h> <symbol>\n\n"
        "CSV training rows contain input blade coefficients followed by target\n"
        "blade coefficients. Prediction input rows contain input coefficients only.\n",
        GEO_GRAD_TOOL_VERSION
    );
}

static char *geo_grad_trim(char *text) {
    char *end;
    while (*text != '\0' && isspace((unsigned char)*text)) ++text;
    end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return text;
}

static int geo_grad_parse_size(const char *text, size_t *output) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *geo_grad_trim(end) != '\0' || value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *output = (size_t)value;
    return 1;
}

static int geo_grad_parse_u64(const char *text, uint64_t *output) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *geo_grad_trim(end) != '\0') return 0;
    *output = (uint64_t)value;
    return 1;
}

static int geo_grad_parse_double(const char *text, double *output) {
    char *end = NULL;
    double value;
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *geo_grad_trim(end) != '\0' || !isfinite(value)) return 0;
    *output = value;
    return 1;
}

static int geo_grad_parse_signature(
    const char *text,
    uint8_t dimension,
    int8_t *signature
) {
    char buffer[128];
    char *cursor;
    uint8_t index = 0u;
    if (strlen(text) >= sizeof(buffer)) return 0;
    strcpy(buffer, text);
    cursor = strtok(buffer, ",");
    while (cursor != NULL) {
        char *trimmed = geo_grad_trim(cursor);
        long value;
        char *end = NULL;
        if (index >= dimension) return 0;
        errno = 0;
        value = strtol(trimmed, &end, 10);
        if (errno != 0 || end == trimmed || *geo_grad_trim(end) != '\0' ||
            (value != 1 && value != -1)) {
            return 0;
        }
        signature[index++] = (int8_t)value;
        cursor = strtok(NULL, ",");
    }
    return index == dimension;
}

static void geo_grad_config_defaults(geo_grad_config_t *config) {
    memset(config, 0, sizeof(*config));
    config->dimension = 3u;
    config->signature[0] = 1;
    config->signature[1] = 1;
    config->signature[2] = -1;
    config->side = GEO_OPERATOR_SIDE_RIGHT;
    config->optimizer = GEO_GRAD_OPTIMIZER_ADAM;
    config->learning_rate = 0.01;
    config->beta1 = 0.9;
    config->beta2 = 0.999;
    config->epsilon = 1e-8;
    config->epochs = 100u;
    config->seed = 1u;
}

static int geo_grad_validate_config(const geo_grad_config_t *config) {
    uint8_t index;
    if (config->dimension < 1u || config->dimension > GEO_OPERATOR_MAX_DIMENSION) return 0;
    for (index = 0u; index < config->dimension; ++index) {
        if (config->signature[index] != 1 && config->signature[index] != -1) return 0;
    }
    if (config->side != GEO_OPERATOR_SIDE_RIGHT && config->side != GEO_OPERATOR_SIDE_LEFT) return 0;
    if (config->optimizer != GEO_GRAD_OPTIMIZER_SGD &&
        config->optimizer != GEO_GRAD_OPTIMIZER_ADAM) return 0;
    if (!isfinite(config->learning_rate) || config->learning_rate <= 0.0) return 0;
    if (config->epochs == 0u) return 0;
    if (!isfinite(config->beta1) || config->beta1 < 0.0 || config->beta1 >= 1.0) return 0;
    if (!isfinite(config->beta2) || config->beta2 < 0.0 || config->beta2 >= 1.0) return 0;
    if (!isfinite(config->epsilon) || config->epsilon <= 0.0) return 0;
    return 1;
}

static int geo_grad_load_config(const char *path, geo_grad_config_t *config) {
    FILE *file;
    char line[GEO_GRAD_LINE_MAX];
    size_t line_number = 0u;
    int dimension_seen = 0;
    int signature_seen = 0;

    geo_grad_config_defaults(config);
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "geo_grad: cannot open model '%s'\n", path);
        return 0;
    }

    while (fgets(line, (int)sizeof(line), file) != NULL) {
        char *text;
        char *equals;
        char *key;
        char *value;
        ++line_number;
        text = geo_grad_trim(line);
        if (*text == '\0' || *text == '#') continue;
        equals = strchr(text, '=');
        if (equals == NULL) {
            fprintf(stderr, "geo_grad: %s:%zu missing '='\n", path, line_number);
            fclose(file);
            return 0;
        }
        *equals = '\0';
        key = geo_grad_trim(text);
        value = geo_grad_trim(equals + 1);

        if (strcmp(key, "version") == 0) {
            if (strcmp(value, GEO_GRAD_TOOL_VERSION) != 0) {
                fprintf(stderr, "geo_grad: unsupported model version '%s'\n", value);
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "dimension") == 0) {
            size_t parsed;
            if (!geo_grad_parse_size(value, &parsed) || parsed < 1u || parsed > GEO_OPERATOR_MAX_DIMENSION) {
                fprintf(stderr, "geo_grad: invalid dimension at %s:%zu\n", path, line_number);
                fclose(file);
                return 0;
            }
            config->dimension = (uint8_t)parsed;
            memset(config->signature, 0, sizeof(config->signature));
            dimension_seen = 1;
        } else if (strcmp(key, "signature") == 0) {
            if (!dimension_seen || !geo_grad_parse_signature(value, config->dimension, config->signature)) {
                fprintf(stderr, "geo_grad: invalid signature at %s:%zu\n", path, line_number);
                fclose(file);
                return 0;
            }
            signature_seen = 1;
        } else if (strcmp(key, "side") == 0) {
            if (strcmp(value, "right") == 0) config->side = GEO_OPERATOR_SIDE_RIGHT;
            else if (strcmp(value, "left") == 0) config->side = GEO_OPERATOR_SIDE_LEFT;
            else {
                fprintf(stderr, "geo_grad: invalid side at %s:%zu\n", path, line_number);
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "optimizer") == 0) {
            if (strcmp(value, "sgd") == 0) config->optimizer = GEO_GRAD_OPTIMIZER_SGD;
            else if (strcmp(value, "adam") == 0) config->optimizer = GEO_GRAD_OPTIMIZER_ADAM;
            else {
                fprintf(stderr, "geo_grad: invalid optimizer at %s:%zu\n", path, line_number);
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "learning_rate") == 0) {
            if (!geo_grad_parse_double(value, &config->learning_rate)) goto invalid_value;
        } else if (strcmp(key, "beta1") == 0) {
            if (!geo_grad_parse_double(value, &config->beta1)) goto invalid_value;
        } else if (strcmp(key, "beta2") == 0) {
            if (!geo_grad_parse_double(value, &config->beta2)) goto invalid_value;
        } else if (strcmp(key, "epsilon") == 0) {
            if (!geo_grad_parse_double(value, &config->epsilon)) goto invalid_value;
        } else if (strcmp(key, "epochs") == 0) {
            if (!geo_grad_parse_size(value, &config->epochs)) goto invalid_value;
        } else if (strcmp(key, "seed") == 0) {
            if (!geo_grad_parse_u64(value, &config->seed)) goto invalid_value;
        } else if (strcmp(key, "model") == 0) {
            if (strcmp(value, "multivector_gp") != 0) {
                fprintf(stderr, "geo_grad: unsupported model '%s'\n", value);
                fclose(file);
                return 0;
            }
        } else {
            fprintf(stderr, "geo_grad: unknown key '%s' at %s:%zu\n", key, path, line_number);
            fclose(file);
            return 0;
        }
        continue;

invalid_value:
        fprintf(stderr, "geo_grad: invalid value for '%s' at %s:%zu\n", key, path, line_number);
        fclose(file);
        return 0;
    }

    if (ferror(file)) {
        fprintf(stderr, "geo_grad: read error in '%s'\n", path);
        fclose(file);
        return 0;
    }
    fclose(file);

    if (!dimension_seen || !signature_seen || !geo_grad_validate_config(config)) {
        fprintf(stderr, "geo_grad: incomplete or invalid model '%s'\n", path);
        return 0;
    }
    return 1;
}

static void geo_grad_mv_init(
    geo_operator_mv_f64_t *value,
    const geo_grad_config_t *config
) {
    memset(value, 0, sizeof(*value));
    value->dimension = config->dimension;
    memcpy(value->signature, config->signature, config->dimension * sizeof(config->signature[0]));
}

static int geo_grad_build_model(
    const geo_grad_config_t *config,
    geo_grad_model_t *model
) {
    geo_operator_mv_f64_t zero;
    geo_v7_node_id_t negative_target;
    geo_v7_node_id_t residual;
    geo_v7_status_t status;

    memset(model, 0, sizeof(*model));
    geo_grad_mv_init(&zero, config);
    status = geo_v7_program_init(
        &model->program,
        config->dimension,
        config->signature,
        GEO_V7_PAIRING_COEFFICIENT_EUCLIDEAN
    );
    if (status != GEO_V7_OK) return 0;
    if (geo_v7_add_input(&model->program, &zero, 0, &model->input_node) != GEO_V7_OK) return 0;
    if (geo_v7_add_input(&model->program, &zero, 0, &model->target_node) != GEO_V7_OK) return 0;
    if (geo_v7_add_parameter(&model->program, &zero, &model->parameter_node) != GEO_V7_OK) return 0;

    if (config->side == GEO_OPERATOR_SIDE_RIGHT) {
        if (geo_v7_add_geometric_product(
                &model->program,
                model->input_node,
                model->parameter_node,
                &model->prediction_node
            ) != GEO_V7_OK) return 0;
    } else {
        if (geo_v7_add_geometric_product(
                &model->program,
                model->parameter_node,
                model->input_node,
                &model->prediction_node
            ) != GEO_V7_OK) return 0;
    }
    if (geo_v7_add_scale(&model->program, model->target_node, -1.0, &negative_target) != GEO_V7_OK) return 0;
    if (geo_v7_add_add(
            &model->program,
            model->prediction_node,
            negative_target,
            &residual
        ) != GEO_V7_OK) return 0;
    if (geo_v7_add_squared_norm(&model->program, residual, &model->loss_node) != GEO_V7_OK) return 0;
    if (geo_v7_compile(&model->program, model->loss_node) != GEO_V7_OK) return 0;
    return 1;
}

static int geo_grad_set_sample(
    geo_grad_model_t *model,
    const geo_grad_config_t *config,
    const double *input,
    const double *target
) {
    geo_operator_mv_f64_t input_value;
    geo_operator_mv_f64_t target_value;
    size_t blade;
    const size_t blade_count = (size_t)1u << config->dimension;
    geo_grad_mv_init(&input_value, config);
    geo_grad_mv_init(&target_value, config);
    for (blade = 0u; blade < blade_count; ++blade) {
        input_value.coefficients[blade] = input[blade];
        target_value.coefficients[blade] = target[blade];
    }
    return geo_v7_set_value(&model->program, model->input_node, &input_value) == GEO_V7_OK &&
        geo_v7_set_value(&model->program, model->target_node, &target_value) == GEO_V7_OK;
}

static int geo_grad_parse_csv_values(
    char *line,
    double *values,
    size_t expected
) {
    size_t count = 0u;
    char *cursor = line;
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
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0' && *cursor != '#') {
            return 0;
        }
    }
    return count == expected;
}

static void geo_grad_dataset_free(geo_grad_dataset_t *dataset) {
    free(dataset->inputs);
    free(dataset->targets);
    memset(dataset, 0, sizeof(*dataset));
}

static int geo_grad_dataset_append(
    geo_grad_dataset_t *dataset,
    const double *values
) {
    double *new_inputs;
    double *new_targets;
    const size_t next_count = dataset->row_count + 1u;
    const size_t coefficient_count = next_count * dataset->blade_count;
    new_inputs = (double *)realloc(dataset->inputs, coefficient_count * sizeof(double));
    if (new_inputs == NULL) return 0;
    dataset->inputs = new_inputs;
    new_targets = (double *)realloc(dataset->targets, coefficient_count * sizeof(double));
    if (new_targets == NULL) return 0;
    dataset->targets = new_targets;
    memcpy(
        dataset->inputs + dataset->row_count * dataset->blade_count,
        values,
        dataset->blade_count * sizeof(double)
    );
    memcpy(
        dataset->targets + dataset->row_count * dataset->blade_count,
        values + dataset->blade_count,
        dataset->blade_count * sizeof(double)
    );
    dataset->row_count = next_count;
    return 1;
}

static int geo_grad_load_dataset(
    const char *path,
    size_t blade_count,
    geo_grad_dataset_t *dataset
) {
    FILE *file;
    char line[GEO_GRAD_LINE_MAX];
    double values[2u * GEO_OPERATOR_MAX_BLADES];
    size_t line_number = 0u;
    memset(dataset, 0, sizeof(*dataset));
    dataset->blade_count = blade_count;
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "geo_grad: cannot open dataset '%s'\n", path);
        return 0;
    }
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        char *text = geo_grad_trim(line);
        ++line_number;
        if (*text == '\0' || *text == '#') continue;
        if (!geo_grad_parse_csv_values(text, values, 2u * blade_count)) {
            fprintf(stderr, "geo_grad: invalid dataset row %s:%zu\n", path, line_number);
            fclose(file);
            geo_grad_dataset_free(dataset);
            return 0;
        }
        if (!geo_grad_dataset_append(dataset, values)) {
            fprintf(stderr, "geo_grad: out of memory reading '%s'\n", path);
            fclose(file);
            geo_grad_dataset_free(dataset);
            return 0;
        }
    }
    if (ferror(file) || dataset->row_count == 0u) {
        fprintf(stderr, "geo_grad: empty or unreadable dataset '%s'\n", path);
        fclose(file);
        geo_grad_dataset_free(dataset);
        return 0;
    }
    fclose(file);
    return 1;
}

static double geo_grad_evaluate_dataset(
    geo_grad_model_t *model,
    const geo_grad_config_t *config,
    const geo_grad_dataset_t *dataset
) {
    size_t row;
    double total = 0.0;
    for (row = 0u; row < dataset->row_count; ++row) {
        const double *input = dataset->inputs + row * dataset->blade_count;
        const double *target = dataset->targets + row * dataset->blade_count;
        const geo_operator_mv_f64_t *loss;
        if (!geo_grad_set_sample(model, config, input, target)) return NAN;
        if (geo_v7_forward(&model->program) != GEO_V7_OK) return NAN;
        loss = geo_v7_value(&model->program, model->loss_node);
        if (loss == NULL || !isfinite(loss->coefficients[0])) return NAN;
        total += loss->coefficients[0];
    }
    return total / (double)dataset->row_count;
}

static int geo_grad_save_checkpoint(
    const char *path,
    const geo_grad_config_t *config,
    const geo_grad_model_t *model
) {
    FILE *file;
    const geo_v7_node_t *parameter = &model->program.nodes[model->parameter_node];
    size_t blade;
    const size_t blade_count = (size_t)1u << config->dimension;
    file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "geo_grad: cannot write checkpoint '%s'\n", path);
        return 0;
    }
    fprintf(file, "geo_grad_checkpoint=1\n");
    fprintf(file, "dimension=%u\n", (unsigned)config->dimension);
    fprintf(file, "signature=");
    for (blade = 0u; blade < config->dimension; ++blade) {
        fprintf(file, "%s%d", blade == 0u ? "" : ",", (int)config->signature[blade]);
    }
    fprintf(file, "\nside=%s\n", config->side == GEO_OPERATOR_SIDE_RIGHT ? "right" : "left");
    fprintf(file, "optimizer_step=%llu\n", (unsigned long long)model->program.optimizer_step);
    fprintf(file, "coefficients=");
    for (blade = 0u; blade < blade_count; ++blade) {
        fprintf(file, "%s%.17g", blade == 0u ? "" : ",", parameter->value.coefficients[blade]);
    }
    fprintf(file, "\nfirst_moment=");
    for (blade = 0u; blade < blade_count; ++blade) {
        fprintf(file, "%s%.17g", blade == 0u ? "" : ",", parameter->first_moment[blade]);
    }
    fprintf(file, "\nsecond_moment=");
    for (blade = 0u; blade < blade_count; ++blade) {
        fprintf(file, "%s%.17g", blade == 0u ? "" : ",", parameter->second_moment[blade]);
    }
    fprintf(file, "\n");
    if (fclose(file) != 0) {
        fprintf(stderr, "geo_grad: failed closing checkpoint '%s'\n", path);
        return 0;
    }
    return 1;
}

static int geo_grad_parse_vector(
    const char *text,
    double *output,
    size_t count
) {
    char buffer[GEO_GRAD_LINE_MAX];
    char *cursor;
    size_t index = 0u;
    if (strlen(text) >= sizeof(buffer)) return 0;
    strcpy(buffer, text);
    cursor = strtok(buffer, ",");
    while (cursor != NULL) {
        if (index >= count || !geo_grad_parse_double(geo_grad_trim(cursor), &output[index])) return 0;
        ++index;
        cursor = strtok(NULL, ",");
    }
    return index == count;
}

static int geo_grad_load_checkpoint(
    const char *path,
    const geo_grad_config_t *config,
    geo_grad_checkpoint_t *checkpoint
) {
    FILE *file;
    char line[GEO_GRAD_LINE_MAX];
    int header_seen = 0;
    int dimension_seen = 0;
    int signature_seen = 0;
    int side_seen = 0;
    int step_seen = 0;
    int coefficients_seen = 0;
    int first_seen = 0;
    int second_seen = 0;
    const size_t blade_count = (size_t)1u << config->dimension;
    memset(checkpoint, 0, sizeof(*checkpoint));
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "geo_grad: cannot open checkpoint '%s'\n", path);
        return 0;
    }
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        char *text = geo_grad_trim(line);
        char *equals;
        char *key;
        char *value;
        if (*text == '\0' || *text == '#') continue;
        equals = strchr(text, '=');
        if (equals == NULL) goto invalid;
        *equals = '\0';
        key = geo_grad_trim(text);
        value = geo_grad_trim(equals + 1);
        if (strcmp(key, "geo_grad_checkpoint") == 0) {
            if (strcmp(value, "1") != 0) goto invalid;
            header_seen = 1;
        } else if (strcmp(key, "dimension") == 0) {
            size_t parsed;
            if (!geo_grad_parse_size(value, &parsed) || parsed != config->dimension) goto invalid;
            checkpoint->dimension = (uint8_t)parsed;
            dimension_seen = 1;
        } else if (strcmp(key, "signature") == 0) {
            if (!geo_grad_parse_signature(value, config->dimension, checkpoint->signature)) goto invalid;
            signature_seen = 1;
        } else if (strcmp(key, "side") == 0) {
            if (strcmp(value, "right") == 0) checkpoint->side = GEO_OPERATOR_SIDE_RIGHT;
            else if (strcmp(value, "left") == 0) checkpoint->side = GEO_OPERATOR_SIDE_LEFT;
            else goto invalid;
            side_seen = 1;
        } else if (strcmp(key, "optimizer_step") == 0) {
            if (!geo_grad_parse_u64(value, &checkpoint->optimizer_step)) goto invalid;
            step_seen = 1;
        } else if (strcmp(key, "coefficients") == 0) {
            if (!geo_grad_parse_vector(value, checkpoint->coefficients, blade_count)) goto invalid;
            coefficients_seen = 1;
        } else if (strcmp(key, "first_moment") == 0) {
            if (!geo_grad_parse_vector(value, checkpoint->first_moment, blade_count)) goto invalid;
            first_seen = 1;
        } else if (strcmp(key, "second_moment") == 0) {
            if (!geo_grad_parse_vector(value, checkpoint->second_moment, blade_count)) goto invalid;
            second_seen = 1;
        } else {
            goto invalid;
        }
    }
    fclose(file);
    if (!header_seen || !dimension_seen || !signature_seen || !side_seen || !step_seen ||
        !coefficients_seen || !first_seen || !second_seen || checkpoint->side != config->side ||
        memcmp(checkpoint->signature, config->signature, config->dimension) != 0) {
        fprintf(stderr, "geo_grad: checkpoint '%s' does not match model\n", path);
        return 0;
    }
    return 1;

invalid:
    fclose(file);
    fprintf(stderr, "geo_grad: invalid checkpoint '%s'\n", path);
    return 0;
}

static void geo_grad_apply_checkpoint(
    const geo_grad_config_t *config,
    geo_grad_model_t *model,
    const geo_grad_checkpoint_t *checkpoint
) {
    geo_v7_node_t *parameter = &model->program.nodes[model->parameter_node];
    size_t blade;
    const size_t blade_count = (size_t)1u << config->dimension;
    for (blade = 0u; blade < blade_count; ++blade) {
        parameter->value.coefficients[blade] = checkpoint->coefficients[blade];
        parameter->first_moment[blade] = checkpoint->first_moment[blade];
        parameter->second_moment[blade] = checkpoint->second_moment[blade];
    }
    model->program.optimizer_step = checkpoint->optimizer_step;
    model->program.forward_valid = 0u;
    model->program.backward_valid = 0u;
}

static int geo_grad_train(
    const geo_grad_config_t *config,
    const char *dataset_path,
    const char *checkpoint_path
) {
    geo_grad_dataset_t dataset;
    geo_grad_model_t model;
    const size_t blade_count = (size_t)1u << config->dimension;
    double initial_loss;
    double final_loss;
    size_t epoch;

    if (!geo_grad_load_dataset(dataset_path, blade_count, &dataset)) return 0;
    if (!geo_grad_build_model(config, &model)) {
        fprintf(stderr, "geo_grad: failed to compile GEO V7 training graph\n");
        geo_grad_dataset_free(&dataset);
        return 0;
    }
    initial_loss = geo_grad_evaluate_dataset(&model, config, &dataset);
    if (!isfinite(initial_loss)) {
        fprintf(stderr, "geo_grad: failed initial model evaluation\n");
        geo_grad_dataset_free(&dataset);
        return 0;
    }

    for (epoch = 0u; epoch < config->epochs; ++epoch) {
        size_t row;
        for (row = 0u; row < dataset.row_count; ++row) {
            const double *input = dataset.inputs + row * dataset.blade_count;
            const double *target = dataset.targets + row * dataset.blade_count;
            geo_v7_status_t status;
            if (!geo_grad_set_sample(&model, config, input, target)) goto train_failure;
            status = geo_v7_forward(&model.program);
            if (status != GEO_V7_OK) goto train_failure;
            status = geo_v7_backward(&model.program);
            if (status != GEO_V7_OK) goto train_failure;
            if (config->optimizer == GEO_GRAD_OPTIMIZER_SGD) {
                status = geo_v7_sgd_step(&model.program, config->learning_rate);
            } else {
                status = geo_v7_adam_step(
                    &model.program,
                    config->learning_rate,
                    config->beta1,
                    config->beta2,
                    config->epsilon
                );
            }
            if (status != GEO_V7_OK) goto train_failure;
        }
    }

    final_loss = geo_grad_evaluate_dataset(&model, config, &dataset);
    if (!isfinite(final_loss) || !geo_grad_save_checkpoint(checkpoint_path, config, &model)) {
        geo_grad_dataset_free(&dataset);
        return 0;
    }
    printf(
        "GEO_GRAD_TRAIN: PASS rows=%zu epochs=%zu optimizer=%s initial_loss=%.17g final_loss=%.17g checkpoint=%s no_external_autograd=TRUE\n",
        dataset.row_count,
        config->epochs,
        config->optimizer == GEO_GRAD_OPTIMIZER_SGD ? "sgd" : "adam",
        initial_loss,
        final_loss,
        checkpoint_path
    );
    geo_grad_dataset_free(&dataset);
    return 1;

train_failure:
    fprintf(stderr, "geo_grad: GEO-native training step failed\n");
    geo_grad_dataset_free(&dataset);
    return 0;
}

static int geo_grad_check(const geo_grad_config_t *config) {
    geo_operator_mv_f64_t input;
    geo_operator_mv_f64_t weight;
    geo_operator_mv_f64_t input_tangent;
    geo_operator_mv_f64_t weight_tangent;
    geo_operator_mv_f64_t output_tangent;
    geo_operator_mv_f64_t output_cotangent;
    geo_operator_mv_f64_t input_cotangent;
    geo_operator_mv_f64_t weight_cotangent;
    geo_grad_model_t model;
    size_t blade;
    const size_t blade_count = (size_t)1u << config->dimension;
    double left_pairing;
    double right_pairing = 0.0;

    geo_grad_mv_init(&input, config);
    geo_grad_mv_init(&weight, config);
    geo_grad_mv_init(&input_tangent, config);
    geo_grad_mv_init(&weight_tangent, config);
    geo_grad_mv_init(&output_cotangent, config);
    for (blade = 0u; blade < blade_count; ++blade) {
        input.coefficients[blade] = (double)((int)((blade + config->seed) % 7u) - 3) / 3.0;
        weight.coefficients[blade] = (double)((int)((blade * 3u + config->seed) % 11u) - 5) / 4.0;
        input_tangent.coefficients[blade] = (double)((int)((blade * 5u + 1u) % 9u) - 4) / 5.0;
        weight_tangent.coefficients[blade] = (double)((int)((blade * 7u + 2u) % 13u) - 6) / 6.0;
        output_cotangent.coefficients[blade] = (double)((int)((blade * 2u + 3u) % 10u) - 5) / 4.0;
    }

    if (config->side == GEO_OPERATOR_SIDE_RIGHT) {
        if (geo_operator_gp_f64_jvp(
                &input,
                &weight,
                &input_tangent,
                &weight_tangent,
                &output_tangent
            ) != GEO_OPERATOR_OK) return 0;
        if (geo_operator_gp_f64_vjp(
                &input,
                &weight,
                &output_cotangent,
                &input_cotangent,
                &weight_cotangent
            ) != GEO_OPERATOR_OK) return 0;
    } else {
        if (geo_operator_gp_f64_jvp(
                &weight,
                &input,
                &weight_tangent,
                &input_tangent,
                &output_tangent
            ) != GEO_OPERATOR_OK) return 0;
        if (geo_operator_gp_f64_vjp(
                &weight,
                &input,
                &output_cotangent,
                &weight_cotangent,
                &input_cotangent
            ) != GEO_OPERATOR_OK) return 0;
    }

    left_pairing = 0.0;
    for (blade = 0u; blade < blade_count; ++blade) {
        left_pairing += output_cotangent.coefficients[blade] * output_tangent.coefficients[blade];
        right_pairing += input_cotangent.coefficients[blade] * input_tangent.coefficients[blade];
        right_pairing += weight_cotangent.coefficients[blade] * weight_tangent.coefficients[blade];
    }
    if (fabs(left_pairing - right_pairing) >
        1e-10 * (1.0 + fabs(left_pairing) + fabs(right_pairing))) {
        fprintf(stderr, "geo_grad: adjoint identity failed\n");
        return 0;
    }

    if (!geo_grad_build_model(config, &model)) return 0;
    if (!geo_grad_set_sample(&model, config, input.coefficients, output_tangent.coefficients)) return 0;
    if (geo_v7_forward(&model.program) != GEO_V7_OK ||
        geo_v7_backward(&model.program) != GEO_V7_OK) return 0;

    printf(
        "GEO_GRAD_CHECK: PASS dimension=%u blades=%zu side=%s adjoint_identity=PASS graph_compile=PASS forward=PASS backward=PASS unsupported_fallbacks=0 no_external_autograd=TRUE\n",
        (unsigned)config->dimension,
        blade_count,
        config->side == GEO_OPERATOR_SIDE_RIGHT ? "right" : "left"
    );
    return 1;
}

static int geo_grad_predict(
    const geo_grad_config_t *config,
    const char *checkpoint_path,
    const char *input_path,
    const char *output_path
) {
    geo_grad_checkpoint_t checkpoint;
    geo_grad_model_t model;
    FILE *input_file;
    FILE *output_file;
    char line[GEO_GRAD_LINE_MAX];
    double values[GEO_OPERATOR_MAX_BLADES];
    const size_t blade_count = (size_t)1u << config->dimension;
    size_t rows = 0u;

    if (!geo_grad_load_checkpoint(checkpoint_path, config, &checkpoint)) return 0;
    if (!geo_grad_build_model(config, &model)) return 0;
    geo_grad_apply_checkpoint(config, &model, &checkpoint);
    input_file = fopen(input_path, "rb");
    if (input_file == NULL) {
        fprintf(stderr, "geo_grad: cannot open prediction input '%s'\n", input_path);
        return 0;
    }
    output_file = fopen(output_path, "wb");
    if (output_file == NULL) {
        fprintf(stderr, "geo_grad: cannot write prediction output '%s'\n", output_path);
        fclose(input_file);
        return 0;
    }

    while (fgets(line, (int)sizeof(line), input_file) != NULL) {
        char *text = geo_grad_trim(line);
        geo_operator_mv_f64_t input_value;
        geo_operator_mv_f64_t target_zero;
        const geo_operator_mv_f64_t *prediction;
        size_t blade;
        if (*text == '\0' || *text == '#') continue;
        if (!geo_grad_parse_csv_values(text, values, blade_count)) {
            fprintf(stderr, "geo_grad: invalid prediction input row\n");
            fclose(input_file);
            fclose(output_file);
            return 0;
        }
        geo_grad_mv_init(&input_value, config);
        geo_grad_mv_init(&target_zero, config);
        for (blade = 0u; blade < blade_count; ++blade) input_value.coefficients[blade] = values[blade];
        if (geo_v7_set_value(&model.program, model.input_node, &input_value) != GEO_V7_OK ||
            geo_v7_set_value(&model.program, model.target_node, &target_zero) != GEO_V7_OK ||
            geo_v7_forward(&model.program) != GEO_V7_OK) {
            fclose(input_file);
            fclose(output_file);
            return 0;
        }
        prediction = geo_v7_value(&model.program, model.prediction_node);
        if (prediction == NULL) {
            fclose(input_file);
            fclose(output_file);
            return 0;
        }
        for (blade = 0u; blade < blade_count; ++blade) {
            fprintf(output_file, "%s%.17g", blade == 0u ? "" : ",", prediction->coefficients[blade]);
        }
        fprintf(output_file, "\n");
        ++rows;
    }
    fclose(input_file);
    if (fclose(output_file) != 0 || rows == 0u) return 0;
    printf("GEO_GRAD_PREDICT: PASS rows=%zu output=%s\n", rows, output_path);
    return 1;
}

static int geo_grad_symbol_valid(const char *symbol) {
    const unsigned char *cursor = (const unsigned char *)symbol;
    if (*cursor == '\0' || !(isalpha(*cursor) || *cursor == '_')) return 0;
    ++cursor;
    while (*cursor != '\0') {
        if (!(isalnum(*cursor) || *cursor == '_')) return 0;
        ++cursor;
    }
    return 1;
}

static int geo_grad_export_c(
    const geo_grad_config_t *config,
    const char *checkpoint_path,
    const char *header_path,
    const char *symbol
) {
    geo_grad_checkpoint_t checkpoint;
    FILE *file;
    size_t blade;
    const size_t blade_count = (size_t)1u << config->dimension;
    if (!geo_grad_symbol_valid(symbol)) {
        fprintf(stderr, "geo_grad: invalid C symbol '%s'\n", symbol);
        return 0;
    }
    if (!geo_grad_load_checkpoint(checkpoint_path, config, &checkpoint)) return 0;
    file = fopen(header_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "geo_grad: cannot write header '%s'\n", header_path);
        return 0;
    }
    fprintf(file, "#ifndef %s_GEO_GRAD_MODEL_H\n#define %s_GEO_GRAD_MODEL_H\n\n", symbol, symbol);
    fprintf(file, "#include <stddef.h>\n#include <stdint.h>\n\n");
    fprintf(file, "#define %s_DIMENSION %uu\n", symbol, (unsigned)config->dimension);
    fprintf(file, "#define %s_BLADE_COUNT %zuu\n", symbol, blade_count);
    fprintf(file, "#define %s_SIDE_%s 1\n\n", symbol,
        config->side == GEO_OPERATOR_SIDE_RIGHT ? "RIGHT" : "LEFT");
    fprintf(file, "static const int8_t %s_signature[%u] = {", symbol, (unsigned)config->dimension);
    for (blade = 0u; blade < config->dimension; ++blade) {
        fprintf(file, "%s%d", blade == 0u ? "" : ", ", (int)config->signature[blade]);
    }
    fprintf(file, "};\nstatic const double %s_coefficients[%zu] = {\n", symbol, blade_count);
    for (blade = 0u; blade < blade_count; ++blade) {
        fprintf(file, "    %.17g%s\n", checkpoint.coefficients[blade], blade + 1u == blade_count ? "" : ",");
    }
    fprintf(file, "};\n\n#endif\n");
    if (fclose(file) != 0) return 0;
    printf("GEO_GRAD_EXPORT_C: PASS header=%s symbol=%s blades=%zu\n", header_path, symbol, blade_count);
    return 1;
}

static int geo_grad_init_example(const char *model_path, const char *dataset_path) {
    FILE *model;
    FILE *dataset;
    const int8_t signature[GEO_OPERATOR_MAX_DIMENSION] = {1, 1, 0, 0, 0, 0};
    const double truth[4] = {0.75, -1.25, 0.5, 1.0};
    size_t source;

    model = fopen(model_path, "wb");
    if (model == NULL) return 0;
    fprintf(
        model,
        "# GEO gradient tool model\n"
        "version=7.1\n"
        "model=multivector_gp\n"
        "dimension=2\n"
        "signature=1,1\n"
        "side=right\n"
        "optimizer=adam\n"
        "learning_rate=0.05\n"
        "epochs=250\n"
        "beta1=0.9\n"
        "beta2=0.999\n"
        "epsilon=1e-8\n"
        "seed=1\n"
    );
    if (fclose(model) != 0) return 0;

    dataset = fopen(dataset_path, "wb");
    if (dataset == NULL) return 0;
    fprintf(dataset, "# input[4],target[4] for y = x * w\n");
    for (source = 0u; source < 4u; ++source) {
        geo_operator_mv_f64_t input;
        geo_operator_mv_f64_t weight;
        geo_operator_mv_f64_t target;
        size_t blade;
        memset(&input, 0, sizeof(input));
        memset(&weight, 0, sizeof(weight));
        input.dimension = 2u;
        weight.dimension = 2u;
        memcpy(input.signature, signature, 2u);
        memcpy(weight.signature, signature, 2u);
        input.coefficients[source] = 1.0;
        for (blade = 0u; blade < 4u; ++blade) weight.coefficients[blade] = truth[blade];
        if (geo_operator_gp_f64(&input, &weight, &target) != GEO_OPERATOR_OK) {
            fclose(dataset);
            return 0;
        }
        for (blade = 0u; blade < 4u; ++blade) fprintf(dataset, "%s%.17g", blade == 0u ? "" : ",", input.coefficients[blade]);
        for (blade = 0u; blade < 4u; ++blade) fprintf(dataset, ",%.17g", target.coefficients[blade]);
        fprintf(dataset, "\n");
    }
    if (fclose(dataset) != 0) return 0;
    printf("GEO_GRAD_INIT_EXAMPLE: PASS model=%s dataset=%s\n", model_path, dataset_path);
    return 1;
}

int main(int argc, char **argv) {
    geo_grad_config_t config;
    if (argc < 2) {
        geo_grad_usage(stderr);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("geo_grad %s GEO_V7_ABI=0x%08x\n", GEO_GRAD_TOOL_VERSION, geo_v7_abi_version());
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "init-example") == 0) {
        if (argc != 4) {
            geo_grad_usage(stderr);
            return EXIT_FAILURE;
        }
        return geo_grad_init_example(argv[2], argv[3]) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "check") == 0) {
        if (argc != 3 || !geo_grad_load_config(argv[2], &config)) return EXIT_FAILURE;
        return geo_grad_check(&config) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "train") == 0) {
        if (argc != 5 || !geo_grad_load_config(argv[2], &config)) return EXIT_FAILURE;
        return geo_grad_train(&config, argv[3], argv[4]) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "predict") == 0) {
        if (argc != 6 || !geo_grad_load_config(argv[2], &config)) return EXIT_FAILURE;
        return geo_grad_predict(&config, argv[3], argv[4], argv[5]) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (strcmp(argv[1], "export-c") == 0) {
        if (argc != 6 || !geo_grad_load_config(argv[2], &config)) return EXIT_FAILURE;
        return geo_grad_export_c(&config, argv[3], argv[4], argv[5]) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    geo_grad_usage(stderr);
    return EXIT_FAILURE;
}
