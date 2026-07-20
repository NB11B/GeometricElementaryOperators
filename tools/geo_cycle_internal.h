#ifndef GEO_CYCLE_INTERNAL_H
#define GEO_CYCLE_INTERNAL_H

#include "geo/full_cycle_v8.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define GEO_CYCLE_VERSION "8.0"
#define GEO_CYCLE_PATH_MAX 1024u

typedef struct {
    geo_v8_program_t program;
    geo_v8_optimizer_t optimizer;
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    size_t epochs;
    size_t batch_size;
    int reset_state_each_epoch;
    int reset_column;
    geo_v8_node_id_t output_node;
    geo_v8_node_id_t *inputs;
    size_t input_count;
    geo_v8_node_id_t *targets;
    size_t target_count;
    geo_v8_node_id_t *parameters;
    size_t parameter_count;
    geo_v8_node_id_t *states;
    geo_operator_mv_f64_t *initial_states;
    size_t state_count;
} geo_cycle_model_t;

typedef struct {
    char *data;
    size_t capacity;
} geo_cycle_line_t;

void geo_cycle_usage(FILE *stream);
char *geo_cycle_trim(char *text);
int geo_cycle_read_line(FILE *file, geo_cycle_line_t *line);
void geo_cycle_line_free(geo_cycle_line_t *line);
int geo_cycle_parse_size(const char *text, size_t *output);
int geo_cycle_parse_u64(const char *text, uint64_t *output);
int geo_cycle_parse_double(const char *text, double *output);
int geo_cycle_parse_bool(const char *text, int *output);
int geo_cycle_parse_signature(const char *text, uint8_t dimension, int8_t *signature);
int geo_cycle_parse_vector(const char *text, double *output, size_t count);
int geo_cycle_parse_csv_row(char *line, double *values, size_t expected);
void geo_cycle_mv_init(geo_operator_mv_f64_t *value, uint8_t dimension, const int8_t *signature);
void geo_cycle_model_free(geo_cycle_model_t *model);
int geo_cycle_load_model(const char *path, geo_cycle_model_t *model);
int geo_cycle_reset_states(geo_cycle_model_t *model);
int geo_cycle_set_leaf_coefficients(geo_cycle_model_t *model, const geo_v8_node_id_t *nodes, size_t node_count, const double *values);
int geo_cycle_atomic_begin(const char *path, char *temporary, size_t capacity, FILE **file);
int geo_cycle_atomic_finish(FILE *file, const char *temporary, const char *path);
int geo_cycle_save_checkpoint(const char *path, const geo_cycle_model_t *model);
int geo_cycle_load_checkpoint(const char *path, geo_cycle_model_t *model);
int geo_cycle_train(const char *data_path, const char *checkpoint_path, const char *resume, geo_cycle_model_t *model);
int geo_cycle_predict(const char *input_path, const char *output_path, geo_cycle_model_t *model);
int geo_cycle_export_c(const char *path, const char *symbol, const geo_cycle_model_t *model);
int geo_cycle_check(geo_cycle_model_t *model);

#endif
