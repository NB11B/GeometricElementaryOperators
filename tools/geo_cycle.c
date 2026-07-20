#include "geo_cycle_internal.h"

#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    geo_cycle_model_t model;
    int success = 0;
    if (argc < 2) {
        geo_cycle_usage(stderr);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("geo_cycle %s GEO_V8_ABI=0x%08x\n", GEO_CYCLE_VERSION, geo_v8_abi_version());
        return EXIT_SUCCESS;
    }
    if (strcmp(argv[1], "check") == 0 && argc == 3) {
        if (geo_cycle_load_model(argv[2], &model)) {
            success = geo_cycle_check(&model);
            geo_cycle_model_free(&model);
        }
    } else if (strcmp(argv[1], "train") == 0 && (argc == 5 || argc == 6)) {
        if (geo_cycle_load_model(argv[2], &model)) {
            success = geo_cycle_train(argv[3], argv[4], argc == 6 ? argv[5] : NULL, &model);
            geo_cycle_model_free(&model);
        }
    } else if (strcmp(argv[1], "predict") == 0 && argc == 6) {
        if (geo_cycle_load_model(argv[2], &model)) {
            if (geo_cycle_load_checkpoint(argv[3], &model)) success = geo_cycle_predict(argv[4], argv[5], &model);
            geo_cycle_model_free(&model);
        }
    } else if (strcmp(argv[1], "export-c") == 0 && argc == 6) {
        if (geo_cycle_load_model(argv[2], &model)) {
            if (geo_cycle_load_checkpoint(argv[3], &model)) success = geo_cycle_export_c(argv[4], argv[5], &model);
            geo_cycle_model_free(&model);
        }
    } else {
        geo_cycle_usage(stderr);
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
