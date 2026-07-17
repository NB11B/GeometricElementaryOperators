#ifndef GEO_CUDA_H
#define GEO_CUDA_H

#include "geo/cl20.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_CUDA_SUCCESS = 0,
    GEO_CUDA_INVALID_ARGUMENT = 1,
    GEO_CUDA_NO_DEVICE = 2,
    GEO_CUDA_ALLOCATION_FAILED = 3,
    GEO_CUDA_COPY_FAILED = 4,
    GEO_CUDA_LAUNCH_FAILED = 5,
    GEO_CUDA_RUNTIME_FAILED = 6,
    GEO_CUDA_UNSUPPORTED_TOOLKIT = 7
} geo_cuda_status_t;

typedef struct geo_cuda_context geo_cuda_context_t;

typedef struct {
    char name[256];
    int device_index;
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    size_t global_memory_bytes;
    int runtime_version;
    int driver_version;
} geo_cuda_device_info_t;

const char *geo_cuda_status_string(geo_cuda_status_t status);

geo_cuda_status_t geo_cuda_context_create(
    int device_index,
    geo_cuda_context_t **context
);

void geo_cuda_context_destroy(geo_cuda_context_t *context);

geo_cuda_status_t geo_cuda_get_device_info(
    const geo_cuda_context_t *context,
    geo_cuda_device_info_t *info
);

geo_cuda_status_t geo_cuda_cl20_add_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_cl20_t *output,
    size_t count
);

geo_cuda_status_t geo_cuda_cl20_product_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_cl20_t *output,
    size_t count
);

geo_cuda_status_t geo_cuda_cl20_reverse_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *input,
    geo_cl20_t *output,
    size_t count
);

geo_cuda_status_t geo_cuda_cl20_vector_dot_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_real_t *output,
    size_t count
);

/*
 * Computes the scalar e12 coefficient of the wedge of each input-vector pair.
 * Unlike generated schedule launchers, this public batch API does not return a
 * full Cl(2,0) multivector for vector wedge.
 */
geo_cuda_status_t geo_cuda_cl20_vector_wedge_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *a,
    const geo_cl20_t *b,
    geo_real_t *output,
    size_t count
);

geo_cuda_status_t geo_cuda_cl20_rotor_action_batch(
    geo_cuda_context_t *context,
    const geo_cl20_t *rotor,
    const geo_cl20_t *value,
    geo_cl20_t *output,
    size_t count
);

#ifdef __cplusplus
}
#endif

#endif
