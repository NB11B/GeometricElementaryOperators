#include "geo/structured.h"

#include <math.h>

static geo_real_t geo_sqrt_value(geo_real_t value) {
#if defined(GEO_REAL_IS_DOUBLE) && GEO_REAL_IS_DOUBLE
    return sqrt(value);
#else
    return sqrtf(value);
#endif
}

geo_unipotent_t geo_unipotent_from_cl20(geo_cl20_t value) {
    geo_unipotent_t result;
    result.payload = value;
    return result;
}

geo_unipotent_t geo_unipotent_identity(void) {
    return geo_unipotent_from_cl20(geo_cl20_zero());
}

geo_unipotent_t geo_unipotent_mul(
    geo_unipotent_t left,
    geo_unipotent_t right
) {
    return geo_unipotent_from_cl20(
        geo_cl20_add(left.payload, right.payload)
    );
}

geo_cl20_t geo_unipotent_extract(geo_unipotent_t value) {
    return value.payload;
}

geo_ordered_pair_t geo_ordered_products(geo_cl20_t a, geo_cl20_t b) {
    geo_ordered_pair_t result;
    result.ab = geo_cl20_mul(a, b);
    result.ba = geo_cl20_mul(b, a);
    return result;
}

geo_hadamard_pair_t geo_hadamard_mix_projective(
    geo_ordered_pair_t ordered
) {
    geo_hadamard_pair_t result;

    result.symmetric.represented = geo_cl20_add(ordered.ab, ordered.ba);
    result.symmetric.scale.numerator = 2;
    result.symmetric.scale.denominator = 1;

    result.antisymmetric.represented = geo_cl20_sub(ordered.ab, ordered.ba);
    result.antisymmetric.scale.numerator = 2;
    result.antisymmetric.scale.denominator = 1;

    return result;
}

geo_hadamard_pair_t geo_hadamard_mix_exact(geo_ordered_pair_t ordered) {
    geo_hadamard_pair_t result;
    const geo_real_t half = (geo_real_t)0.5;

    result.symmetric.represented = geo_cl20_scale(
        geo_cl20_add(ordered.ab, ordered.ba),
        half
    );
    result.symmetric.scale = geo_scale_one();

    result.antisymmetric.represented = geo_cl20_scale(
        geo_cl20_sub(ordered.ab, ordered.ba),
        half
    );
    result.antisymmetric.scale = geo_scale_one();

    return result;
}

geo_status_t geo_scaled_cl20_normalize(
    const geo_scaled_cl20_t *input,
    geo_scaled_cl20_t *output
) {
    geo_real_t factor;

    if (input == NULL || output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (input->scale.numerator == 0 || input->scale.denominator == 0) {
        return GEO_STATUS_SCALE_OVERFLOW;
    }

    factor = (geo_real_t)input->scale.denominator /
        (geo_real_t)input->scale.numerator;

    output->represented = geo_cl20_scale(input->represented, factor);
    output->scale = geo_scale_one();
    return GEO_STATUS_OK;
}

geo_status_t geo_vector_inverse(geo_cl20_t vector, geo_cl20_t *output) {
    const geo_real_t norm_squared = geo_cl20_vector_norm_squared(vector);

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (norm_squared <= (geo_real_t)0) {
        return GEO_STATUS_ZERO_NORM;
    }

    *output = geo_cl20_scale(vector, (geo_real_t)1 / norm_squared);
    return GEO_STATUS_OK;
}

geo_status_t geo_vector_normalize(geo_cl20_t vector, geo_cl20_t *output) {
    const geo_real_t norm_squared = geo_cl20_vector_norm_squared(vector);
    geo_real_t inverse_norm;

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (norm_squared <= (geo_real_t)0) {
        return GEO_STATUS_ZERO_NORM;
    }

    inverse_norm = (geo_real_t)1 / geo_sqrt_value(norm_squared);
    *output = geo_cl20_scale(vector, inverse_norm);
    return GEO_STATUS_OK;
}

geo_status_t geo_vector_projection(
    geo_cl20_t vector,
    geo_cl20_t onto,
    geo_cl20_t *output
) {
    const geo_real_t denominator = geo_cl20_vector_norm_squared(onto);
    geo_real_t coefficient;

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    if (denominator <= (geo_real_t)0) {
        return GEO_STATUS_ZERO_NORM;
    }

    coefficient = geo_cl20_vector_dot(vector, onto) / denominator;
    *output = geo_cl20_scale(onto, coefficient);
    return GEO_STATUS_OK;
}

geo_status_t geo_vector_rejection(
    geo_cl20_t vector,
    geo_cl20_t onto,
    geo_cl20_t *output
) {
    geo_cl20_t projection;
    geo_status_t status;

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    status = geo_vector_projection(vector, onto, &projection);
    if (status != GEO_STATUS_OK) {
        return status;
    }

    *output = geo_cl20_sub(vector, projection);
    return GEO_STATUS_OK;
}

geo_status_t geo_vector_reflection(
    geo_cl20_t vector,
    geo_cl20_t normal,
    geo_cl20_t *output
) {
    geo_cl20_t projection;
    geo_status_t status;

    if (output == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }

    status = geo_vector_projection(vector, normal, &projection);
    if (status != GEO_STATUS_OK) {
        return status;
    }

    *output = geo_cl20_sub(
        vector,
        geo_cl20_scale(projection, (geo_real_t)2)
    );
    return GEO_STATUS_OK;
}
