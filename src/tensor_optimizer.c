#include "geo/tensor_optimizer.h"

#include <math.h>

static int geo_tensor_adamw_config_valid(geo_tensor_adamw_config config) {
    return config.learning_rate >= (geo_real_t)0 &&
           config.beta1 >= (geo_real_t)0 && config.beta1 < (geo_real_t)1 &&
           config.beta2 >= (geo_real_t)0 && config.beta2 < (geo_real_t)1 &&
           config.epsilon > (geo_real_t)0 &&
           config.weight_decay >= (geo_real_t)0 &&
           config.max_grad_norm >= (geo_real_t)0 &&
           config.step > 0u;
}

geo_tensor_status geo_tensor_grad_square_accumulate(
    const geo_real_t *gradient,
    size_t count,
    geo_real_t *sum_square
) {
    if (gradient == NULL || sum_square == NULL || count == 0u) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    geo_real_t sum = *sum_square;
    for (size_t index = 0u; index < count; ++index) {
        sum += gradient[index] * gradient[index];
    }
    *sum_square = sum;
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_grad_clip_scale(
    geo_real_t sum_square,
    geo_real_t max_grad_norm,
    geo_real_t *clip_scale
) {
    if (clip_scale == NULL || sum_square < (geo_real_t)0 ||
        max_grad_norm < (geo_real_t)0) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }
    if (max_grad_norm == (geo_real_t)0) {
        *clip_scale = (geo_real_t)1;
        return GEO_TENSOR_OK;
    }
    const geo_real_t norm = (geo_real_t)sqrt((double)sum_square);
    const geo_real_t candidate = max_grad_norm /
        (norm + (geo_real_t)1e-6);
    *clip_scale = candidate < (geo_real_t)1 ? candidate : (geo_real_t)1;
    return GEO_TENSOR_OK;
}

geo_tensor_status geo_tensor_adamw_step(
    geo_real_t *parameter,
    const geo_real_t *gradient,
    geo_real_t *first_moment,
    geo_real_t *second_moment,
    size_t count,
    geo_real_t clip_scale,
    geo_tensor_adamw_config config
) {
    if (parameter == NULL || gradient == NULL || first_moment == NULL ||
        second_moment == NULL || count == 0u || clip_scale < (geo_real_t)0 ||
        clip_scale > (geo_real_t)1 || !geo_tensor_adamw_config_valid(config)) {
        return GEO_TENSOR_INVALID_ARGUMENT;
    }

    const geo_real_t one = (geo_real_t)1;
    const geo_real_t bias_correction1 = one - (geo_real_t)pow(
        (double)config.beta1, (double)config.step
    );
    const geo_real_t bias_correction2 = one - (geo_real_t)pow(
        (double)config.beta2, (double)config.step
    );
    const geo_real_t decay = one - config.learning_rate * config.weight_decay;

    for (size_t index = 0u; index < count; ++index) {
        const geo_real_t grad = gradient[index] * clip_scale;
        const geo_real_t moment1 =
            config.beta1 * first_moment[index] +
            (one - config.beta1) * grad;
        const geo_real_t moment2 =
            config.beta2 * second_moment[index] +
            (one - config.beta2) * grad * grad;
        first_moment[index] = moment1;
        second_moment[index] = moment2;

        const geo_real_t corrected1 = moment1 / bias_correction1;
        const geo_real_t corrected2 = moment2 / bias_correction2;
        parameter[index] = parameter[index] * decay -
            config.learning_rate * corrected1 /
            ((geo_real_t)sqrt((double)corrected2) + config.epsilon);
    }
    return GEO_TENSOR_OK;
}
