#include "geo/batch_gp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int batch_dimension_valid(uint8_t dimension) {
    return dimension >= 1u && dimension <= GEO_OPERATOR_MAX_DIMENSION;
}

static int batch_signature_valid(const int8_t *signature, uint8_t dimension) {
    uint8_t index;
    if (signature == NULL || !batch_dimension_valid(dimension)) return 0;
    for (index = 0u; index < dimension; ++index) {
        if (signature[index] != 1 && signature[index] != -1) return 0;
    }
    return 1;
}

static unsigned batch_popcount(uint8_t value) {
    unsigned count = 0u;
    while (value != 0u) {
        value = (uint8_t)(value & (uint8_t)(value - 1u));
        ++count;
    }
    return count;
}

static int batch_gp_sign(
    uint8_t left,
    uint8_t right,
    const int8_t *signature,
    uint8_t dimension
) {
    int sign = 1;
    uint8_t axis;
    for (axis = 0u; axis < dimension; ++axis) {
        if (((left >> axis) & 1u) != 0u) {
            const uint8_t lower = (uint8_t)(right & (uint8_t)((1u << axis) - 1u));
            if ((batch_popcount(lower) & 1u) != 0u) sign = -sign;
            if (((right >> axis) & 1u) != 0u) sign *= signature[axis];
        }
    }
    return sign;
}

static int batch_plan_valid(const geo_batch_gp_plan_t *plan) {
    const size_t expected_blades = plan == NULL ? 0u : (size_t)1u << plan->dimension;
    return plan != NULL &&
        plan->abi_version == GEO_BATCH_GP_ABI_VERSION &&
        batch_signature_valid(plan->signature, plan->dimension) &&
        plan->blade_count == expected_blades &&
        plan->term_count == expected_blades * expected_blades;
}

static int batch_values_finite(const double *values, size_t count) {
    size_t index;
    if (values == NULL) return 0;
    for (index = 0u; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

uint32_t geo_batch_gp_abi_version(void) {
    return GEO_BATCH_GP_ABI_VERSION;
}

geo_batch_gp_status_t geo_batch_gp_plan_init(
    geo_batch_gp_plan_t *plan,
    uint8_t dimension,
    const int8_t *signature
) {
    size_t left;
    size_t right;
    size_t term = 0u;
    const size_t blade_count = batch_dimension_valid(dimension) ? (size_t)1u << dimension : 0u;

    if (plan == NULL || signature == NULL) return GEO_BATCH_GP_INVALID_ARGUMENT;
    if (!batch_dimension_valid(dimension)) return GEO_BATCH_GP_UNSUPPORTED_DIMENSION;
    if (!batch_signature_valid(signature, dimension)) return GEO_BATCH_GP_INVALID_ARGUMENT;

    memset(plan, 0, sizeof(*plan));
    plan->abi_version = GEO_BATCH_GP_ABI_VERSION;
    plan->dimension = dimension;
    memcpy(plan->signature, signature, (size_t)dimension * sizeof(signature[0]));
    plan->blade_count = (uint16_t)blade_count;
    plan->term_count = (uint16_t)(blade_count * blade_count);

    for (left = 0u; left < blade_count; ++left) {
        for (right = 0u; right < blade_count; ++right) {
            plan->left_blade[term] = (uint8_t)left;
            plan->right_blade[term] = (uint8_t)right;
            plan->output_blade[term] = (uint8_t)(left ^ right);
            plan->sign[term] = (int8_t)batch_gp_sign(
                (uint8_t)left,
                (uint8_t)right,
                signature,
                dimension
            );
            ++term;
        }
    }
    return GEO_BATCH_GP_OK;
}

static geo_batch_gp_status_t batch_forward(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    size_t batch_size,
    const double *parameter,
    int parameter_on_left,
    double *outputs
) {
    size_t term;
    size_t sample;
    const size_t blades = plan == NULL ? 0u : plan->blade_count;
    const size_t values = batch_size * blades;

    if (!batch_plan_valid(plan) || batch_size == 0u || outputs == NULL ||
        !batch_values_finite(inputs, values) ||
        !batch_values_finite(parameter, blades)) {
        return GEO_BATCH_GP_INVALID_ARGUMENT;
    }
    memset(outputs, 0, values * sizeof(outputs[0]));

    for (term = 0u; term < plan->term_count; ++term) {
        const size_t input_blade = parameter_on_left != 0 ?
            plan->right_blade[term] : plan->left_blade[term];
        const size_t parameter_blade = parameter_on_left != 0 ?
            plan->left_blade[term] : plan->right_blade[term];
        const size_t output_blade = plan->output_blade[term];
        const double multiplier = parameter[parameter_blade] * (double)plan->sign[term];
        for (sample = 0u; sample < batch_size; ++sample) {
            outputs[sample * blades + output_blade] +=
                inputs[sample * blades + input_blade] * multiplier;
        }
    }
    if (!batch_values_finite(outputs, values)) return GEO_BATCH_GP_NUMERIC_FAILURE;
    return GEO_BATCH_GP_OK;
}

geo_batch_gp_status_t geo_batch_gp_right_forward_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    size_t batch_size,
    const double *right_parameter,
    double *outputs
) {
    return batch_forward(plan, inputs, batch_size, right_parameter, 0, outputs);
}

geo_batch_gp_status_t geo_batch_gp_left_forward_f64(
    const geo_batch_gp_plan_t *plan,
    const double *left_parameter,
    const double *inputs,
    size_t batch_size,
    double *outputs
) {
    return batch_forward(plan, inputs, batch_size, left_parameter, 1, outputs);
}

static geo_batch_gp_status_t batch_parameter_vjp(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *output_cotangents,
    size_t batch_size,
    int parameter_on_left,
    double *parameter_cotangent
) {
    size_t term;
    size_t sample;
    const size_t blades = plan == NULL ? 0u : plan->blade_count;
    const size_t values = batch_size * blades;

    if (!batch_plan_valid(plan) || batch_size == 0u || parameter_cotangent == NULL ||
        !batch_values_finite(inputs, values) ||
        !batch_values_finite(output_cotangents, values)) {
        return GEO_BATCH_GP_INVALID_ARGUMENT;
    }
    memset(parameter_cotangent, 0, blades * sizeof(parameter_cotangent[0]));

    for (term = 0u; term < plan->term_count; ++term) {
        const size_t input_blade = parameter_on_left != 0 ?
            plan->right_blade[term] : plan->left_blade[term];
        const size_t parameter_blade = parameter_on_left != 0 ?
            plan->left_blade[term] : plan->right_blade[term];
        const size_t output_blade = plan->output_blade[term];
        const double sign = (double)plan->sign[term];
        double sum = 0.0;
        for (sample = 0u; sample < batch_size; ++sample) {
            sum += inputs[sample * blades + input_blade] *
                output_cotangents[sample * blades + output_blade];
        }
        parameter_cotangent[parameter_blade] += sign * sum;
    }
    if (!batch_values_finite(parameter_cotangent, blades)) {
        return GEO_BATCH_GP_NUMERIC_FAILURE;
    }
    return GEO_BATCH_GP_OK;
}

geo_batch_gp_status_t geo_batch_gp_right_vjp_parameter_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *output_cotangents,
    size_t batch_size,
    double *parameter_cotangent
) {
    return batch_parameter_vjp(
        plan,
        inputs,
        output_cotangents,
        batch_size,
        0,
        parameter_cotangent
    );
}

geo_batch_gp_status_t geo_batch_gp_left_vjp_parameter_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *output_cotangents,
    size_t batch_size,
    double *parameter_cotangent
) {
    return batch_parameter_vjp(
        plan,
        inputs,
        output_cotangents,
        batch_size,
        1,
        parameter_cotangent
    );
}

static geo_batch_gp_status_t batch_mse_sgd_step(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *targets,
    size_t batch_size,
    double learning_rate,
    int parameter_on_left,
    double *parameter,
    double *mean_loss
) {
    double *residuals;
    double gradient[GEO_OPERATOR_MAX_BLADES];
    size_t index;
    double loss = 0.0;
    const size_t blades = plan == NULL ? 0u : plan->blade_count;
    const size_t values = batch_size * blades;
    geo_batch_gp_status_t status;

    if (!batch_plan_valid(plan) || batch_size == 0u || !isfinite(learning_rate) ||
        learning_rate <= 0.0 || mean_loss == NULL ||
        !batch_values_finite(inputs, values) ||
        !batch_values_finite(targets, values) ||
        !batch_values_finite(parameter, blades)) {
        return GEO_BATCH_GP_INVALID_ARGUMENT;
    }

    residuals = (double *)malloc(values * sizeof(residuals[0]));
    if (residuals == NULL) return GEO_BATCH_GP_INVALID_ARGUMENT;

    status = batch_forward(
        plan,
        inputs,
        batch_size,
        parameter,
        parameter_on_left,
        residuals
    );
    if (status != GEO_BATCH_GP_OK) {
        free(residuals);
        return status;
    }
    for (index = 0u; index < values; ++index) {
        residuals[index] -= targets[index];
        loss += 0.5 * residuals[index] * residuals[index];
    }
    status = batch_parameter_vjp(
        plan,
        inputs,
        residuals,
        batch_size,
        parameter_on_left,
        gradient
    );
    if (status != GEO_BATCH_GP_OK) {
        free(residuals);
        return status;
    }
    for (index = 0u; index < blades; ++index) {
        const double candidate = parameter[index] -
            learning_rate * gradient[index] / (double)batch_size;
        if (!isfinite(candidate)) {
            free(residuals);
            return GEO_BATCH_GP_NUMERIC_FAILURE;
        }
    }
    for (index = 0u; index < blades; ++index) {
        parameter[index] -= learning_rate * gradient[index] / (double)batch_size;
    }
    *mean_loss = loss / (double)batch_size;
    free(residuals);
    return isfinite(*mean_loss) ? GEO_BATCH_GP_OK : GEO_BATCH_GP_NUMERIC_FAILURE;
}

geo_batch_gp_status_t geo_batch_gp_right_mse_sgd_step_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *targets,
    size_t batch_size,
    double learning_rate,
    double *right_parameter,
    double *mean_loss
) {
    return batch_mse_sgd_step(
        plan,
        inputs,
        targets,
        batch_size,
        learning_rate,
        0,
        right_parameter,
        mean_loss
    );
}

geo_batch_gp_status_t geo_batch_gp_left_mse_sgd_step_f64(
    const geo_batch_gp_plan_t *plan,
    const double *inputs,
    const double *targets,
    size_t batch_size,
    double learning_rate,
    double *left_parameter,
    double *mean_loss
) {
    return batch_mse_sgd_step(
        plan,
        inputs,
        targets,
        batch_size,
        learning_rate,
        1,
        left_parameter,
        mean_loss
    );
}
