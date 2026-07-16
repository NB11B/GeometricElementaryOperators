#include "geo/cortex_m_dwt.h"

#include <stddef.h>

uint64_t geo_cortex_m_dwt_now(void *opaque_context) {
    geo_cortex_m_dwt_context_t *context =
        (geo_cortex_m_dwt_context_t *)opaque_context;
    uint32_t current;

    if (context == NULL || !context->initialized ||
        context->registers.dwt_cyccnt == NULL) {
        return UINT64_C(0);
    }

    current = *context->registers.dwt_cyccnt;
    if (current < context->previous_counter) {
        context->wrap_base += UINT64_C(1) << 32;
    }
    context->previous_counter = current;
    return context->wrap_base + (uint64_t)current;
}

geo_cortex_m_dwt_status_t geo_cortex_m_dwt_start(
    geo_cortex_m_dwt_context_t *context,
    const geo_cortex_m_dwt_registers_t *registers,
    uint64_t core_clock_hz,
    geo_cycle_source_t *source
) {
    if (context == NULL || registers == NULL || source == NULL) {
        return GEO_CORTEX_M_DWT_NULL_ARGUMENT;
    }
    if (core_clock_hz == UINT64_C(0)) {
        return GEO_CORTEX_M_DWT_BAD_CLOCK;
    }
    if (registers->demcr == NULL || registers->dwt_ctrl == NULL ||
        registers->dwt_cyccnt == NULL ||
        registers->demcr_trace_enable_mask == UINT32_C(0) ||
        registers->dwt_cycle_enable_mask == UINT32_C(0)) {
        return GEO_CORTEX_M_DWT_UNSUPPORTED;
    }

    *registers->demcr |= registers->demcr_trace_enable_mask;
    *registers->dwt_cyccnt = UINT32_C(0);
    *registers->dwt_ctrl |= registers->dwt_cycle_enable_mask;

    if ((*registers->demcr & registers->demcr_trace_enable_mask) == UINT32_C(0) ||
        (*registers->dwt_ctrl & registers->dwt_cycle_enable_mask) == UINT32_C(0)) {
        return GEO_CORTEX_M_DWT_ENABLE_FAILED;
    }

    context->registers = *registers;
    context->wrap_base = UINT64_C(0);
    context->previous_counter = UINT32_C(0);
    context->initialized = true;

    source->now = geo_cortex_m_dwt_now;
    source->context = context;
    source->frequency_hz = core_clock_hz;
    return GEO_CORTEX_M_DWT_OK;
}
