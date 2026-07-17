#ifndef GEO_CORTEX_M_DWT_H
#define GEO_CORTEX_M_DWT_H

#include <stdbool.h>
#include <stdint.h>

#include "geo/perf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GEO_CORTEX_M_DWT_OK = 0,
    GEO_CORTEX_M_DWT_NULL_ARGUMENT = 1,
    GEO_CORTEX_M_DWT_UNSUPPORTED = 2,
    GEO_CORTEX_M_DWT_BAD_CLOCK = 3,
    GEO_CORTEX_M_DWT_ENABLE_FAILED = 4
} geo_cortex_m_dwt_status_t;

/*
 * Vendor-neutral register view. CMSIS users normally bind these fields to
 * &CoreDebug->DEMCR, &DWT->CTRL, and &DWT->CYCCNT.
 */
typedef struct {
    volatile uint32_t *demcr;
    volatile uint32_t *dwt_ctrl;
    volatile uint32_t *dwt_cyccnt;
    uint32_t demcr_trace_enable_mask;
    uint32_t dwt_cycle_enable_mask;
} geo_cortex_m_dwt_registers_t;

typedef struct {
    geo_cortex_m_dwt_registers_t registers;
    uint64_t wrap_base;
    uint32_t previous_counter;
    bool initialized;
} geo_cortex_m_dwt_context_t;

/*
 * Enables CYCCNT, resets it to zero, and returns a source suitable for the
 * existing geo_benchmark_* APIs. The caller owns both context and registers.
 */
geo_cortex_m_dwt_status_t geo_cortex_m_dwt_start(
    geo_cortex_m_dwt_context_t *context,
    const geo_cortex_m_dwt_registers_t *registers,
    uint64_t core_clock_hz,
    geo_cycle_source_t *source
);

/*
 * Extends the 32-bit CYCCNT register to a software 64-bit count. Calls must be
 * made more frequently than one complete 32-bit hardware wrap period.
 */
uint64_t geo_cortex_m_dwt_now(void *context);

#ifdef __cplusplus
}
#endif

#endif
