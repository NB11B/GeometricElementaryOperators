#include "geo/cortex_m_dwt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static void test_start_and_ticks(void) {
    volatile uint32_t demcr = UINT32_C(0x10);
    volatile uint32_t ctrl = UINT32_C(0x20);
    volatile uint32_t cyccnt = UINT32_C(123);
    const geo_cortex_m_dwt_registers_t registers = {
        &demcr,
        &ctrl,
        &cyccnt,
        UINT32_C(1) << 24,
        UINT32_C(1)
    };
    geo_cortex_m_dwt_context_t context;
    geo_cycle_source_t source;

    memset(&context, 0, sizeof(context));
    memset(&source, 0, sizeof(source));
    expect(
        geo_cortex_m_dwt_start(&context, &registers, UINT64_C(168000000), &source) ==
            GEO_CORTEX_M_DWT_OK,
        "DWT start succeeds with modeled registers"
    );
    expect((demcr & (UINT32_C(1) << 24)) != 0u, "trace enable bit is set");
    expect((ctrl & UINT32_C(1)) != 0u, "cycle counter enable bit is set");
    expect(cyccnt == 0u, "cycle counter is reset");
    expect(source.now == geo_cortex_m_dwt_now, "cycle source binds DWT reader");
    expect(source.context == &context, "cycle source retains caller context");
    expect(source.frequency_hz == UINT64_C(168000000), "cycle source records core clock");

    cyccnt = UINT32_C(10);
    expect(source.now(source.context) == UINT64_C(10), "first counter value is read");
    cyccnt = UINT32_C(25);
    expect(source.now(source.context) == UINT64_C(25), "monotonic counter advances");
}

static void test_wrap_extension(void) {
    volatile uint32_t demcr = 0u;
    volatile uint32_t ctrl = 0u;
    volatile uint32_t cyccnt = 0u;
    const geo_cortex_m_dwt_registers_t registers = {
        &demcr,
        &ctrl,
        &cyccnt,
        UINT32_C(1) << 24,
        UINT32_C(1)
    };
    geo_cortex_m_dwt_context_t context;
    geo_cycle_source_t source;

    memset(&context, 0, sizeof(context));
    expect(
        geo_cortex_m_dwt_start(&context, &registers, UINT64_C(480000000), &source) ==
            GEO_CORTEX_M_DWT_OK,
        "wrap fixture starts"
    );
    cyccnt = UINT32_C(0xfffffff0);
    expect(source.now(source.context) == UINT64_C(0xfffffff0),
        "pre-wrap value is preserved");
    cyccnt = UINT32_C(0x00000010);
    expect(source.now(source.context) == UINT64_C(0x100000010),
        "single hardware wrap extends to 64 bits");
    cyccnt = UINT32_C(0x00000020);
    expect(source.now(source.context) == UINT64_C(0x100000020),
        "post-wrap values remain monotonic");
}

static void test_invalid_configuration(void) {
    volatile uint32_t register_value = 0u;
    geo_cortex_m_dwt_registers_t registers = {
        &register_value,
        &register_value,
        &register_value,
        UINT32_C(1) << 24,
        UINT32_C(1)
    };
    geo_cortex_m_dwt_context_t context;
    geo_cycle_source_t source;

    memset(&context, 0, sizeof(context));
    expect(
        geo_cortex_m_dwt_start(NULL, &registers, 1u, &source) ==
            GEO_CORTEX_M_DWT_NULL_ARGUMENT,
        "null context is rejected"
    );
    expect(
        geo_cortex_m_dwt_start(&context, &registers, 0u, &source) ==
            GEO_CORTEX_M_DWT_BAD_CLOCK,
        "zero core clock is rejected"
    );
    registers.dwt_cyccnt = NULL;
    expect(
        geo_cortex_m_dwt_start(&context, &registers, 1u, &source) ==
            GEO_CORTEX_M_DWT_UNSUPPORTED,
        "missing DWT register is reported as unsupported"
    );
    expect(geo_cortex_m_dwt_now(NULL) == UINT64_C(0),
        "null cycle-reader context is harmless");
}

int main(void) {
    test_start_and_ticks();
    test_wrap_extension();
    test_invalid_configuration();

    if (failures != 0) {
        fprintf(stderr, "%d Cortex-M DWT assertion(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    puts("All Cortex-M DWT tests passed.");
    return EXIT_SUCCESS;
}
