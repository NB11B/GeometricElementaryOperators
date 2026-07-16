#include "geo/banked.h"
#include "geo/structured_program.h"

geo_status_t geo_struct_program_execute(
    const geo_struct_program_t *program,
    geo_struct_value_t *registers,
    size_t register_capacity
) {
    if (program == NULL || registers == NULL) return GEO_STATUS_NULL_ARGUMENT;
    if (program->instruction_count != 0u && program->instructions == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    return geo_struct_program_execute_impl(program, registers, register_capacity);
}

geo_status_t geo_banked_execute(
    const geo_banked_program_t *program,
    geo_banked_storage_t *storage
) {
    if (program == NULL || storage == NULL) return GEO_STATUS_NULL_ARGUMENT;
    if (program->instruction_count != 0u && program->instructions == NULL) {
        return GEO_STATUS_NULL_ARGUMENT;
    }
    return geo_banked_execute_impl(program, storage);
}
