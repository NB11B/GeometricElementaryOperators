set(GEO_ESP32_CORE_SRCS
    "../../src/cl20.c"
    "../../src/cl30.c"
    "../../src/opposite.c"
    "../../src/omega.c"
    "../../src/eml_embedded.c"
    "../../src/structured.c"
    "../../src/structured_program.c"
    "../../src/fused.c"
    "../../src/control.c"
    "../../src/geb36.c"
    "../../src/compiler.c"
    "../../src/optimizer.c"
    "../../src/folding.c"
    "../../src/banked.c"
    "../../src/lowering.c"
    "../../src/report.c"
    "../../src/geb_witness.c"
    "../../src/perf.c"
    "../../src/fixed.c"
    "../../src/fixed_geb36.c"
    "../../src/native_generated.c"
)

if(COMMAND idf_component_register)
    idf_component_register(
        SRCS
            ${GEO_ESP32_CORE_SRCS}
            "geo_esp32.c"
        INCLUDE_DIRS
            "../../include"
            "."
        REQUIRES
            esp_timer
    )
    set_source_files_properties("../../src/structured_program.c" PROPERTIES
        COMPILE_DEFINITIONS GEO_STRUCTURED_PROGRAM_IMPLEMENTATION=1
    )
    set_source_files_properties("../../src/banked.c" PROPERTIES
        COMPILE_DEFINITIONS GEO_BANKED_IMPLEMENTATION=1
    )
    target_compile_definitions(${COMPONENT_LIB} PUBLIC GEO_ESP32=1)
else()
    cmake_minimum_required(VERSION 3.16)
    project(geo_esp32_component_validation LANGUAGES C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_C_EXTENSIONS OFF)

    add_library(geo_esp32_core STATIC ${GEO_ESP32_CORE_SRCS})
    target_include_directories(geo_esp32_core PUBLIC "../../include")
    set_source_files_properties("../../src/structured_program.c" PROPERTIES
        COMPILE_DEFINITIONS GEO_STRUCTURED_PROGRAM_IMPLEMENTATION=1
    )
    set_source_files_properties("../../src/banked.c" PROPERTIES
        COMPILE_DEFINITIONS GEO_BANKED_IMPLEMENTATION=1
    )
    target_compile_definitions(geo_esp32_core PUBLIC GEO_ESP32_COMPONENT_VALIDATION=1)

    if(MSVC)
        target_compile_options(geo_esp32_core PRIVATE /W4)
    else()
        target_compile_options(geo_esp32_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
        target_link_libraries(geo_esp32_core PUBLIC m)
    endif()
endif()
