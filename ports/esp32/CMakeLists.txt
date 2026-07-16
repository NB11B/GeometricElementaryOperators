idf_component_register(
    SRCS
        "../../src/cl20.c"
        "../../src/opposite.c"
        "../../src/omega.c"
        "../../src/structured.c"
        "../../src/structured_program.c"
        "../../src/control.c"
        "../../src/geb36.c"
        "../../src/compiler.c"
        "../../src/optimizer.c"
        "../../src/folding.c"
        "../../src/banked.c"
        "../../src/lowering.c"
        "../../src/geb_witness.c"
        "../../src/perf.c"
        "../../src/fixed.c"
        "geo_esp32.c"
    INCLUDE_DIRS
        "../../include"
        "."
    REQUIRES
        esp_timer
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC GEO_ESP32=1)
