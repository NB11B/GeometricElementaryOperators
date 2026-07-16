idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES
        esp32
        esp_system
        esp_timer
        heap
)
