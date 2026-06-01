# Main component for ESP-IDF build.
FILE(GLOB_RECURSE app_sources "*.c" "*.cpp" "*.cc" "*.S")

idf_component_register(
    SRCS ${app_sources}
    INCLUDE_DIRS "." "../include"
    REQUIRES esp-dsp esp32-camera
)
