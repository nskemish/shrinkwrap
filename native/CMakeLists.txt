pybind11_add_module(
    _native
    MODULE

    bindings.cpp
    rasterizer.cpp
    relief.cpp
    mesher.cpp
)

target_include_directories(
    _native
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(
    _native
    PRIVATE

    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:
        -O3
        -Wall
        -Wextra
        -Wpedantic
    >
)

#
# Modul ide direktno u core/, tako da Python može:
#
#     from core import _native
#
set_target_properties(
    _native
    PROPERTIES

    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_SOURCE_DIR}/core"
)
