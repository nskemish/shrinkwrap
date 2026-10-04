pybind11_add_module(
    _native
    MODULE

    bindings.cpp

    components/rasterizer/rasterizer.cpp
    components/relief/relief.cpp
    components/regularizer/regularizer.cpp
    components/adaptive_mesher/adaptive_mesher.cpp
    components/qem_simplifier/qem_simplifier.cpp
)

target_include_directories(
    _native
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/components/rasterizer
    ${CMAKE_CURRENT_SOURCE_DIR}/components/relief
    ${CMAKE_CURRENT_SOURCE_DIR}/components/regularizer
    ${CMAKE_CURRENT_SOURCE_DIR}/components/adaptive_mesher
    ${CMAKE_CURRENT_SOURCE_DIR}/components/qem_simplifier
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

target_link_libraries(
    _native
    PRIVATE

    CGAL::CGAL
    Eigen3::Eigen
)

set_target_properties(
    _native
    PROPERTIES

    LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_SOURCE_DIR}/src/shrinkwrap"
)
