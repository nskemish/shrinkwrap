cmake_minimum_required(VERSION 3.20)

project(
    ShrinkWrapNative
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    Python
    COMPONENTS
        Interpreter
        Development.Module
    REQUIRED
)

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(
    pybind11
    CONFIG
    REQUIRED
)

add_subdirectory(native)
