CMAKE_MINIMUM_REQUIRED (VERSION 3.14)

project(
    atlasim
    VERSION 1.0
    LANGUAGES CXX
)

# ramulator2 vendors yaml-cpp 0.7.0 under simulator/ext/yaml-cpp. That copy
# still declares cmake_minimum_required(VERSION 3.4), which newer CMake
# releases reject unless the parent project raises the policy floor.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The patched ramulator2 submodule expects a top-level yaml-cpp target and
# yaml-cpp_SOURCE_DIR include root.
set(yaml-cpp_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd/yaml-cpp)

# Reuse vendored dependencies for both plain CMake builds and setuptools/CMake
# extension builds so we don't redownload spdlog/argparse on each path.
set(local_spdlog_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/dram/ramulator2/ext/spdlog)
if(EXISTS ${local_spdlog_source_dir}/CMakeLists.txt)
    set(FETCHCONTENT_SOURCE_DIR_SPDLOG ${local_spdlog_source_dir} CACHE PATH "Reuse vendored spdlog" FORCE)
endif()

set(local_argparse_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/dram/ramulator2/ext/argparse)
if(EXISTS ${local_argparse_source_dir}/CMakeLists.txt)
    set(FETCHCONTENT_SOURCE_DIR_ARGPARSE ${local_argparse_source_dir} CACHE PATH "Reuse vendored argparse" FORCE)
endif()

include_directories(3rd/yaml-cpp/include)
include_directories(src)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(3rd)
add_subdirectory(src)
add_subdirectory(tests)
