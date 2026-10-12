# # *****************************************************************************
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# #
# # Copyright (c) 2024 Georgia Institute of Technology
# # *****************************************************************************

# CMake Requirement
cmake_minimum_required(VERSION 3.22)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Setup project
project(AstraSim)

# Build only the vendored dependency targets needed by the simulator. Keeping
# yaml-cpp in-tree makes a clean checkout reproducible without a configure-time
# network download.
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/fmt")

# spdlog 1.14.1 is not source-compatible with this repository's customized
# fmt 11 headers when SPDLOG_FMT_EXTERNAL is enabled. Use spdlog's bundled fmt,
# matching the known-good build configuration in astra-sim-sh_old.
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/spdlog")

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/yaml-cpp")

# die-internal cost model library (die内仿真器设计文档 §8.1; stage A1 wiring).
# Builds the die_model static library and its unit tests; the target is linked
# into AstraSim below so the runtime stages can adopt it incrementally.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/die_model")


# Find Protobuf
# Status note (2026-09-26 review, replaces the two bare upstream TODOs):
# the advanced 'protobuf CONFIG REQUIRED' semantics require protobuf to be
# compiled from source, which is only enabled in Docker containers (the
# documented setup installs protobuf from apt-get), so CONFIG is selected
# only when PROTOBUF_FROM_SOURCE=True is set in the environment -- the
# env-var check below IS the source-built detection; no better detection is
# currently planned.
if(DEFINED ENV{PROTOBUF_FROM_SOURCE} AND "$ENV{PROTOBUF_FROM_SOURCE}" STREQUAL "True")
    find_package(protobuf CONFIG REQUIRED)
else()
    find_package(Protobuf REQUIRED)
endif()

# Files to compile
file(GLOB srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/*.cc"
    # (2026-09-30) Removed the native_collectives/*.cc glob entry: that
    # directory only holds the two subdirectories globbed right below and
    # the pattern matched zero files.
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/logical_topology/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/collective_algorithm/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/custom_collectives/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/common/*.cc"
    # third_party/utils/*.cc glob removed 2026-09-26: protoio.cc/.hh were
    # deleted (unused protobuf-io helper) and the directory holds no
    # sources, so the glob matched an empty set.
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/schema/protobuf/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/src/feeder_v3/*.cpp")

# Compile AstraSim Library
add_library(AstraSim STATIC ${srcs})

# Link libraries
target_link_libraries(AstraSim PUBLIC fmt::fmt)
target_link_libraries(AstraSim PUBLIC spdlog::spdlog)
target_link_libraries(AstraSim PUBLIC yaml-cpp)
# PUBLIC: AstraSim consumers see extern/die_model/include (DieModel.hh) and
# the shared extern/helper nlohmann/json include via die_model's interface.
target_link_libraries(AstraSim PUBLIC die_model)

# Same as above.
if(DEFINED ENV{PROTOBUF_FROM_SOURCE} AND "$ENV{PROTOBUF_FROM_SOURCE}" STREQUAL "True")
    target_link_libraries(AstraSim PUBLIC protobuf::libprotobuf)
else()
    target_link_libraries(AstraSim PUBLIC ${Protobuf_LIBRARIES})
endif()


# Include Directories
target_include_directories(AstraSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(AstraSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/network_backend/analytical/include/astra-network-analytical)
target_include_directories(AstraSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/)
target_include_directories(AstraSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/schema/protobuf)
target_include_directories(AstraSim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/src/third_party/utils)
target_include_directories(AstraSim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/)
target_include_directories(AstraSim PUBLIC ${Protobuf_INCLUDE_DIR})

# (2026-09-30) Removed the empty PROTOBUF_FROM_SOURCE branch: its only
# content was a ${Protobuf_INCLUDE_DIRS} line that is a FindProtobuf-module
# variable and EMPTY under the CONFIG package mode used here; the include
# paths arrive via the protobuf::libprotobuf target linkage.

# Properties
set_target_properties(AstraSim PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
set_target_properties(AstraSim
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
)

# ON by default (review 2026-09-26 infra-1/global-1): the journal test was
# previously opt-in and never enabled by any script, so it never ran. The
# fixture's checks are runtime expect() calls (no NDEBUG dependence), and the
# target registers into CTest whenever it is built.
option(ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST
       "Build the communicator-scoped OfflineGreedy schedule journal test" ON)
if(ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST)
    enable_testing()
    add_executable(AstraSim_OfflineGreedyScheduleJournalTest
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/OfflineGreedyScheduleJournalTest.cc)
    target_link_libraries(AstraSim_OfflineGreedyScheduleJournalTest
        PRIVATE AstraSim)
    target_include_directories(AstraSim_OfflineGreedyScheduleJournalTest
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/helper)
    set_target_properties(AstraSim_OfflineGreedyScheduleJournalTest
        PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    add_test(NAME AstraSim_OfflineGreedyScheduleJournalTest
        COMMAND AstraSim_OfflineGreedyScheduleJournalTest)
endif()
