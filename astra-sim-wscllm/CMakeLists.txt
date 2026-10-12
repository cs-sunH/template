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

# Build only the vendored dependency targets needed by the simulator.
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/fmt")

# Keep spdlog on its bundled fmt (SPDLOG_FMT_EXTERNAL OFF): the vendored
# extern/helper/fmt and spdlog's bundled fmt are both fmt 11.0.2
# (FMT_VERSION 110002); using the bundled copy matches the known-good build
# configuration in astra-sim-sh_old.
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/spdlog")

# Keeping yaml-cpp in-tree makes a clean checkout reproducible without a
# configure-time network download.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/yaml-cpp")

# die-internal cost model library (die内仿真器设计文档 §8.1; wscllm rollout
# wiring, byte-identical vendor of the shared library). Builds the die_model
# static library, the die_model_cabi shared C ABI and their unit tests; the
# target is linked into AstraSim below so the runtime stages can adopt it.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/die_model")


# Find Protobuf
# TODO: We want to use the advanced semantics 'protobuf CONFIG REQUIRED'.
# However, this requires protobuf to be compiled from source,
# which is enabled only in Docker containers. (Our documentation makes users install protobuc from apt-get)
# Therefore, enable 'protobuf CONFIG REQUIRED' only when we are building in Docker.
# TODO: Find a better way to detect if protobuf has been compiled by source, OR
# find a way to enable 'protobuf CONFIG REQUIRED' regardless of how protubuf was installed.
if(DEFINED ENV{PROTOBUF_FROM_SOURCE} AND "$ENV{PROTOBUF_FROM_SOURCE}" STREQUAL "True")
    find_package(protobuf CONFIG REQUIRED)
else()
    find_package(Protobuf REQUIRED)
endif()

# Files to compile
# (the astraccl custom_collectives and chakra feeder_v3 globs were removed
# with the custom collectives / ET feeder path, retired with the offline
# static arm; retired custom config keys fail closed at parse time.
# CONFIGURE_DEPENDS (P3-3.9 E2, 2026-09-28): without it, adding a source
# file to an existing build tree is silently ignored until a manual
# reconfigure. The chakra third_party/utils glob was removed here as well:
# that directory holds only json.hpp/protolib.py (zero .cc files, zero
# includes of a bare <json.hpp> anywhere, zero protolib references), so the
# glob always matched nothing and its include directory was dead too.)
file(GLOB srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/logical_topology/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/collective_algorithm/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/common/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/schema/protobuf/*.cc")

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
target_include_directories(AstraSim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/)
target_include_directories(AstraSim PUBLIC ${Protobuf_INCLUDE_DIR})
# P3-3.9 E2 (2026-09-28): the "Same as above" include block was removed as
# dead code. In CONFIG mode (PROTOBUF_FROM_SOURCE=True, find_package(protobuf
# CONFIG)) no `Protobuf_*` variable exists at all, so `${Protobuf_INCLUDE_DIRS}`
# expanded empty and the branch was a no-op; the protobuf::libprotobuf imported
# target already carries its include directories (linked above). In MODULE
# mode FindProtobuf defines Protobuf_INCLUDE_DIRS := Protobuf_INCLUDE_DIR
# (already added one line above), so the branch merely duplicated it.

# Properties
set_target_properties(AstraSim PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
set_target_properties(AstraSim
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
)

option(ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST
       "Build the communicator-scoped OfflineGreedy schedule journal test" ON)
# 缺省 ON（2026-09-27，sched-D1/cpp-tests-E1 配套，对齐 F/WL）：测试体为
# 运行时 expect()+退出码制（无 NDEBUG 恒绿依赖），默认构建并注册 CTest，
# tests/run_all.sh 的 ctest 全量入口由此获得唯一常驻用例；显式
# -DASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST=OFF 可关。
if(ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST)
    enable_testing()
    add_executable(AstraSim_OfflineGreedyScheduleJournalTest
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/OfflineGreedyScheduleJournalTest.cc)
    target_link_libraries(AstraSim_OfflineGreedyScheduleJournalTest
        PRIVATE AstraSim)
    target_include_directories(AstraSim_OfflineGreedyScheduleJournalTest
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/helper)
    add_test(NAME AstraSim_OfflineGreedyScheduleJournalTest
        COMMAND AstraSim_OfflineGreedyScheduleJournalTest)
endif()
