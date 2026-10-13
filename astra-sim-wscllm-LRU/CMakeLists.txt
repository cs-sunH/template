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

# Keeping yaml-cpp in-tree makes a clean checkout reproducible without a
# configure-time network download.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/helper/yaml-cpp")

# die-internal cost model library (die内仿真器设计文档 §8.1; vendored
# byte-identical with the face reference implementation). Builds the
# die_model static library, the die_model_cabi shared target and their unit
# tests; the target is linked into AstraSim below so the runtime stages can
# adopt it incrementally.
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

# Generate Chakra bindings in the build tree. A clean checkout contains the
# .proto source, not generated C++ files; use the selected Protobuf compiler
# and rebuild when that schema changes.
set(CHAKRA_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/schema/protobuf")
set(CHAKRA_PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/chakra")
set(CHAKRA_PROTO_CC "${CHAKRA_PROTO_OUTPUT_DIR}/et_def.pb.cc")
set(CHAKRA_PROTO_HH "${CHAKRA_PROTO_OUTPUT_DIR}/et_def.pb.h")
file(MAKE_DIRECTORY "${CHAKRA_PROTO_OUTPUT_DIR}")
add_custom_command(
    OUTPUT "${CHAKRA_PROTO_CC}" "${CHAKRA_PROTO_HH}"
    COMMAND protobuf::protoc
        "--proto_path=${CHAKRA_PROTO_DIR}"
        "--cpp_out=${CHAKRA_PROTO_OUTPUT_DIR}"
        "${CHAKRA_PROTO_DIR}/et_def.proto"
    DEPENDS "${CHAKRA_PROTO_DIR}/et_def.proto" protobuf::protoc
    VERBATIM)

# Files to compile
# H1(2026-09-30): CONFIGURE_DEPENDS 与另两份 CMakeLists(前端/后端)
# 同款——裸 GLOB 的新增 .cc 对增量构建不可见(链接错/行为陈旧、零诊断)。
file(GLOB srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/logical_topology/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/collective_algorithm/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/custom_collectives/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/common/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/src/third_party/utils/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/src/feeder_v3/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/remote_memory_backend/analytical/*.cc")

# Compile AstraSim Library
add_library(AstraSim STATIC ${srcs} "${CHAKRA_PROTO_CC}" "${CHAKRA_PROTO_HH}")
target_include_directories(AstraSim PUBLIC "${CHAKRA_PROTO_OUTPUT_DIR}")

# Link libraries
target_link_libraries(AstraSim PUBLIC fmt::fmt)
target_link_libraries(AstraSim PUBLIC spdlog::spdlog)
target_link_libraries(AstraSim PUBLIC yaml-cpp)
# PUBLIC: AstraSim consumers see extern/die_model/include (DieModel.hh) and
# the shared extern/helper nlohmann/json include via die_model's interface.
target_link_libraries(AstraSim PUBLIC die_model)
# Link the coarse_die_v2 C++ runtime while preserving die_model for explicit
# legacy mesh_die_v1/tx8_reference_v1 profiles.
target_link_libraries(AstraSim PUBLIC coarse_die)

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

# Same as above.
if(DEFINED ENV{PROTOBUF_FROM_SOURCE} AND "$ENV{PROTOBUF_FROM_SOURCE}" STREQUAL "True")
    target_include_directories(AstraSim PUBLIC ${Protobuf_INCLUDE_DIRS})
endif()

# Properties
set_target_properties(AstraSim PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
set_target_properties(AstraSim
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
)

# Enable testing unconditionally at top-level scope (F2/TEST-01 scope fix,
# 2026-09-26): add_test() registrations in this tree must never depend on a
# build option being ON.
enable_testing()

option(ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST
       "Build the communicator-scoped OfflineGreedy schedule journal test" ON)
if(ASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST)
    add_executable(AstraSim_OfflineGreedyScheduleJournalTest
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/OfflineGreedyScheduleJournalTest.cc)
    target_link_libraries(AstraSim_OfflineGreedyScheduleJournalTest
        PRIVATE AstraSim)
    target_include_directories(AstraSim_OfflineGreedyScheduleJournalTest
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/helper)
    add_test(NAME AstraSim_OfflineGreedyScheduleJournalTest
        COMMAND AstraSim_OfflineGreedyScheduleJournalTest)
endif()
