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

# die-internal cost model library (die内仿真器设计文档 §8.1; joint rollout
# stage 1, vendored byte-identical with the face reference). Builds the
# die_model static library, the die_model_cabi shared C ABI target (consumed
# by sh_test_mesh/config_resolver.py through ctypes) and the library unit
# tests; the target is linked into AstraSim below so the runtime stages can
# adopt it incrementally.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/die_model")

# Native joint dual-ledger receiver (joint加速改造 v4): the shared C ABI
# target libjoint_ledger_cabi.so is consumed by the Python facade
# (sh_test_mesh/.../joint/native_ledger.py via ctypes).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/joint_ledger")


# Joint v7 event long-tail attribution (joint改造方案v7 §4): diagnostic-only
# callback-classification build. Adds the JOINT_V7_EVENT_PROFILE compile
# definition consumed by EventProfile.h brackets (backend queue phases, Sys
# sub-dispatch labels). Never changes simulation behavior; default OFF for
# every production build. The same option block is mirrored in
# extern/network_backend/analytical/CMakeLists.txt and
# astra-sim/network_frontend/analytical/CMakeLists.txt (assembly builds add
# each directory independently, so the cache option is declared here once
# and only consumed there).
option(JOINT_EVENT_PROFILE
       "Diagnostic event-classification instrumentation (v7 attribution)" OFF)
if(JOINT_EVENT_PROFILE)
    add_compile_definitions(JOINT_V7_EVENT_PROFILE)
endif()

# Find Protobuf
# TODO: We want to use the advanced semantics 'protobuf CONFIG REQUIRED'.
# However, this requires protobuf to be compiled from source,
# which is enabled only in Docker containers. (Our documentation makes users install protobuc from apt-get)
# Therefore, enable 'protobuf CONFIG REQUIRED' only when we are building in Docker.
# TODO: Find a better way to detect if protobuf has been compiled by source, OR
# find a way to enable 'protobuf CONFIG REQUIRED' regardless of how protubuf was installed.
# R5-F7 note (ported from the face repo, 2026-09-30): the comparison is a
# case-sensitive exact match on the string "True". Intuitive spellings
# ("true", "TRUE", "1") silently take the system-protobuf branch instead of
# erroring -- set exactly PROTOBUF_FROM_SOURCE=True or unset it.
if(DEFINED ENV{PROTOBUF_FROM_SOURCE} AND "$ENV{PROTOBUF_FROM_SOURCE}" STREQUAL "True")
    find_package(protobuf CONFIG REQUIRED)
else()
    find_package(Protobuf REQUIRED)
endif()

# Generate Chakra bindings in the build tree. A clean checkout contains the
# .proto source, not generated C++ files; compile exactly this generated copy
# instead of relying on a source-tree wildcard.
set(CHAKRA_PROTO_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/schema/protobuf")
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
# L24(2026-09-29 审查修复):GLOB 加 CONFIGURE_DEPENDS——增删源文件后不再
# 依赖手工重跑 cmake 才能被发现(对齐 face-LRU 仓同位置修复)。
file(GLOB srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/logical_topology/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/collective_algorithm/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/common/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/src/third_party/utils/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/graph_frontend/chakra/src/feeder_v3/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/remote_memory_backend/analytical/*.cc")

# Joint window algebra + native projection core (D4/D5: shrank to the die
# timer-window kernel and the projection engine when the remote-read tile
# descriptors were retired; the PipelineKernelContract implementation lives
# in PipelineKernel.cc). Keep these sources out of both the root source glob
# and the analytical frontend's execution-driven glob so an executable never
# receives duplicate definitions.
set(joint_pipeline_core_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/PipelineKernel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/NativeProjection.cc")
list(REMOVE_ITEM srcs ${joint_pipeline_core_sources})
add_library(joint_pipeline_core STATIC ${joint_pipeline_core_sources})
target_include_directories(joint_pipeline_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/extern/helper")
set_target_properties(joint_pipeline_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_link_libraries(joint_pipeline_core PUBLIC die_model)
target_compile_options(joint_pipeline_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off>)

set(joint_projection_api_source
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/NativeProjectionCApi.cc")
list(REMOVE_ITEM srcs ${joint_projection_api_source})
add_library(joint_projection SHARED ${joint_projection_api_source})
target_link_libraries(joint_projection PRIVATE joint_pipeline_core)
target_include_directories(joint_projection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(joint_projection PRIVATE JOINT_PROJECTION_BUILD)
set_target_properties(joint_projection PROPERTIES
    OUTPUT_NAME joint_projection
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

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
target_link_libraries(AstraSim PUBLIC coarse_die)
target_link_libraries(AstraSim PUBLIC joint_pipeline_core)
# v6 compact round: the execution-driven producer builds jointledger typed
# records and shares the compact wire codec with the native receiver
# (PUBLIC: the joint_ledger interface carries its own include paths).
target_link_libraries(AstraSim PUBLIC joint_ledger)

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

# CTest entry point (die rollout stage 1): this repository's default
# configure historically had zero ctest entries (regression runs drive the
# test binaries directly by path, per README §6), so no root-level
# enable_testing() existed. The vendored extern/die_model registers its own
# library tests and the die backend registers its three hermetic frontend
# fixtures through add_test(); enable_testing() here (after the vendored
# add_subdirectory() blocks, same ordering discipline as the die_model
# CMakeLists note) makes both visible to a plain `ctest` run from the build
# tree without touching the explicit-binary regression convention.
enable_testing()

# Bounded integer window operator and mutation/invalidation counters, plus
# the migrated PipelineKernelContract parse/validate assertions. This is
# deliberately linked to the same immutable kernel library used by
# NativeProjection and the online runtime.
add_executable(JointPipelineWindowTest
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/tests/pipeline_window_test.cc")
target_link_libraries(JointPipelineWindowTest PRIVATE joint_pipeline_core)
target_compile_options(JointPipelineWindowTest PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off>)
set_target_properties(JointPipelineWindowTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME JointPipelineWindowTest COMMAND JointPipelineWindowTest)

# C-ABI integration fixture covers context creation, explicit ownership,
# fail-closed validation and copied thread/context error lifetime.
add_executable(NativeProjectionCApiTest
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/tests/native_projection_c_api_test.cc")
target_link_libraries(NativeProjectionCApiTest PRIVATE joint_projection)
target_include_directories(NativeProjectionCApiTest PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/helper")
set_target_properties(NativeProjectionCApiTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME NativeProjectionCApiTest COMMAND NativeProjectionCApiTest)
