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

# die-internal cost model library (die内仿真器设计文档 §8.1; stage A1 wiring;
# face-LRU rollout 2026-10-03).  Vendored byte-identical with the face base
# repository -- the five repos must run the same model version.  Builds the
# die_model static library plus its unit tests; the target is linked into
# AstraSim below so the runtime stages can adopt it incrementally.
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
# L24(2026-09-29 审查修复):GLOB 加 CONFIGURE_DEPENDS——增删源文件后不再
# 依赖手工重跑 cmake 才能被发现。
# 批次K核实修正(2026-09-29):原写法 `file(GLOB CONFIGURE_DEPENDS srcs ...)`
# 把 CONFIGURE_DEPENDS 放在了 <variable> 位——CMake 语义下结果存进名为
# "CONFIGURE_DEPENDS" 的变量,"srcs" 反被当成 glob 模式,srcs 恒空,
# add_library(AstraSim ...) 直接 "No SOURCES" 失败(任何含本仓的配置树
# 均无法生成)。按 file() 语法把关键字移到变量之后,修复且保留 L24 意图。
file(GLOB srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/execution_driven/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/logical_topology/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/native_collectives/collective_algorithm/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/astraccl/custom_collectives/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/common/*.cc"
    # L24(2026-09-29 审查修复):删除 third_party/utils/*.cc glob——该目录
    # 只有 protolib.py(+__pycache__),无任何 .cc,glob 恒空(连带:其
    # target_include_directories 行已按同批裁定删除,见下方 Include
    # Directories 处 infra-N 注)。
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
# infra-N(2026-09-30,L24 批次裁定的死代码):删除对 chakra/src/third_party/
# utils 的 include 行——该目录仅 .py(protolib.py),全仓零消费者,且随 L24
# 删除其 glob 后更无引用。
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

