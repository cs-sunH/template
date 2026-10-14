# # *****************************************************************************
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# # *****************************************************************************

# Native joint dual-ledger receiver (joint加速改造 v4, 2026-10-12).
#
# joint_ledger (STATIC): typed readers, descriptor port, WorkLedger core,
# envelope reader/stage core -- the semantic 1:1 port of the Python
# work_ledger/pipeline_ledger receive path.
# joint_ledger_cabi (SHARED): the C ABI consumed by the Python facade
# (joint/native_ledger.py through ctypes).  Conventions mirror extern/
# die_model: outputs anchored at CMAKE_BINARY_DIR/{lib,bin}, tests register
# with CTest unconditionally.

cmake_minimum_required(VERSION 3.22)
project(JointLedger CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JOINT_LEDGER_HELPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../helper")
if(NOT EXISTS "${JOINT_LEDGER_HELPER_DIR}/json/json.hpp")
    message(FATAL_ERROR
        "nlohmann/json not found at ${JOINT_LEDGER_HELPER_DIR}/json/json.hpp")
endif()

add_library(joint_ledger STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/LedgerTypes.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/TypedRows.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/CompactWire.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/CompactReader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/JsonScanner.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PipelineDescriptor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/WorkLedgerCore.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/EnvelopeReader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/RowParsers.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PipelineStage.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/DualLedger.cc")
target_include_directories(joint_ledger PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${JOINT_LEDGER_HELPER_DIR}>)
target_compile_features(joint_ledger PUBLIC cxx_std_17)
set_target_properties(joint_ledger PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(joint_ledger_cabi SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Capi.cc")
target_link_libraries(joint_ledger_cabi PUBLIC joint_ledger)
set_target_properties(joint_ledger_cabi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

enable_testing()

set(JOINT_LEDGER_TESTS
    json_scanner_test
    ledger_types_test
    pipeline_descriptor_port_test
    work_ledger_core_test
    envelope_reader_test
    dual_ledger_test
    compact_wire_test)
foreach(t IN LISTS JOINT_LEDGER_TESTS)
    add_executable(${t} "${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cc")
    target_link_libraries(${t} PRIVATE joint_ledger)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

add_executable(joint_ledger_capi_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/joint_ledger_capi_test.cc")
target_link_libraries(joint_ledger_capi_test PRIVATE joint_ledger_cabi)
set_target_properties(joint_ledger_capi_test PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
add_test(NAME joint_ledger_capi_test COMMAND joint_ledger_capi_test)
