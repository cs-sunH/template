# # *****************************************************************************
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# # *****************************************************************************

# die-internal cost model library (design doc §8.1; cross-stage contract §1).
#
# Stage A1 targets: die_model (STATIC) + the four unit test executables.
# Stage A2 added the mapping/cost-model sources and four more tests.
# Stage A3 adds src/Execution.cc to the static library, the die_model_cabi
# SHARED target (src/Capi.cc + include/DieCApi.h) and two more tests.

cmake_minimum_required(VERSION 3.22)
project(DieModel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only dependencies: the repository's vendored nlohmann/json under
# extern/helper (contract §1).  The path is resolved relative to this
# directory so the library builds both standalone and via add_subdirectory.
set(DIE_MODEL_HELPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../helper")
if(NOT EXISTS "${DIE_MODEL_HELPER_DIR}/json/json.hpp")
    message(FATAL_ERROR
        "nlohmann/json not found at ${DIE_MODEL_HELPER_DIR}/json/json.hpp")
endif()

add_library(die_model STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Hardware.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Workload.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Mapping.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/CostModel.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Tx8Reference.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Execution.cc")
target_include_directories(die_model PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${DIE_MODEL_HELPER_DIR}>)
target_compile_features(die_model PUBLIC cxx_std_17)
# The static archive is linked into the SHARED die_model_cabi target below,
# so its objects must be position independent.  This adds -fPIC only; no
# other compile options or symbol visibility changes are made.
set_target_properties(die_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

# C ABI shared library (stage A3; contract §1/§4).  Output directories are
# anchored at CMAKE_BINARY_DIR (the top of the build tree) so the .so lands
# next to the root project's own lib/bin outputs both standalone and via
# add_subdirectory.  Anchoring at ${CMAKE_CURRENT_BINARY_DIR}/.. would put it
# one directory too deep when this project is nested (e.g. the README §2
# assembled tree build_congestion_aware/astra_sim_build/extern/die_model),
# which would miss the Python discovery order (contract §7 ruling 1:
# build/astra_analytical/build_congestion_aware/{lib,bin}/libdie_model_cabi.so).
add_library(die_model_cabi SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Capi.cc")
# PUBLIC so the include directories of die_model (DieCApi.h, DieTypes.hh,
# nlohmann/json) reach the ABI consumers (capi_test here, ctypes via the .so).
target_link_libraries(die_model_cabi PUBLIC die_model)
set_target_properties(die_model_cabi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Tests: minimal self-contained checkers in the repository's fixture style
# (no gtest).  enable_testing() is called unconditionally: it is idempotent
# when the parent already enabled testing, and it is REQUIRED when nested --
# the astra-sim-face root calls enable_testing() only after its vendored
# add_subdirectory() blocks, so a guard like PROJECT_IS_TOP_LEVEL would leave
# this subdirectory without a generated CTestTestfile.cmake and its tests
# invisible to the parent's ctest.
enable_testing()

set(DIE_MODEL_TESTS
    tx8_reference_test
    hardware_test
    spec_parse_test
    configs_consistency_test
    workload_test
    mapping_test
    costmodel_test
    tx8_compile_test
    execution_test
    internal_slice_test
    internal_service_progress_test
    # v11 Phase 4 (K2 epoch-keyed timer-window memo, 方案 §6.2): twin
    # equivalence of memoized vs fresh query_internal_timer_window answers
    # across [query -> mutate -> query -> ...] drive scripts, including an
    # accepted pure-PE window and the preview-transaction invalidation.
    k2_timer_window_memo_test
    # v12 Phase 4 (K2 structural negative cache, 方案 §6.2): twin-oracle
    # equivalence of the process-level (plan key, approx config fingerprint)
    # keyed negative cache across plan families and full/2x/4x memory
    # drives, the P_struct fire-face Sound check, fingerprint isolation and
    # the master-switch / argument-validation guards.
    k2_plan_negative_cache_test
    selection_fuzz_test)
foreach(t IN LISTS DIE_MODEL_TESTS)
    add_executable(${t} "${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cc")
    target_link_libraries(${t} PRIVATE die_model)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# The C-ABI test links the shared library directly (the die_* symbols live
# only in die_model_cabi); BUILD_RPATH points at its output directory so the
# test runs without installation.
add_executable(capi_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/capi_test.cc")
target_link_libraries(capi_test PRIVATE die_model_cabi)
set_target_properties(capi_test PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
add_test(NAME capi_test COMMAND capi_test)

# The consistency test reads configs/ straight from the source tree
# (contract §1: JSON mirrors are checked against the code constants).
target_compile_definitions(configs_consistency_test PRIVATE
    DIE_MODEL_CONFIGS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/configs")
