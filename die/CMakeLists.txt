cmake_minimum_required(VERSION 3.16)
project(CoarseDie VERSION 2.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED COARSE_DIE_JSON_INCLUDE)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../helper/json/json.hpp")
    set(COARSE_DIE_JSON_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../helper")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../astra-sim-wscllm/extern/helper/json/json.hpp")
    set(COARSE_DIE_JSON_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../astra-sim-wscllm/extern/helper")
  else()
    message(FATAL_ERROR "Set COARSE_DIE_JSON_INCLUDE to the directory containing json/json.hpp")
  endif()
endif()
add_library(coarse_die STATIC src/Hardware.cc src/Plan.cc src/Compiler.cc src/Mapping.cc src/Execution.cc)
target_include_directories(coarse_die PUBLIC include "${COARSE_DIE_JSON_INCLUDE}")
target_compile_features(coarse_die PUBLIC cxx_std_17)
set_target_properties(coarse_die PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(coarse_die PRIVATE -Wall -Wextra -Wpedantic)
add_library(coarse_die_cabi SHARED src/Capi.cc)
target_link_libraries(coarse_die_cabi PRIVATE coarse_die)
set_target_properties(coarse_die_cabi PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
option(COARSE_DIE_BUILD_TESTS "Build standalone coarse die verification" ON)
if(COARSE_DIE_BUILD_TESTS)
  enable_testing()
  foreach(t runtime_subunit_credit_test runtime_snapshot_split_owner_test repeated_region_availability_test compiler_summary_test norm_row_accounting_test prediction_availability_test region_delay_math_test region_overhead_test elementwise_service_test packing_allocator_test ratio_allocator_test os_service_curve_test os_service_scaling_test service_curve_schema_test service_curve_runtime_test capi_cost_query_test source_golden_test hardware_input_test compile_execution_test compiler_credit_test gemm_os_access_counter_test gemm_os_geometry_test gemm_os_route_counter_test gemm_os_resource_facts_test gemm_os_readiness_test operator_accounting_test gemm_accounting_test gemm_tail_budget_test task_schema_test compact_batch_semantics_test equivalent_work_test fractional_work_test fused_ffn_accounting_test attention_accounting_test endpoint_allocation_test runtime_credit_scale_test runtime_clock_translation_test runtime_counters_test estimate_driver_test runtime_semantics_test oracle_granularity_test capi_lifecycle_test capi_cache_response_test workspace_metadata_test workspace_envelope_test runtime_submit_transaction_test runtime_owner_symmetry_test)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cc")
      add_executable(coarse_${t} tests/${t}.cc)
      target_link_libraries(coarse_${t} PRIVATE coarse_die coarse_die_cabi)
      target_compile_definitions(coarse_${t} PRIVATE
        COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        COARSE_DIE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
      add_test(NAME coarse_${t} COMMAND coarse_${t})
    endif()
  endforeach()
  # Synthetic numerical controls write their full observations only in the
  # build directory. Their independent tiny-time/credit oracles remain in C++.
  foreach(t runtime_positive_time_test runtime_credit_representability_test runtime_failure_snapshot_test runtime_tiny_ready_test runtime_subepsilon_event_test)
    add_executable(coarse_${t} tests/${t}.cc)
    target_link_libraries(coarse_${t} PRIVATE coarse_die)
    target_compile_definitions(coarse_${t} PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      COARSE_DIE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
    add_test(NAME coarse_${t}
      COMMAND coarse_${t} "${CMAKE_CURRENT_BINARY_DIR}/numerical/${t}" ctest)
  endforeach()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_core_source.py")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME coarse_core_docx_source_test
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_core_source.py")
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/estimate_cli_test.py")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME coarse_estimate_cli_test
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/estimate_cli_test.py"
        --library "$<TARGET_FILE:coarse_die_cabi>" -v)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/grouped_gemm_oracle_test.py")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME coarse_grouped_gemm_oracle_test
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/grouped_gemm_oracle_test.py")
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/analytical_service_oracle_test.py")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME coarse_analytical_service_oracle_test
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/analytical_service_oracle_test.py")
  endif()
  # Independent bounded Fraction LP regression against public Runtime samples.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/owner_lp_oracle_driver.cc" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/validation/owner_lp_oracle.py")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(coarse_owner_lp_oracle_driver
      tests/owner_lp_oracle_driver.cc)
    target_include_directories(coarse_owner_lp_oracle_driver PRIVATE tests)
    target_link_libraries(coarse_owner_lp_oracle_driver PRIVATE coarse_die)
    target_compile_definitions(coarse_owner_lp_oracle_driver PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      COARSE_DIE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
    add_test(NAME coarse_owner_lp_oracle_test
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/validation/owner_lp_oracle.py"
        "$<TARGET_FILE:coarse_owner_lp_oracle_driver>")
  endif()
  # A measurement executable, not a passing granularity assertion. The
  # independent Python oracle applies the frozen error limits to its output.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/elementwise_oracle_driver.cc")
    add_executable(coarse_elementwise_oracle_driver tests/elementwise_oracle_driver.cc)
    target_link_libraries(coarse_elementwise_oracle_driver PRIVATE coarse_die)
    target_compile_definitions(coarse_elementwise_oracle_driver PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/compiled_gemm_oracle_driver.cc")
    add_executable(coarse_compiled_gemm_oracle_driver tests/compiled_gemm_oracle_driver.cc)
    target_link_libraries(coarse_compiled_gemm_oracle_driver PRIVATE coarse_die)
    target_compile_definitions(coarse_compiled_gemm_oracle_driver PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/k_readiness_runtime_driver.cc")
    add_executable(coarse_k_readiness_runtime_driver
      tests/k_readiness_runtime_driver.cc)
    target_link_libraries(coarse_k_readiness_runtime_driver PRIVATE coarse_die)
    target_compile_definitions(coarse_k_readiness_runtime_driver PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME coarse_k_readiness_runtime_self_test
      COMMAND coarse_k_readiness_runtime_driver --self-test)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/norm_oracle_driver.cc")
    add_executable(coarse_norm_oracle_driver tests/norm_oracle_driver.cc)
    target_link_libraries(coarse_norm_oracle_driver PRIVATE coarse_die)
    target_compile_definitions(coarse_norm_oracle_driver PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/service_curve_benchmark.cc")
    add_executable(coarse_service_curve_benchmark tests/service_curve_benchmark.cc)
    target_link_libraries(coarse_service_curve_benchmark PRIVATE coarse_die coarse_die_cabi)
    target_compile_definitions(coarse_service_curve_benchmark PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      COARSE_DIE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.cc")
    add_executable(coarse_die_benchmark tests/benchmark.cc)
    target_link_libraries(coarse_die_benchmark PRIVATE coarse_die)
    target_compile_definitions(coarse_die_benchmark PRIVATE
      COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
      COARSE_DIE_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
  endif()
endif()
