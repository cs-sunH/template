# CMake Requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Setup project
project(AstraSim_Analytical)

# network-F5 (2026-10-01): the Online entry point spawns its command-FIFO
# reader on a detached std::thread (congestion_aware/main_online.cc), so the
# thread library must be linked explicitly instead of riding on the
# transitive spdlog -> Threads::Threads chain through AstraSim.
find_package(Threads REQUIRED)

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_aware)")
# P3-3.9 E2 (2026-09-28): whitelist validation, fail-closed (same rationale as
# the backend CMakeLists; a wrong-case typo used to configure cleanly and
# silently produce no online/test targets at all).
set(_BUILDTARGET_VALUES "all" "congestion_aware")
if(NOT BUILDTARGET IN_LIST _BUILDTARGET_VALUES)
    message(FATAL_ERROR
            "BUILDTARGET='${BUILDTARGET}' is invalid; legal values: all, congestion_aware")
endif()

# Include src files to compile
# (CONFIGURE_DEPENDS, P3-3.9 E2 2026-09-28: new sources are picked up by an
# existing build tree instead of being silently dropped until reconfigure.)
file(GLOB srcs_common CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
)

file(GLOB srcs_congestion_aware CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/*.cc
)

# Chunk-id lifecycle fixture (test-only): exercises same-key concurrency,
# recv-first matching, and reclamation without the full simulator stack.
add_executable(AstraSim_Analytical_ChunkIdGeneratorTest
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ChunkIdGeneratorEntry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ChunkIdGenerator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/tests/chunk_id_generator_test.cc)
target_include_directories(AstraSim_Analytical_ChunkIdGeneratorTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/network_backend/analytical/include)
set_target_properties(AstraSim_Analytical_ChunkIdGeneratorTest
        PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
# Same output relocation as every fixture below (port-gap fix, 2026-09-27):
# without it the binary landed in the AstraSim_Analytical/ root instead of
# bin/.
set_target_properties(AstraSim_Analytical_ChunkIdGeneratorTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
)

# P2-7 regression (2026-09-28): the shared CmdLineParser must honor the
# separated value form of its boolean option (--rendezvous-protocol false
# used to parse the implicit "true" and silently drop the value token).
# Same minimal recipe as ChunkIdGeneratorTest: only the parser source, no
# AstraSim/backend dependency.
add_executable(AstraSim_Analytical_CmdLineParserTest
        ${CMAKE_CURRENT_SOURCE_DIR}/common/CmdLineParser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/tests/cmdline_parser_test.cc)
target_include_directories(AstraSim_Analytical_CmdLineParserTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
set_target_properties(AstraSim_Analytical_CmdLineParserTest
        PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
set_target_properties(AstraSim_Analytical_CmdLineParserTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
)

# tests-F2 (2026-10-01): register the two common fixtures with CTest. Both
# are no-arg mains and self-sufficient (temp paths, when any, are created by
# the fixtures themselves), so a plain no-arg add_test is enough; they sit
# outside the BUILDTARGET guard, mirroring their add_executable placement.
add_test(NAME AstraSim_Analytical_ChunkIdGeneratorTest
        COMMAND AstraSim_Analytical_ChunkIdGeneratorTest)
add_test(NAME AstraSim_Analytical_CmdLineParserTest
        COMMAND AstraSim_Analytical_CmdLineParserTest)

# Rank-local coarse workspace lease ledger fixture. This header-only test
# exercises checked admission, revision binding and RAII release without
# building the simulator/network stack.
add_executable(AstraSim_Analytical_WorkspaceAdmissionTest
        ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/workspace_admission_test.cc)
target_include_directories(AstraSim_Analytical_WorkspaceAdmissionTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
set_target_properties(AstraSim_Analytical_WorkspaceAdmissionTest
        PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
set_target_properties(AstraSim_Analytical_WorkspaceAdmissionTest
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
)
add_test(NAME AstraSim_Analytical_WorkspaceAdmissionTest
        COMMAND AstraSim_Analytical_WorkspaceAdmissionTest)

# Compile Congestion Aware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_aware")
    # Path-1 removal (2026-08-18): the static aware target and
    # congestion_aware/main.cc were deleted with the offline static pipeline
    # (ruling A); the shared-source glob still excludes the online entry
    # file so the online target declares its own main only.
    set(srcs_congestion_aware_shared ${srcs_congestion_aware} ${srcs_common})
    list(REMOVE_ITEM srcs_congestion_aware_shared
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main_online.cc)

    # Execution-driven mechanism sources. The root CMake glob over
    # astra-sim/workload/*.cc is non-recursive, so execution_driven/ is
    # listed explicitly here for the online target and the fixtures below
    # (the root CMakeLists.txt also globs the same sources into the AstraSim
    # static library). CONFIGURE_DEPENDS same as the globs above.
    file(GLOB srcs_execution_driven CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/*.cc)

    # Online entry point: shared sources + main_online.cc + execution_driven layer.
    add_executable(AstraSim_Analytical_Congestion_Aware_Online
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main_online.cc
            ${srcs_execution_driven})


    # Step-1-4 NodeStore / GraphSource fixture (test-only): standalone
    # NodeStore + NodeStoreGraphSource tests; never part of the baseline
    # gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_NodeStoreTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/node_store_test.cc
            ${srcs_execution_driven})

    # Step-1-5 watch/fence fixture (test-only): stage-completion watch
    # semantics (registration/no-ops, fire once, explicit status policy,
    # lifecycle/audit, online hook integration); never part of the baseline
    # gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/watch_registry_test.cc
            ${srcs_execution_driven})

    # Step-1-6 DecisionMailbox fixture (test-only): decision-event
    # aggregation (order/drain/StateDelta, same-epoch dedup, counters);
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/decision_mailbox_test.cc
            ${srcs_execution_driven})

    # Step-1-7 FileDecisionBridge loopback fixture (test-only): C++ <->
    # Python round-trip over the v1 protocol (request/response/commit ack,
    # atomic publish, crash+timeout fail-closed); never part of the baseline
    # gates. Requires python3 on PATH and the echo script path as argv[1].
    add_executable(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/bridge_loopback_fixture.cc
            ${srcs_execution_driven})

    # E1 (2026-09-28): register the loopback pair with CTest so the bridge
    # regression (and its Python peer online/verify/bridge_echo.py -- until
    # now an add_executable-only orphan that no automated entry exercised)
    # actually runs under tests/run_all.sh's full ctest sweep. The echo
    # script is passed as an absolute argv[1]; the fixture is
    # pid-namespaced under /tmp and python3-stdlib-only (official-image
    # compatible). TIMEOUT 600: scenario C burns the ~10s response-writer
    # startup grace by design.
    add_test(NAME AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            COMMAND AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../../sh_test_mesh/workload/llama2_7b_inference/online/verify/bridge_echo.py")
    set_tests_properties(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            PROPERTIES TIMEOUT 600)

    # Phase-5 GraphBatchCommitter fixture (test-only): two-phase atomic
    # commit (validate purity + commit semantics + zero side effects on any
    # validation failure, 方案 §8.1/§8.2); never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/graph_batch_committer_test.cc
            ${srcs_execution_driven})

    # C1 (2026-08-29) ParsedGraphBatch parser fixture (test-only): the full
    # malformed-protocol matrix (T/N/E/W/A/S/O rule families) against the
    # single structural parse; never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/parsed_graph_batch_test.cc
            ${srcs_execution_driven})

    # Phase-7 §10.4 WindowedTraceReader fixture (test-only): bounded-window
    # CSV reader (window topping / turn-0 vs turn>0 row semantics /
    # out-of-range rejection / late-arrival clamp counter / no-CSV path);
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/windowed_trace_reader_test.cc
            ${srcs_execution_driven})

    # P0 turn-0 fix V2 oracle (2026-08-30): calendar reader vs the VERBATIM
    # pre-fix baseline reader (tests/LegacyOracleWindowedTraceReader.{hh,cc})
    # in its unbounded arm -- elementwise (alarm_tick, queue_index) fire
    # sequence equality on the same input; optional argv[1] runs a real
    # queue CSV (default: hermetic synthetic fixture). Test-only.
    add_executable(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/calendar_reader_oracle_test.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/LegacyOracleWindowedTraceReader.cc
            ${srcs_execution_driven})

    # Defect-A regression fixture (2026-08-16, face主动测试错误分析.md):
    # deterministic unit-level reproduction of the finished() counting
    # vacuum (pump queues Submits that only the NEXT drain registers) and
    # proof that the drain-after-pump order closes it; test-only, never
    # part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/service_vacuum_test.cc
            ${srcs_execution_driven})

    # hbm-bandwidth-contention unit fixture (test-only): LocalHbmBandwidthModel
    # N-user fair sharing / completion reallocation numerics, comm-join both
    # orders, per-job latency, flag-off fallback, single-COMP roofline
    # parity; standalone fake network/memory backends, no .et / no CSV
    # inputs; never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc
            ${srcs_execution_driven})

    # Link libraries
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE Analytical_Congestion_Aware)
    # network-F5 (2026-10-01): explicit thread library for the detached
    # command-FIFO reader thread in congestion_aware/main_online.cc (see
    # find_package(Threads) at the top of this file).
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE Threads::Threads)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NodeStoreTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NodeStoreTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest LINK_PRIVATE Analytical_Congestion_Aware)

    # Include directories
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NodeStoreTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NodeStoreTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

    # Properties
    set_target_properties(AstraSim_Analytical_Congestion_Aware_Online PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_Online
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_NodeStoreTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_NodeStoreTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # sh_1.0 fix (same gap as sh_3.0): the blueprint CMake lacked this block,
    # so the reader-test binary landed in AstraSim_Analytical/ instead of bin/
    # (port gap; phase 5-6 parallel-reader validation runs it from bin/).
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # P0 V2 oracle (2026-08-30): same output relocation as the reader test.
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # Statistics compaction regression: zero-retirement live GPU failures and
    # microbenchmark history preservation. Standalone and test-only.
    add_executable(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/statistics_online_compaction_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # BaseStream/UsageTracker lifecycle regression: no retired stream-id
    # tables, exact ready-list scheduling gate, and bounded online history.
    add_executable(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/system_history_lifecycle_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # MEM dispatch fail-closed negative regression (remote-memory removal,
    # 2026-09-24): MEM_LOAD/MEM_STORE dispatch and the COMP remote-weight
    # roofline path must throw std::runtime_error.  Micro fixture: nodes are
    # built directly in the NodeStore, no .et / no CSV / no simulation;
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/mem_dispatch_throw_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # MetricCollector code-7 regression: online raw memory anchors spool in
    # arrival order while transfer-complete keeps only action-bounded maxima.
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_collector_anchor_spool_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # R2 (2026-08-29): MetricCollector one-shot bucket erase + OnlineNode
    # anchor fast-path regression.  Routing buckets drain per fired node,
    # the fire path stays byte-identical to the direct-apply baseline, A2
    # release coexists with the erase, and sizeof(OnlineNode) is locked.
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_collector_one_shot_erase_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # P2 batch A2 (2026-09-28) regressions, same single-source recipe as the
    # metric fixtures above (AstraSim static library, no backend events):
    #   MetricCapacityReplayTest        same-tick ledger-delta sampling;
    #   MetricManifestTopLevelTest      top-level manifest type guard;
    #   MetricLedgerDomainTest          ledger rank/sequence/negative-value
    #                                   domain guards (C2 2026-09-28);
    #   QueueLevelsBoundsTest           QueueLevels out-of-range guard;
    #   CustomOnlySystemConstructTest   retired custom keys fail closed at
    #                                   parse; native-only config constructs.
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_capacity_replay_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_manifest_top_level_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_ledger_domain_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    add_executable(AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/queue_levels_bounds_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    add_executable(AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/custom_only_system_construct_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)

    # ------------------------------------------------------------------
    # Die backend (wscllm rollout of the die-internal simulator): three
    # hermetic fixtures over a real Sys + EventQueue (fake network API; no
    # sh_test_mesh artifacts). Same recipe as the shared-source fixtures
    # above: shared sources + the execution_driven glob, linked against
    # AstraSim (which carries workload/DieExecution.cc + DieModelAdapter.cc
    # and the die_model library) and the Analytical_Congestion_Aware static
    # library (which carries the backend EventQueue sources).
    # ------------------------------------------------------------------

    # Die backend dispatch: backend resolution + the conflict matrix (die +
    # roofline), die_spec -> adapter -> exactly one finish_generic_node,
    # control_marker legacy micro-job, missing-spec fail-closed throw,
    # parser runtime_ns conflict.
    add_executable(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/die_model_adapter_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # DieExecution lifecycle: Timer/Memory mixed plan end-to-end, one-shot
    # local_mem_latency, contention=0 self-drive vs an independent
    # full-speed oracle, contention=1 arbitration, slot occupancy/release,
    # single terminal, teardown of an in-flight execution.
    add_executable(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/die_execution_lifecycle_test.cc
            ${srcs_execution_driven})

    # CoarseDieRuntime semantic compile-cache key (roofline batch P2-C):
    # identity-only task changes hit, one-field semantic changes miss, the
    # collision guard rejects crossed plans, and the cache observation
    # counters (schedule_next, key_ns) stay machine-readable.
    add_executable(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/coarse_semantic_key_cache_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # LocalHbmBandwidthModel DIE_COMPUTE jobs: equal-share service drain,
    # sink completion ticks, actual-vs-service separation, malformed-request
    # guards, cancel_die_jobs_of teardown (direct model fixture).
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_hbm_die_memory_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # The six single-source test targets above had no RUNTIME_OUTPUT_
    # DIRECTORY, so their binaries landed in AstraSim_Analytical/ instead of
    # bin/ (README §4 documents them as run directly from bin/). Relocate
    # them with the same output convention as the fixtures above.
    set_target_properties(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # P2 tail-block byte regression (2026-09-28): Sys-level collective
    # generation against a recording fake network; needs the shared sources
    # (Sys + ring algorithms) and the Analytical_Congestion_Aware library,
    # same recipe as TerminalTailDrainTest below.
    add_executable(AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/collective_tail_chunk_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # F1-CPP C1-1 (2026-09-29) HalvingDoubling phase-arithmetic 1-byte floor
    # regression: same recipe as CollectiveTailChunkTest (shared sources +
    # Analytical_Congestion_Aware library), 8-NPU ring, halvingDoubling keys.
    add_executable(AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/halving_doubling_phase_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    # Online shutdown-tail regression: a logical REQUEST_COMPLETE may precede
    # committed end-barrier terminals.  The fixture proves that shutdown drains
    # the main queue and deferred issue pass before the terminal audit exits.
    add_executable(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/terminal_tail_drain_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Ordered communicator identity regression: custom-algorithm ET suffixes
    # use the caller-provided rank order, not a sorted membership copy.
    add_executable(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/communicator_group_order_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Outer alarm cancellation regression (frozen plan §3.3/§6.2-1): the
    # end-to-end Sys timestamp bucket -> CommonNetworkApi registry ->
    # backend EventQueue chain.  Physically removes the outer alarm when the
    # last bucket event is cancelled, cascades only on an empty bucket,
    # proves repeat cancellation idempotent, and checks the legacy/fake
    # backend fallback plus both stale guards.  Test-only.
    add_executable(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/alarm_cancellation_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    # edtests-F1 (2026-09-25): keep asserts live in Release (see
    # CliOnlineTest). ed-graph-E3 fix (2026-09-27): the guard covers EVERY
    # non-Debug config -- RelWithDebInfo/MinSizeRel also inject -DNDEBUG, so
    # a Release-only -UNDEBUG left those builds vacuously ALL PASS.
    target_compile_options(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest PRIVATE $<$<NOT:$<CONFIG:Debug>>:-UNDEBUG>)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # M20 orphan-test registration: the three tests below previously had
    # only a manual g++ line in their header comments (this repo root).
    # They join the build with the same recipe as the fixtures above:
    # shared sources + execution_driven glob, linked against AstraSim and the
    # Analytical_Congestion_Aware static library (which carries the backend
    # event-queue/fluid sources these tests exercise); never part of the
    # baseline gates.

    # Phase-1 step 1-2 online CLI contract test (R1-R14, incl. the FP1
    # hardened unsigned lexicon and watchdog bounds).
    add_executable(AstraSim_Analytical_Congestion_Aware_CliOnlineTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/cli_online_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CliOnlineTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CliOnlineTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CliOnlineTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CliOnlineTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    # edtests-F1 (2026-09-25): the assert() calls ARE the R1-R14 contract
    # checks (103 of them, incl. the negative-parse cases -- with NDEBUG the
    # parser would not even run). Every non-Debug config (Release,
    # RelWithDebInfo, MinSizeRel) injects -DNDEBUG, which compiles every
    # check out and makes the binary print ALL PASS vacuously. Re-enable
    # assertions for this test target in all of them (ed-graph-E3 fix,
    # 2026-09-27: a Release-only guard left RelWithDebInfo/MinSizeRel
    # vacuous): -UNDEBUG lands after the config flags' -DNDEBUG and
    # undefines it. Same policy for the two sibling assert-based targets
    # above/below (AlarmCancellationTest, EventQueueDeferredTest).
    target_compile_options(AstraSim_Analytical_Congestion_Aware_CliOnlineTest PRIVATE $<$<NOT:$<CONFIG:Debug>>:-UNDEBUG>)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CliOnlineTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Phase-1 step 1-1 EventQueue tick-end closing + same-tick deferred
    # channel unit tests, incl. the FluidScheduler deferred-flush integration
    # (the scheduler comes from the linked Analytical_Congestion_Aware
    # library).
    add_executable(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/event_queue_deferred_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    # edtests-F1 (2026-09-25): keep asserts live in Release (see
    # CliOnlineTest). ed-graph-E3 fix (2026-09-27): non-Debug guard, as
    # there.
    target_compile_options(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest PRIVATE $<$<NOT:$<CONFIG:Debug>>:-UNDEBUG>)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Phase-1 step 1-2 IDLE lifecycle fixture (IDLE/ACTIVE/DRAINING/FINISHED
    # transitions, EOF/Error terminals, overflow audit, close-vs-submit race).
    # Uses std::thread (the only thread-using test fixture in this
    # directory; the production entry point congestion_aware/main_online.cc
    # spawns threads too and links Threads::Threads explicitly, see
    # network-F5 above), so it additionally links pthread, matching the
    # -pthread of its old manual g++ line.
    add_executable(AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/ingress_idle_fixture.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_IngressIdleTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_IngressIdleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_IngressIdleTest LINK_PRIVATE pthread)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_IngressIdleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_IngressIdleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    # edtests-F1 (2026-09-25): keep asserts live in Release (see
    # CliOnlineTest). ed-graph-E3 fix (2026-09-27): non-Debug guard, as
    # there.
    target_compile_options(AstraSim_Analytical_Congestion_Aware_IngressIdleTest PRIVATE $<$<NOT:$<CONFIG:Debug>>:-UNDEBUG>)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # tests-F2 (2026-10-01): register every remaining test binary with CTest.
    # Each target below was audited for self-sufficiency before registration:
    # no required argv (CalendarOracleTest's argv[1] is optional and its
    # default arm is hermetic; BridgeLoopbackTest -- the only fixture needing
    # an external file, the echo script -- keeps its dedicated registration
    # with an absolute argv[1] above), and every /tmp path is created by the
    # fixture itself as an mkdtemp/mkstemp template or a pid-suffixed name,
    # so plain no-arg add_test lines are enough. The assert-contract targets
    # (CliOnlineTest / EventQueueDeferredTest / IngressIdleTest /
    # AlarmCancellationTest) keep their -UNDEBUG compile options above.
    # tests/run_all.sh's full ctest sweep picks these up automatically.
    add_test(NAME AstraSim_Analytical_Congestion_Aware_NodeStoreTest
            COMMAND AstraSim_Analytical_Congestion_Aware_NodeStoreTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
            COMMAND AstraSim_Analytical_Congestion_Aware_WatchRegistryTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            COMMAND AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_ParsedBatchTest
            COMMAND AstraSim_Analytical_Congestion_Aware_ParsedBatchTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            COMMAND AstraSim_Analytical_Congestion_Aware_WindowedReaderTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CalendarOracleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CalendarOracleTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest
            COMMAND AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest
            COMMAND AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest
            COMMAND AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest
            COMMAND AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest
            COMMAND AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            COMMAND AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            COMMAND AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CliOnlineTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CliOnlineTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest
            COMMAND AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_IngressIdleTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest)

    add_test(NAME AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            COMMAND AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest)
endif ()
