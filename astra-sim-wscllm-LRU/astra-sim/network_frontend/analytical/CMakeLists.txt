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

# Testing must be enabled before any add_test() (F2/TEST-01/M8-03 fix,
# 2026-09-26): the fixture executables below were previously built but never
# registered, so ctest could not run them. Registration policy: only
# fixtures that run bare-repo (no materialized input, no extra argv) with a
# zero exit code are registered; every fixture below satisfies that policy
# since the static-ET fixtures were retired with the offline static path
# (2026-09-27).
enable_testing()

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_aware)")
# Fail closed on an illegal value: the guard below used to be a plain
# if()/endif() over the two legal names, so a typo'd BUILDTARGET configured
# successfully and produced zero targets -- the breakage only surfaced as a
# missing binary at build time with no diagnosis.
if(NOT BUILDTARGET STREQUAL "all" AND NOT BUILDTARGET STREQUAL "congestion_aware")
    message(FATAL_ERROR
            "BUILDTARGET='${BUILDTARGET}' is not a legal value; supported: all, congestion_aware")
endif()

# Include src files to compile. CONFIGURE_DEPENDS: without it a newly added
# .cc is invisible until the next manual re-configure (the glob result is
# cached and the build silently keeps using the stale file list).
file(GLOB srcs_common
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
)

file(GLOB srcs_congestion_aware
        CONFIGURE_DEPENDS
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
add_test(NAME AstraSim_Analytical_ChunkIdGeneratorTest
        COMMAND AstraSim_Analytical_ChunkIdGeneratorTest)
# F-05 (2026-09-27): bound every registered test; see the TIMEOUT note at
# the bottom of the congestion_aware block.
set_tests_properties(AstraSim_Analytical_ChunkIdGeneratorTest
        PROPERTIES TIMEOUT 300)

# M-23/M-24 regression (2026-09-30, 自 wscllm 母仓 P2-7/C3 同款移植): the
# shared CmdLineParser must honor the separated value form of its boolean
# option (--rendezvous-protocol false used to parse the implicit "true" and
# silently drop the value token) and validate the shared numeric options at
# the parse boundary. Same minimal recipe as ChunkIdGeneratorTest: only the
# parser source, no AstraSim/backend dependency.
add_executable(AstraSim_Analytical_CmdLineParserTest
        ${CMAKE_CURRENT_SOURCE_DIR}/common/CmdLineParser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/tests/cmdline_parser_test.cc)
target_include_directories(AstraSim_Analytical_CmdLineParserTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
set_target_properties(AstraSim_Analytical_CmdLineParserTest
        PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
add_test(NAME AstraSim_Analytical_CmdLineParserTest
        COMMAND AstraSim_Analytical_CmdLineParserTest)
set_tests_properties(AstraSim_Analytical_CmdLineParserTest
        PROPERTIES TIMEOUT 300)

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

    # Execution-driven mechanism sources (M5-01/F6 fix, 2026-09-26): the
    # root CMakeLists globs astra-sim/workload/execution_driven/*.cc into
    # the AstraSim static library, and every target below links AstraSim --
    # so the frontend targets resolve these objects from the archive and
    # must NOT recompile the glob locally. The previous per-target
    # recompilation (plus the stale comment claiming the root static target
    # "must not contain any of them") was an ODR/duplicate-symbol hazard and
    # doubled the compile cost of this layer. Test-only sources under
    # workload/execution_driven/tests/ are outside the root glob (it is
    # non-recursive) and stay listed explicitly per target.

    # Online entry point: shared sources + main_online.cc (execution-driven
    # layer via the AstraSim archive).
    add_executable(AstraSim_Analytical_Congestion_Aware_Online
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main_online.cc)

    # Step-1-4 NodeStore / GraphSource fixture (test-only): standalone
    # NodeStore + NodeStoreGraphSource tests; never part of the baseline
    # gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_NodeStoreTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/node_store_test.cc)

    # Step-1-5 watch/fence fixture (test-only): stage-completion watch
    # semantics (registration/no-ops, fire once, explicit status policy,
    # lifecycle/audit, online hook integration); never part of the baseline
    # gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/watch_registry_test.cc)

    # Step-1-6 DecisionMailbox fixture (test-only): decision-event
    # aggregation (order/drain/StateDelta, same-epoch dedup, counters);
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/decision_mailbox_test.cc)

    # Step-1-7 FileDecisionBridge loopback fixture (test-only): C++ <->
    # Python round-trip over the v0 protocol (request/response/commit ack,
    # atomic publish, crash+timeout fail-closed); never part of the baseline
    # gates. Requires python3 on PATH and the echo script path as argv[1].
    add_executable(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/bridge_loopback_fixture.cc)

    # Phase-5 GraphBatchCommitter fixture (test-only): two-phase atomic
    # commit (validate purity + commit semantics + zero side effects on any
    # validation failure, 方案 §8.1/§8.2); never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/graph_batch_committer_test.cc)

    # C1 (2026-08-29) ParsedGraphBatch parser fixture (test-only): the full
    # malformed-protocol matrix (T/N/E/W/A/S/O rule families) against the
    # single structural parse; never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/parsed_graph_batch_test.cc)

    # Phase-7 §10.4 WindowedTraceReader fixture (test-only): bounded-window
    # CSV reader (window topping / turn-0 vs turn>0 row semantics /
    # out-of-range rejection / late-arrival clamp counter / no-CSV path);
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/windowed_trace_reader_test.cc)

    # P0 turn-0 fix V2 oracle (2026-08-30): calendar reader vs the VERBATIM
    # pre-fix baseline reader (tests/LegacyOracleWindowedTraceReader.{hh,cc})
    # in its unbounded arm -- elementwise (alarm_tick, queue_index) fire
    # sequence equality on the same input; optional argv[1] runs a real
    # queue CSV (default: hermetic synthetic fixture). Test-only.
    add_executable(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/calendar_reader_oracle_test.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/LegacyOracleWindowedTraceReader.cc)

    # Defect-A regression fixture (2026-08-16): deterministic unit-level
    # reproduction of the finished() counting
    # vacuum (pump queues Submits that only the NEXT drain registers) and
    # proof that the drain-after-pump order closes it; test-only, never
    # part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/service_vacuum_test.cc)

    # hbm-bandwidth-contention unit fixture (test-only): LocalHbmBandwidthModel
    # N-user fair sharing / completion reallocation numerics, comm-join both
    # orders, per-job latency, flag-off fallback, single-COMP roofline
    # parity; standalone fake network/memory backends, no .et / no CSV
    # inputs; never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_hbm_bandwidth_model_test.cc)

    # sh_2.0 N-way HBM contention fixture (test-only): LocalHbmBandwidthModel
    # N-way equal-split numerics (3-way 1/3 rate, 2-way then full-rate
    # reallocation), comm/pool endpoint join completing exactly once in both
    # callback orders, hbm-charge=false creating no job, and NoC transit
    # charging nothing. Self-written tiny configs (no generated inputs);
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_hbm_model_test.cc)

    # Die backend (stage B, die内仿真器) hermetic fixtures: the backend
    # dispatch order (contract §7.3) on a real online Sys, the DieExecution
    # lifecycle (oracle-driven timer/memory spans, teardown), and the
    # LocalHbmBandwidthModel DIE_COMPUTE jobs competing in the same N-way
    # equal-split arbiter as COMPUTE/RESTORE/POOL_*/COMM_* (contract §7.4;
    # scenario 2 covers the die-vs-restore/pool dynamic contention the LRU
    # repository must keep). Self-written tiny configs (resolved die point
    # of contract §3.5); no generated inputs.
    add_executable(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/die_model_adapter_test.cc)
    add_executable(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/die_execution_lifecycle_test.cc)
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_hbm_die_memory_test.cc)

    # Link libraries
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE Analytical_Congestion_Aware)
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest LINK_PRIVATE Analytical_Congestion_Aware)

    # Include directories
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NodeStoreTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NodeStoreTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NodeStoreTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

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
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
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
    # BaseStream lifecycle regression: no retired stream-id tables and no
    # scheduler linkage retained across stream lifecycles.
    add_executable(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/system_history_lifecycle_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
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
    # (LocalMemTrackerTest removed with the ETFeeder-bound LocalMemUsageTracker
    # chain: every feeder and the Workload::report() consumer were dead code.)
    # Online shutdown-tail regression: a logical REQUEST_COMPLETE may precede
    # committed end-barrier terminals.  The fixture proves that shutdown drains
    # the main queue and deferred issue pass before the terminal audit exits.
    add_executable(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/terminal_tail_drain_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
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

    # Outer alarm cancellation regression (frozen plan §3.3/§6.2-1): the
    # end-to-end Sys timestamp bucket -> CommonNetworkApi registry ->
    # backend EventQueue chain.  Physically removes the outer alarm when the
    # last bucket event is cancelled, cascades only on an empty bucket,
    # proves repeat cancellation idempotent, and checks the legacy/fake
    # backend fallback plus both stale guards.  Test-only.
    add_executable(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/alarm_cancellation_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Remote-memory port-model exact fixture (SerDes 并发化方案 阶段 5.1/5.2,
    # 2026-09-24): self-contained /tmp configs in official template shape;
    # PER_NPU/PER_NODE/MEMORY_POOL mapping, same/staggered latency, 2-stream
    # long-short re-split, latency-ready at a stream-completion tick, same-tick
    # multi-port/multi-job delivery order, non-divisible single transaction,
    # tiny residue clamp, dual-zero (+1ns async), zero-byte positive latency,
    # huge/inf/invalid-bandwidth fail-closed children, per-segment service
    # invariants via a reference fluid model, and the plan's named math anchors
    # (300 / 90-120-140-150 / 1B-next-tick). Test-only; never part of the
    # baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/remote_port_nway_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Remote-memory ONLINE GATE system fixture (SerDes 并发化方案 阶段 5.3,
    # 2026-09-24): real online stack (Sys + Workload + NodeStoreGraphSource/
    # NodeView + AnalyticalRemoteMemory + congestion-aware network). One issue
    # pass on the production entry must put two same-rank dependency-free MEMs
    # on one port (backend PortStats issued/in-flight == 2; interval-integrated
    # peak_streaming >= 2, shared_busy_ns > 0) with the remote slot never
    # blocking (stage-2 gate), plus an independent MEM+COMM_SEND pair proving
    # MEM and comm doors no longer collide. Self-written /tmp configs in
    # official template shape; test-only, never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/remote_port_online_gate_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Remote-memory SENSING transactions positive fixture (SerDes 并发化方案
    # 阶段 5.1, round-2 smoke verification routing, 2026-09-24): with real
    # pool traffic, enable_transaction_log writes remote_memory_transactions
    # .jsonl rows that recompute shared_busy_ns/port_busy_ns/bytes_served
    # (row-interval sweep == backend PortStats), callback_tick == ceil(fluid
    # _finish_ns), completion distribution rows == delivery loop; armed +
    # zero transactions and unarmed + traffic both leave no file (lazy create
    # semantics the 2s smoke surface depends on). Test-only.
    add_executable(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/remote_port_sensing_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Orphan-test registration (face M12② same fix): the three tests below
    # previously had only a manual g++ line in their header comments (pointing
    # at the sister-repo root template/astra-sim-wscllm), so they sat outside
    # every build and their regressions escaped the tree. They join the build
    # with the same recipe as the fixtures above: shared sources, linked
    # against AstraSim (which carries the execution-driven layer) and the
    # Analytical_Congestion_Aware static library (which carries the backend
    # event-queue/fluid sources these tests exercise); never part of the
    # baseline gates.

    # Phase-1 step 1-2 online CLI contract test (R1-R14, incl. the FP1
    # hardened unsigned lexicon and watchdog bounds).
    add_executable(AstraSim_Analytical_Congestion_Aware_CliOnlineTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/cli_online_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CliOnlineTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CliOnlineTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CliOnlineTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CliOnlineTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CliOnlineTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
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
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/event_queue_deferred_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # Phase-1 step 1-2 IDLE lifecycle fixture (IDLE/ACTIVE/DRAINING/FINISHED
    # transitions, EOF/Error terminals, overflow audit, close-vs-submit race).
    # Uses std::thread (the only thread-using test in this directory), so it
    # additionally links pthread, matching the -pthread of its old manual
    # g++ line.
    add_executable(AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/ingress_idle_fixture.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_IngressIdleTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_IngressIdleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_IngressIdleTest LINK_PRIVATE pthread)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_IngressIdleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_IngressIdleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_IngressIdleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            PROPERTIES
            COMPILE_WARNING_AS_ERROR OFF
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )

    # ---- CTest registration (F2/TEST-01/M8-03 fix, 2026-09-26) ----
    # Registered: every fixture that runs bare-repo (self-written configs,
    # no materialized input, no extra argv) and exits 0. Each binary lands
    # in <build-tree>/bin/ via RUNTIME_OUTPUT_DIRECTORY above; the test
    # name equals the target name.
    add_test(NAME AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
            COMMAND AstraSim_Analytical_Congestion_Aware_WatchRegistryTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_NodeStoreTest
            COMMAND AstraSim_Analytical_Congestion_Aware_NodeStoreTest)
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
    add_test(NAME AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest
            COMMAND AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            COMMAND AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
            COMMAND AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            COMMAND AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest
            COMMAND AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            COMMAND AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest)
    # BridgeLoopback resolves its default echo-script path relative to the
    # repo root, so the test pins that working directory.
    add_test(NAME AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            COMMAND AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
            COMMAND AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
            COMMAND AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest
            COMMAND AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CliOnlineTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CliOnlineTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest
            COMMAND AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_IngressIdleTest)

    # Per-test TIMEOUTs (F-05 fix, 2026-09-27): ctest's 1500s default is a
    # real stall surface (a hung bridge poll used to wait the full default).
    # The million-node GraphBatchCommitter pressure test needs a larger
    # budget on slow/debug builds (600s); everything else is bounded at
    # 300s.
    set_tests_properties(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            PROPERTIES TIMEOUT 600)
    set_tests_properties(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
            AstraSim_Analytical_Congestion_Aware_NodeStoreTest
            AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
            AstraSim_Analytical_Congestion_Aware_ParsedBatchTest
            AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            AstraSim_Analytical_Congestion_Aware_CalendarOracleTest
            AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest
            AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest
            AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest
            AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
            AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
            AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest
            AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest
            AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
            AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
            AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
            AstraSim_Analytical_Congestion_Aware_RemotePortSensingTest
            AstraSim_Analytical_Congestion_Aware_CliOnlineTest
            AstraSim_Analytical_Congestion_Aware_EventQueueDeferredTest
            AstraSim_Analytical_Congestion_Aware_IngressIdleTest
            PROPERTIES TIMEOUT 300)
endif ()
