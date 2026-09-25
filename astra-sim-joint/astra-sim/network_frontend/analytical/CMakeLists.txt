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

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_aware)")

# Include src files to compile
file(GLOB srcs_common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
)

file(GLOB srcs_congestion_aware
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

# Compile Congestion Aware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_aware")
    # Path-1 removal (2026-08-18): the static aware target and
    # congestion_aware/main.cc were deleted with the offline static pipeline
    # (ruling A); the shared-source glob still excludes both entry files so
    # the online target declares its own main only.
    set(srcs_congestion_aware_shared ${srcs_congestion_aware} ${srcs_common})
    list(REMOVE_ITEM srcs_congestion_aware_shared
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main_online.cc)

    # Execution-driven mechanism sources (online target only). The root CMake
    # glob over astra-sim/workload/*.cc is non-recursive, so execution_driven/
    # is listed explicitly here; the static target must not contain any of them.
    file(GLOB srcs_execution_driven
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/*.cc)

    # Online entry point: shared sources + main_online.cc + execution_driven layer.
    add_executable(AstraSim_Analytical_Congestion_Aware_Online
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main_online.cc
            ${srcs_execution_driven})

    # Step-1-3 hook-count fixture (test-only): static entry replicated with a
    # counting CompletionObserver hook; never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_CompletionFixture
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/completion_observer_fixture_main.cc
            ${srcs_execution_driven})

    # Step-1-4 NodeStore / GraphSource fixture (test-only): standalone
    # NodeStore + NodeStoreGraphSource + ETFeederGraphSource adapter tests;
    # never part of the baseline gates.
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
    # Python round-trip over the v0 protocol (request/response/commit ack,
    # atomic publish, crash+timeout fail-closed); never part of the baseline
    # gates. Requires python3 on PATH and the echo script path as argv[1].
    add_executable(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/bridge_loopback_fixture.cc
            ${srcs_execution_driven})

    # Phase-5 GraphBatchCommitter fixture (test-only): two-phase atomic
    # commit (validate purity + commit semantics + zero side effects on any
    # validation failure, 方案 §8.1/§8.2); never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/graph_batch_committer_test.cc
            ${srcs_execution_driven})

    # C1 (2026-08-29) ParsedGraphBatch fixture (test-only): the one-shot
    # structural parse of one GraphBatch response into the typed batch --
    # the full malformed-protocol matrix (T/N/E/W/A/S/O rule families:
    # missing/unknown keys, integer overflow, negatives, array-type
    # violations, unicode passthrough, max-shape round trip, error
    # skeleton) plus the array-order preservation contract; never part of
    # the baseline gates.
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

    # Defect-A regression fixture (2026-08-16, synced from
    # face-defectfix2-done): deterministic unit-level reproduction of the
    # finished() counting vacuum (pump queues Submits that only the NEXT
    # drain registers) and proof that the drain-after-pump order closes it;
    # test-only, never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/service_vacuum_test.cc
            ${srcs_execution_driven})

    # N-way HBM contention fixture (test-only): end-to-end static-ET run of
    # the equal-split LocalHbmBandwidthModel over the synthetic 2-rank graph
    # (make_hbm_nway_fixture.py) -- 3-way/4-way numeric rate assertions,
    # comm/pool endpoint joins firing exactly once, hbm-charge=false
    # pass-through, and the contention-off legacy A/B; never part of the
    # baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_HbmNwayTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/hbm_nway_test.cc
            ${srcs_execution_driven})

    # SerDes片外链路并发化改造 remote-port fixtures (方案 阶段5.1/5.3/5.4,
    # test-only, never part of the production gates): port-model precision
    # fixture, ONLINE NodeView/NodeStore gate fixture, and STATIC/ETFeeder
    # gate fixture (paired with
    # ../../workload/execution_driven/tests/make_remote_port_static_fixture_et.py,
    # contract: python3 make_remote_port_static_fixture_et.py --out-dir
    # <临时目录> then --fixture-dir <同一临时目录>).
    add_executable(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/remote_port_nway_test.cc
            ${srcs_execution_driven})
    add_executable(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/remote_port_online_gate_test.cc
            ${srcs_execution_driven})
    add_executable(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/remote_port_static_gate_test.cc
            ${srcs_execution_driven})

    # PARTIAL 跨实例 copy 流水化 fixtures (2026-09-25, test-only, never part
    # of the production gates): the ONLINE pipeline-gate fixture (same-rank
    # COMP + KV-restore + 2 charged sends issued in ONE pass after the
    # comm/hbm_dma gates became UNLIMITED counted slots; two concurrent
    # restores sharing the LocalHbmBandwidthModel N-way pool) and the NoC
    # shared-link N-way equal-split fixture (three p2p flows, two sharing
    # link01, one solo control).
    add_executable(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/pipeline_concurrent_gate_test.cc
            ${srcs_execution_driven})
    add_executable(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/noc_shared_link_nway_test.cc
            ${srcs_execution_driven})

    # Link libraries
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CompletionFixture LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CompletionFixture LINK_PRIVATE Analytical_Congestion_Aware)
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HbmNwayTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HbmNwayTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest LINK_PRIVATE Analytical_Congestion_Aware)

    # Include directories
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_Online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CompletionFixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CompletionFixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CompletionFixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
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
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HbmNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HbmNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HbmNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

    # Properties
    # TODO: Switch to OFF after binary_function deprecation has been resolved
    set_target_properties(AstraSim_Analytical_Congestion_Aware_Online PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_Online
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CompletionFixture PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_CompletionFixture
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
    # sh_3.0 fix: the blueprint CMake lacked this block, so the reader-test
    # binary landed in AstraSim_Analytical/ instead of bin/ (port gap).
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
    set_target_properties(AstraSim_Analytical_Congestion_Aware_HbmNwayTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_HbmNwayTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest
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
    # release coexists with the erase, and sizeof(OnlineNode) is locked
    # (sh_3.0 anchor: 456 -- the variant MemAttrs member shifts the layout
    # relative to sh_1.0's 448; the mirror assert stays ABI-agnostic).
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_collector_one_shot_erase_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # Local-memory trace regression: byte-exact streamed JSON, same-tick
    # lifetime ordering, and a high-overlap timeline that must not retain one
    # live-tensor snapshot per timestamp.
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_mem_usage_tracker_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest LINK_PRIVATE AstraSim)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
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
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/alarm_cancellation_test.cc
            ${srcs_execution_driven})
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
endif ()
