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
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_unaware/congestion_aware)")

# Include src files to compile
file(GLOB srcs_common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
)

file(GLOB srcs_congestion_unaware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/*.cc
)

file(GLOB srcs_congestion_aware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/*.cc
)

# Compile Congestion Unaware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_unaware")
    add_executable(AstraSim_Analytical_Congestion_Unaware ${srcs_congestion_unaware} ${srcs_common})
    target_sources(AstraSim_Analytical_Congestion_Unaware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/congestion_unaware/main.cc)

    # Link libraries
    target_link_libraries(AstraSim_Analytical_Congestion_Unaware LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Unaware LINK_PRIVATE Analytical_Congestion_Unaware)

    # Include directories
    target_include_directories(AstraSim_Analytical_Congestion_Unaware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Unaware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Unaware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

    # Properties
    # TODO: Switch to OFF after binary_function deprecation has been resolved
    set_target_properties(AstraSim_Analytical_Congestion_Unaware PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Unaware
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
endif ()

# Compile Congestion Aware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_aware")
    # Shared implementation sources = GLOB result minus main files. Each target
    # declares its own main below; main_online.cc must never enter the static
    # target (two mains would collide at link time), and main.cc must never
    # enter the online target.
    set(srcs_congestion_aware_shared ${srcs_congestion_aware} ${srcs_common})
    list(REMOVE_ITEM srcs_congestion_aware_shared
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main_online.cc)

    # Execution-driven mechanism sources (online target only). The root CMake
    # glob over astra-sim/workload/*.cc is non-recursive, so execution_driven/
    # is listed explicitly here; the static target must not contain any of them.
    file(GLOB srcs_execution_driven
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/*.cc)

    add_executable(AstraSim_Analytical_Congestion_Aware
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/main.cc)

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

    # Phase-7 §10.4 WindowedTraceReader fixture (test-only): bounded-window
    # CSV reader (window topping / turn-0 vs turn>0 row semantics /
    # out-of-range rejection / late-arrival clamp counter / no-CSV path);
    # never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/windowed_trace_reader_test.cc
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

    # Link libraries
    target_link_libraries(AstraSim_Analytical_Congestion_Aware LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware LINK_PRIVATE Analytical_Congestion_Aware)
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest LINK_PRIVATE AstraSim)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest LINK_PRIVATE Analytical_Congestion_Aware)

    # Include directories
    target_include_directories(AstraSim_Analytical_Congestion_Aware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
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
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

    # Properties
    # TODO: Switch to OFF after binary_function deprecation has been resolved
    set_target_properties(AstraSim_Analytical_Congestion_Aware PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
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
    # sh_3.0 fix: the blueprint CMake lacked this block, so the reader-test
    # binary landed in AstraSim_Analytical/ instead of bin/ (port gap).
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
endif ()
