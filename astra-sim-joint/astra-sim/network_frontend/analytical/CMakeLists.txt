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


# Joint v7 event attribution: consume the cache option declared by the repo
# root CMakeLists (option JOINT_EVENT_PROFILE). This assembly-added
# directory does not inherit the root's directory scope, so the definition
# must be applied here too. Diagnostic-only; never changes behavior.
if(JOINT_EVENT_PROFILE)
    add_compile_definitions(JOINT_V7_EVENT_PROFILE)
endif()

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_aware)")
# Fail closed on an illegal value: a bare if() over the legal names used to
# configure successfully for any typo and produce zero targets -- the
# breakage only surfaced as a missing binary at build time with no
# diagnosis.
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
            CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/*.cc)
    list(REMOVE_ITEM srcs_execution_driven
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/NativeCoarseDieDomain.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/PipelineKernel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/NativeProjection.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/NativeProjectionCApi.cc)

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

    # W1 (joint work ledger, 2026-09-29) bridge fixture (test-only): the
    # state-derived diff (six lifecycle kinds, monotonic local_seq),
    # NodeStore::work_ledger_view, and the additive event_seqs[] wire
    # disclosure; never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/work_ledger_bridge_test.cc
            ${srcs_execution_driven})

    # Step-1-7 ShmDecisionBridge loopback fixture (test-only): C++ <->
    # Python round-trip over the v1 protocol (request/response/commit ack
    # shm frames + FIFO doorbells, crash+timeout fail-closed); never part
    # of the baseline gates. Requires python3 on PATH and the echo script
    # path as argv[1].
    add_executable(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/bridge_loopback_fixture.cc
            ${srcs_execution_driven})

    # v11 §8 保底: in-process ShmDecisionBridge loopback (test-only; no
    # python3 dependency): ShmFrameChannel byte round-trip + fail-closed
    # matrix (oversize/crc/stale generation), the forked-peer three-way
    # byte-identical exchange (incl. binary ledger splice), stats
    # accounting, and segment teardown. Registered into CTest.
    add_executable(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/shm_decision_bridge_loopback_test.cc
            ${srcs_execution_driven})

    # Phase-5 GraphBatchCommitter fixture (test-only): two-phase atomic
    # commit (validate purity + commit semantics + zero side effects on any
    # validation failure, 方案 §8.1/§8.2); never part of the baseline gates.
    add_executable(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/graph_batch_committer_test.cc
            ${srcs_execution_driven})

    # B3 ready-scan / issue-order differential fixture (test-only, exec doc
    # 10.2): hermetic NodeStore golden replay plus the full-stack issue-path
    # scenarios (real EventQueue/FluidScheduler/Sys).
    add_executable(JointReadyIssueOrderTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/ready_issue_order_test.cc
            ${srcs_execution_driven})

    # D1 window mutation fixture (test-only, exec doc 17.3/17.4, M13 +
    # M07/M10 interplay): value-level on_mutation/on_wakeup arithmetic plus
    # the real resource-version -> window-generation linkage (HBM rank via
    # DieExecution, NoC link via FluidScheduler, remote port via
    # AnalyticalRemoteMemory) on a hermetic Sys/EventQueue.
    add_executable(JointPipelineMutationTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/pipeline_mutation_test.cc
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

    # ------------------------------------------------------------------
    # Die backend (joint rollout stage 1, 2026-10-03): three hermetic
    # fixtures over a real Sys + EventQueue (fake network API + no-op
    # remote-memory API; no sh_test_mesh artifacts). Same recipe as the
    # other fixtures in this block: shared sources + the
    # execution_driven glob, linked against AstraSim (which carries
    # workload/DieExecution.cc + DieModelAdapter.cc and the vendored
    # die_model library) and the Analytical_Congestion_Aware static
    # library (which carries the backend EventQueue sources). Unlike the
    # historical fixtures these three also register into CTest (with the
    # root enable_testing added by the die rollout) so a plain ctest run
    # covers them; the explicit-binary regression convention for the
    # pre-existing fixtures is unchanged.
    # ------------------------------------------------------------------

    # Die backend dispatch: backend resolution + conflict matrix (die +
    # roofline/replay), die_spec -> adapter -> exactly one
    # finish_general_node, control_marker legacy micro-job, missing-spec
    # fail-closed abort (fork probe) + adapter throw, parser runtime_ns
    # conflict.
    add_executable(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/die_model_adapter_test.cc
            ${srcs_execution_driven})
    # DieExecution lifecycle: Timer/Memory mixed plan end-to-end, one-shot
    # local_mem_latency, contention=0 self-drive vs an independent
    # full-speed oracle, contention=1 arbitration, slot occupancy/release,
    # single terminal, teardown of an in-flight execution.
    add_executable(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/die_execution_lifecycle_test.cc
            ${srcs_execution_driven})
    # LocalHbmBandwidthModel DIE_COMPUTE jobs: equal-share service drain,
    # sink completion ticks, actual-vs-service separation, malformed-request
    # guards, cancel_die_jobs_of teardown (direct model fixture).
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_hbm_die_memory_test.cc
            ${srcs_execution_driven})
    # v7 E2 closed-loop component fixture: decode-burst die nodes over the
    # real arbitration, first equal-share boundary closed form, service
    # conservation, observer lifecycle, dispatch snapshot semantics, and the
    # canonical byte-diff summary used by the engine A/B differential.
    add_executable(AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/hbm_tail_component_test.cc
            ${srcs_execution_driven})
    # coarse_die_v2 workspace admission + shared Runtime seam: typed Plan
    # acquire/block/retry/terminal/cancel and a JOINT local-HBM endpoint
    # submitted concurrently to the same per-rank Runtime.
    add_executable(AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/coarse_workspace_runtime_test.cc
            ${srcs_execution_driven})
    target_compile_definitions(
        AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest PRIVATE
        COARSE_DIE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/die_model/coarse")

    # Link libraries
    # v6 compact round: execution_driven builds typed ledger records and the
    # shared compact codec lives in the joint_ledger static library.
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_Online LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CompletionFixture LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CompletionFixture LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NodeStoreTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NodeStoreTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WatchRegistryTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(JointReadyIssueOrderTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(JointReadyIssueOrderTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(JointPipelineMutationTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(JointPipelineMutationTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ParsedBatchTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_WindowedReaderTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CalendarOracleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HbmNwayTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HbmNwayTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

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
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(JointReadyIssueOrderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(JointReadyIssueOrderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(JointReadyIssueOrderTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(JointPipelineMutationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(JointPipelineMutationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(JointPipelineMutationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
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
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)

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
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest
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
    set_target_properties(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest
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
    set_target_properties(JointReadyIssueOrderTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(JointReadyIssueOrderTest
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
    )
    set_target_properties(JointPipelineMutationTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    set_target_properties(JointPipelineMutationTest
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_StatisticsCompactionTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # BaseStream/UsageTracker lifecycle regression: no retired stream-id
    # tables, exact ready-list scheduling gate, and bounded online history.
    add_executable(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/system_history_lifecycle_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_SystemHistoryLifecycleTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # MetricCollector code-7 regression: online raw memory anchors spool in
    # arrival order while transfer-complete keeps only action-bounded maxima.
    add_executable(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/metric_collector_anchor_spool_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest LINK_PRIVATE AstraSim joint_ledger)
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest LINK_PRIVATE Analytical_Congestion_Aware)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # Local-memory trace regression: byte-exact streamed JSON, same-tick
    # lifetime ordering, and a high-overlap timeline that must not retain one
    # live-tensor snapshot per timestamp.
    add_executable(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/local_mem_usage_tracker_test.cc)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest LINK_PRIVATE AstraSim joint_ledger)
    target_include_directories(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../extern/helper)
    set_target_properties(AstraSim_Analytical_Congestion_Aware_LocalMemTrackerTest PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
    # Online shutdown-tail regression: a logical REQUEST_COMPLETE may precede
    # committed end-barrier terminals.  The fixture proves that shutdown drains
    # the main queue and deferred issue pass before the terminal audit exits.
    add_executable(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest
            ${srcs_congestion_aware_shared}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workload/execution_driven/tests/terminal_tail_drain_test.cc
            ${srcs_execution_driven})
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_TerminalTailDrainTest LINK_PRIVATE AstraSim joint_ledger)
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest LINK_PRIVATE AstraSim joint_ledger)
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_CommunicatorGroupOrderTest LINK_PRIVATE Analytical_Congestion_Aware)
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
    target_link_libraries(AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest LINK_PRIVATE AstraSim joint_ledger)
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

    # ------------------------------------------------------------------
    # CTest registrations (die rollout stage 1, 2026-10-03): the three die
    # fixtures register here so a plain ctest from the assembled build tree
    # covers them (paired with the root-level enable_testing added by the
    # same rollout). The historical fixtures above keep their
    # explicit-binary regression convention (README section 6: ctest green
    # is not their execution evidence) and are deliberately NOT registered.
    # ------------------------------------------------------------------
    enable_testing()
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest
            COMMAND AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
            COMMAND AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest
            COMMAND AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest)
    add_test(NAME AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest
            COMMAND AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest)
    add_test(NAME JointReadyIssueOrderTest
            COMMAND JointReadyIssueOrderTest)
    add_test(NAME JointPipelineMutationTest
            COMMAND JointPipelineMutationTest)
    # v11 §8 保底: the in-process shm bridge loopback (no external deps).
    add_test(NAME ShmDecisionBridgeLoopbackTest
            COMMAND AstraSim_Analytical_Congestion_Aware_ShmDecisionBridgeLoopbackTest)
endif ()
