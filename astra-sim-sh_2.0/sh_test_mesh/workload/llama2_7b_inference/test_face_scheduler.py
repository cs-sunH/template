#!/usr/bin/env python3
"""Focused tests for the trace-generation-time FACE scheduler."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    FaceRooflineEstimate,
    FaceRequest,
    InstanceTaskLoadSnapshot,
    KVAllocator,
    KVCacheManager,
    KVTransfer,
    KVTransferShard,
    PREFILL_CHUNK_SIZE,
    WeightedInstanceGraph,
    attention_heads_by_tp_rank,
    build_instances,
    deterministic_xy_route,
    estimate_decode_remaining_task_load_ns,
    estimate_iteration_time_ns,
    estimate_model_weight_bytes,
    estimate_prefill_task_load_ns,
    kv_cache_bytes_for_tokens,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
    manhattan_hops,
    model_weight_shard_bytes_by_tp_rank,
    nearest_edge_rank,
    physical_edge_ranks,
    select_decode_instance,
    select_prefill_instance,
)
from generate_face_trace import (  # noqa: E402
    PendingHistoryGate,
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_kv_transfer,
    _emit_tp_point_to_point_readiness_barrier,
    _emit_tp_readiness_barrier,
    derive_prefill_work_tokens,
    load_face_trace_config,
    reconcile_pending_history_location,
    select_first_session_requests,
)
from generate_trace import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    REMOTE_WEIGHT_ATTR,
    RequestSpec,
    TraceBuilder,
    load_remote_memory_config,
    shard_extent,
    transformer_pass,
    transformer_pass_aggregated,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from metrics_integration import (  # noqa: E402
    PlannerRooflineStatsAccumulator,
    write_planner_roofline_stats,
)


def line_topology() -> tuple[FaceHardware, object]:
    hardware = FaceHardware(
        mesh_rows=3,
        mesh_cols=2,
        local_hbm_capacity_bytes=100,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
            FaceInstanceSpec("ins2", "3", (4, 5)),
        ),
    )
    return hardware, topology


class FaceSchedulerTests(unittest.TestCase):
    # request-neutral（裸仓库还原，2026-08-16）：物化输入删除后这两个
    # config 依赖用例跳过——按 traces/materialize_first_30s.py 物化输入后自动恢复。
    _MATERIALIZED = (
        MODULE_DIR / "traces" /
        "astra_compute_20_first_30_seconds_request_queue.csv"
    ).is_file()

    @staticmethod
    def _request(
        session_id: str,
        turn_index: int,
        request_id: str,
    ) -> RequestSpec:
        return RequestSpec(
            session_id=session_id,
            turn_index=turn_index,
            request_id=request_id,
            prefill_length=10 + turn_index,
            decode_length=2 + turn_index,
            session_arrival_time_ns=0 if turn_index == 0 else None,
            inter_request_interval_ns=None if turn_index == 0 else 1000,
        )

    @staticmethod
    def _tiny_kv_manager(
        *,
        capacity_bytes: int = 200,
        reserve_context_tokens: int = 40,
        layers: int = 1,
    ) -> tuple[FaceHardware, FaceModel, object, KVCacheManager]:
        hardware = FaceHardware(
            mesh_rows=2,
            mesh_cols=2,
            local_hbm_capacity_bytes=capacity_bytes,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        model = FaceModel(
            layers=layers,
            hidden_size=2,
            ffn_size=2,
            num_heads=2,
            vocab_size=2,
            bytes_per_elem=1,
            mlp_variant="gelu",
        )
        topology = build_instances(
            hardware,
            (
                FaceInstanceSpec("ins0", "1", (0, 1)),
                FaceInstanceSpec("ins1", "2", (2, 3)),
            ),
        )
        return (
            hardware,
            model,
            topology,
            KVCacheManager(
                topology,
                model,
                reserve_context_tokens=reserve_context_tokens,
            ),
        )

    def test_trace_builder_can_stream_without_retaining_nodes(self) -> None:
        streamed = []
        builder = TraceBuilder(
            remote_operand_loads=False,
            node_sink=streamed.append,
            retain_nodes=False,
        )
        builder.comp("streamed_comp", 123, 456)
        builder.all_reduce("streamed_collective", 789, "1")
        self.assertEqual(builder.node_count, 2)
        self.assertEqual(builder.nodes, [])
        self.assertEqual([node.id for node in streamed], [0, 1])
        self.assertEqual(streamed[1].data_deps, [0])
        self.assertTrue(
            any(
                attr.name == "comm_size" and attr.uint64_val == 789
                for attr in streamed[1].attr
            )
        )

    @staticmethod
    def _seed_local_session(
        manager: KVCacheManager,
        *,
        session_id: str,
        instance_index: int,
        context_tokens: int,
        completion_ns: int | None,
        next_request_type: str | None = None,
    ) -> None:
        before, prefix_transfer, transfer, evictions = manager.prepare_prefill(
            session_id=session_id,
            target_instance_index=instance_index,
            history_tokens=0,
            trigger_request_id=f"{session_id}_initial",
        )
        if before is not None or prefix_transfer is not None or transfer is not None or evictions:
            raise AssertionError("new session unexpectedly required history movement")
        growth_evictions = manager.expand_prefill(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=context_tokens,
            trigger_request_id=f"{session_id}_initial",
        )
        if growth_evictions:
            raise AssertionError("test fixture unexpectedly evicted a session")
        if completion_ns is not None:
            manager.mark_complete(
                session_id, completion_ns, next_request_type=next_request_type
            )

    def test_first_n_session_selection_keeps_all_source_rows(self) -> None:
        requests = (
            self._request("s0", 0, "s0r0"),
            self._request("s1", 0, "s1r0"),
            self._request("s0", 1, "s0r1"),
            self._request("s2", 0, "s2r0"),
            self._request("s1", 1, "s1r1"),
        )
        selected, session_ids = select_first_session_requests(requests, 2)
        self.assertEqual(session_ids, ("s0", "s1"))
        self.assertEqual(
            tuple(request.request_id for request in selected),
            ("s0r0", "s1r0", "s0r1", "s1r1"),
        )
        all_requests, all_sessions = select_first_session_requests(requests, 0)
        self.assertEqual(all_requests, requests)
        self.assertEqual(all_sessions, ("s0", "s1", "s2"))
        with self.assertRaises(ValueError):
            select_first_session_requests(requests, 4)
        with self.assertRaises(ValueError):
            select_first_session_requests(requests, -1)

    def test_zero_context_truncation_reconciles_partial_history_gate(self) -> None:
        gate = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=(None, None),
            location="partial_hbm_remote",
        )
        request_plan = SimpleNamespace(
            request_id="request",
            history_tokens_discarded=10,
            history_tokens_before=0,
            history_location_before=SimpleNamespace(
                location="local_hbm",
                total_bytes=0,
            ),
        )
        reconcile_pending_history_location(gate, request_plan)
        self.assertEqual(gate.location, "local_hbm")

    def test_remote_memory_mesh_shape_rejects_malformed_legacy_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "remote_memory.json"
            source.write_text(
                json.dumps(
                    {
                        "memory-type": "PER_NPU_MEMORY_EXPANSION",
                        "npu-ids": [0],
                        "mesh-shape": 4,
                        "remote-mem-latency": 0,
                        "remote-mem-bw": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mesh-shape"):
                load_remote_memory_config(source, 4, mesh_shape=(2, 2))

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_checked_in_astra_compute_selection_uses_three_minute_window(
        self,
    ) -> None:
        config = load_face_trace_config()
        self.assertEqual(config.model_name, "llama2_7b")
        self.assertEqual(config.mlp_variant, "swiglu")
        self.assertEqual(config.npus_count, 54)
        self.assertEqual((config.hardware.mesh_rows, config.hardware.mesh_cols), (9, 6))
        self.assertEqual(config.hardware.d2d_latency_ns, 5)
        self.assertEqual(config.hardware.local_hbm_latency_ns, 100)
        self.assertEqual(len(config.inference_groups), 9)
        self.assertEqual({len(group.ranks) for group in config.inference_groups}, {6})
        self.assertEqual(
            (config.layers, config.hidden_size, config.ffn_size, config.num_heads, config.vocab_size),
            (32, 4096, 11008, 32, 32000),
        )
        # Execution-Driven 改造 步骤 0-1/0-6: simulation input is the
        # materialized 20.csv first-30-seconds window (user directive
        # 2026-08-15); expectations updated to the materialized values.
        expected_request_queue = (
            MODULE_DIR
            / "traces"
            / "astra_compute_20_first_30_seconds_request_queue.csv"
        ).resolve()
        self.assertEqual(config.request_queue_csv, expected_request_queue)
        self.assertEqual(config.request_queue_session_limit, 0)
        self.assertEqual(config.trace_granularity, "request_aggregated")
        self.assertEqual(config.source_request_count, 1177)
        self.assertEqual(config.source_session_count, 112)
        self.assertEqual(len(config.request_queue), 1177)
        self.assertEqual(len(config.selected_session_ids), 112)
        self.assertEqual(
            config.selected_session_ids[:5],
            ("session_0", "session_1", "session_2", "session_3", "session_4"),
        )
        self.assertEqual(
            config.selected_session_ids[-5:],
            ("session_107", "session_108", "session_109", "session_110", "session_111"),
        )
        self.assertEqual(
            {request.session_id for request in config.request_queue},
            {f"session_{index}" for index in range(112)},
        )
        first_request_indexes: dict[str, int] = {}
        for index, request in enumerate(config.request_queue):
            first_request_indexes.setdefault(request.session_id, index)
        self.assertEqual(len(first_request_indexes), 112)
        for index in first_request_indexes.values():
            request = config.request_queue[index]
            self.assertEqual(request.turn_index, 0)
            self.assertIsNotNone(request.session_arrival_time_ns)
        prefill_work_tokens = derive_prefill_work_tokens(config.request_queue)
        for index in first_request_indexes.values():
            self.assertEqual(
                prefill_work_tokens[index],
                config.request_queue[index].prefill_length,
            )
        prefill_lengths = [request.prefill_length for request in config.request_queue]
        decode_lengths = [request.decode_length for request in config.request_queue]
        arrival_times = [
            request.session_arrival_time_ns
            for request in config.request_queue
            if request.session_arrival_time_ns is not None
        ]
        self.assertEqual(PREFILL_CHUNK_SIZE, 512)
        self.assertAlmostEqual(
            config.source_average_decode_length,
            459.4944774851317,
        )
        self.assertEqual((min(prefill_lengths), max(prefill_lengths)), (3, 53924))
        self.assertEqual((min(decode_lengths), max(decode_lengths)), (1, 13812))
        self.assertEqual(len(arrival_times), 112)
        self.assertEqual(
            (min(arrival_times), max(arrival_times)),
            (94835000, 25959142000),
        )
        request_arrivals: dict[str, int] = {}
        for request in config.request_queue:
            if request.turn_index == 0:
                self.assertIsNotNone(request.session_arrival_time_ns)
                request_arrival = request.session_arrival_time_ns
            else:
                self.assertIn(request.session_id, request_arrivals)
                self.assertIsNotNone(request.inter_request_interval_ns)
                request_arrival = (
                    request_arrivals[request.session_id]
                    + request.inter_request_interval_ns
                )
            self.assertIsNotNone(request_arrival)
            request_arrivals[request.session_id] = request_arrival
        self.assertEqual(max(request_arrivals.values()), 29988879000)
        self.assertLessEqual(max(request_arrivals.values()), 30000000000)

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_llama2_7b_tp6_partition_is_exact_without_model_padding(self) -> None:
        config = load_face_trace_config()
        attention_heads = tuple(
            shard_extent(config.num_heads, 6, index) for index in range(6)
        )
        self.assertEqual(attention_heads, (6, 6, 5, 5, 5, 5))
        self.assertEqual(
            tuple(
                shard_extent(config.ffn_size, 6, index)
                for index in range(6)
            ),
            (1835, 1835, 1835, 1835, 1834, 1834),
        )
        self.assertEqual(
            tuple(
                shard_extent(config.vocab_size, 6, index)
                for index in range(6)
            ),
            (5334, 5334, 5333, 5333, 5333, 5333),
        )
        self.assertEqual(
            estimate_model_weight_bytes(config.model), 13_476_831_232
        )
        weight_shards = model_weight_shard_bytes_by_tp_rank(config.model, 6)
        self.assertEqual(
            weight_shards,
            (
                2_335_890_091,
                2_335_890_091,
                2_201_655_979,
                2_201_655_979,
                2_200_869_546,
                2_200_869_546,
            ),
        )
        self.assertEqual(
            sum(weight_shards),
            estimate_model_weight_bytes(config.model),
        )

        topology = build_instances(
            config.hardware,
            tuple(
                FaceInstanceSpec(group.name, group.pg_name, group.ranks)
                for group in config.inference_groups
            ),
        )
        manager = KVCacheManager(
            topology,
            config.model,
            edge_ranks=config.remote_memory.edge_npus,
            reserve_context_tokens=config.kv_reserve_context_tokens,
        )
        for instance in topology.instances:
            self.assertEqual(
                tuple(
                    snapshot.model_weight_bytes
                    for snapshot in manager.hbm_snapshots(instance.index)
                ),
                weight_shards,
            )

    def test_llama2_7b_tp6_whole_head_kv_shards_are_exact(self) -> None:
        model = FaceModel(
            layers=32,
            hidden_size=4096,
            ffn_size=11008,
            num_heads=32,
            vocab_size=32000,
            bytes_per_elem=2,
            mlp_variant="swiglu",
        )
        heads = attention_heads_by_tp_rank(model.num_heads, 6)
        self.assertEqual(heads, (6, 6, 5, 5, 5, 5))

        tokens = 1_000_000
        bytes_per_head = (
            2
            * model.layers
            * tokens
            * (model.hidden_size // model.num_heads)
            * model.bytes_per_elem
        )
        shards = kv_cache_shard_bytes_for_tokens(model, tokens, 6)
        self.assertEqual(
            shards,
            tuple(head_count * bytes_per_head for head_count in heads),
        )
        self.assertEqual(sum(shards), kv_cache_bytes_for_tokens(model, tokens))

    def test_per_rank_hbm_accounting_invariants_include_weights_and_kv(self) -> None:
        _, _, _, manager = self._tiny_kv_manager()
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=10,
            completion_ns=None,
        )

        snapshots = manager.hbm_snapshots()
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in snapshots),
            (20, 20, 0, 0),
        )
        for snapshot in snapshots:
            self.assertEqual(
                snapshot.used_bytes,
                snapshot.model_weight_bytes + snapshot.kv_cache_bytes,
            )
            self.assertEqual(
                snapshot.remaining_bytes,
                snapshot.capacity_bytes - snapshot.used_bytes,
            )

        session = manager.session_snapshot("session")
        self.assertEqual(session.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(session.instance_index, 0)
        self.assertEqual(session.rank_bytes, ((0, 20), (1, 20)))
        self.assertEqual(sum(bytes_count for _, bytes_count in session.rank_bytes), 40)

    def test_reserve_threshold_and_fifo_can_evict_multiple_sessions(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=40,
        )
        for session_id, completion_ns in (
            ("session_b", 10),
            ("session_a", 10),
            ("session_c", 20),
            ("session_d", 30),
            ("session_e", 40),
        ):
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=10,
                completion_ns=completion_ns,
            )

        self.assertEqual(
            tuple(snapshot.remaining_bytes for snapshot in manager.hbm_snapshots(0)),
            manager.reserve_bytes_by_tp_rank,
        )
        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="equal_threshold",
        )
        self.assertEqual(evictions, ())
        self.assertEqual(reserve_unmet, ())

        self._seed_local_session(
            manager,
            session_id="active_trigger",
            instance_index=0,
            context_tokens=25,
            completion_ns=None,
        )
        self.assertTrue(
            all(
                snapshot.remaining_bytes < reserve
                for snapshot, reserve in zip(
                    manager.hbm_snapshots(0),
                    manager.reserve_bytes_by_tp_rank,
                )
            )
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="below_threshold",
        )
        self.assertEqual(
            tuple(transfer.session_id for transfer in evictions),
            ("session_a", "session_b", "session_c"),
        )
        self.assertEqual(reserve_unmet, ())
        for transfer in evictions:
            self.assertEqual(transfer.kind, "remote_store")
            self.assertEqual(
                sum(shard.bytes for shard in transfer.shards),
                transfer.total_bytes,
            )
            self.assertTrue(
                all(shard.edge_rank in manager.edge_ranks for shard in transfer.shards)
            )
        for session_id in ("session_a", "session_b", "session_c"):
            self.assertEqual(
                manager.session_snapshot(session_id).location,
                KVCacheManager.REMOTE_MEMORY,
            )

    def test_unreachable_reserve_returns_reserve_unmet_without_looping(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=100,
            reserve_context_tokens=50,
        )
        self._seed_local_session(
            manager,
            session_id="old",
            instance_index=0,
            context_tokens=5,
            completion_ns=1,
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="unreachable",
        )
        self.assertEqual(tuple(transfer.session_id for transfer in evictions), ("old",))
        self.assertEqual(reserve_unmet, (0, 1))
        self.assertEqual(
            manager.session_snapshot("old").location,
            KVCacheManager.REMOTE_MEMORY,
        )

        second_evictions, second_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="unreachable_again",
        )
        self.assertEqual(second_evictions, ())
        self.assertEqual(second_unmet, reserve_unmet)

    def test_half_suffix_then_full_fifo_fallback_skips_active_session(self) -> None:
        _, model, _, manager = self._tiny_kv_manager(
            capacity_bytes=300,
            reserve_context_tokens=19,
            layers=4,
        )
        for session_id, completion_ns, context_tokens in (
            ("oldest", 10, 10),
            ("second", 20, 10),
            ("active", None, 5),
        ):
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=context_tokens,
                completion_ns=completion_ns,
            )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="two_stage_pressure",
        )
        self.assertEqual(reserve_unmet, ())
        self.assertEqual(
            [
                (
                    transfer.session_id,
                    transfer.layer_start,
                    transfer.layer_end,
                    transfer.resident_prefix_layers_after,
                )
                for transfer in evictions
            ],
            [
                ("oldest", 2, 4, 2),
                ("second", 2, 4, 2),
                ("oldest", 0, 2, 0),
            ],
        )
        half_shards = kv_cache_shard_bytes_for_layer_range(
            model,
            10,
            2,
            layer_start=2,
            layer_end=4,
        )
        self.assertEqual(evictions[0].total_bytes, sum(half_shards))
        self.assertEqual(
            manager.session_snapshot("oldest").location,
            KVCacheManager.REMOTE_MEMORY,
        )
        second = manager.session_snapshot("second")
        self.assertEqual(second.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(second.resident_prefix_layers, 2)
        self.assertEqual(second.local_bytes, second.remote_bytes)
        active = manager.session_snapshot("active")
        self.assertEqual(active.location, KVCacheManager.LOCAL_HBM)
        self.assertTrue(active.active)
        self.assertEqual(active.resident_prefix_layers, 4)

        before, prefix_transfer, restore, restore_evictions = manager.prepare_prefill(
            session_id="second",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="restore_suffix",
        )
        self.assertEqual(before.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertIsNone(prefix_transfer)
        self.assertEqual(restore_evictions, ())
        self.assertEqual(restore.kind, "remote_load")
        self.assertEqual((restore.layer_start, restore.layer_end), (2, 4))
        self.assertEqual(restore.total_bytes, before.remote_bytes)
        restored = manager.session_snapshot("second")
        self.assertEqual(restored.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(restored.resident_prefix_layers, 4)

    def test_typed_eviction_prefers_human_class_over_older_tool_session(self) -> None:
        # Typed eviction (2026-08-18): human-return sessions are reclaimed
        # before tool-call sessions even when the tool session has waited
        # longer (completed earlier).
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=300,
            reserve_context_tokens=19,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="tool_old",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
            next_request_type="tool",
        )
        self._seed_local_session(
            manager,
            session_id="human_new",
            instance_index=0,
            context_tokens=10,
            completion_ns=20,
            next_request_type="human",
        )
        self._seed_local_session(
            manager,
            session_id="active",
            instance_index=0,
            context_tokens=5,
            completion_ns=None,
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="typed_order",
        )
        self.assertEqual(reserve_unmet, ())
        # Human class runs to completion (half then full) before the tool
        # class gets its first half-suffix eviction.
        self.assertEqual(
            [
                (transfer.session_id, transfer.layer_start, transfer.layer_end)
                for transfer in evictions
            ],
            [
                ("human_new", 2, 4),
                ("human_new", 0, 2),
                ("tool_old", 2, 4),
            ],
        )
        self.assertEqual(
            manager.session_snapshot("human_new").location,
            KVCacheManager.REMOTE_MEMORY,
        )
        tool_snapshot = manager.session_snapshot("tool_old")
        self.assertEqual(tool_snapshot.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(tool_snapshot.resident_prefix_layers, 2)
        # The active session stays fully local regardless of class.
        active = manager.session_snapshot("active")
        self.assertEqual(active.location, KVCacheManager.LOCAL_HBM)
        self.assertTrue(active.active)

    def test_typed_eviction_orders_within_class_by_completion_then_id(self) -> None:
        # Inside one class the original FIFO key applies: oldest completion
        # first, session_id as the deterministic tie-break.
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=500,
            reserve_context_tokens=19,
            layers=4,
        )
        for session_id, completion_ns in (
            ("human_b", 30),
            ("human_a", 30),
            ("human_c", 20),
        ):
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=10,
                completion_ns=completion_ns,
                next_request_type="human",
            )
        self._seed_local_session(
            manager,
            session_id="tool_older",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
            next_request_type="tool",
        )
        self._seed_local_session(
            manager,
            session_id="active",
            instance_index=0,
            context_tokens=5,
            completion_ns=None,
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="in_class_fifo",
        )
        self.assertEqual(reserve_unmet, ())
        # human_c completed first; human_a/human_b tie on completion_ns and
        # break by session_id. The older tool session is not touched.
        self.assertEqual(
            tuple(transfer.session_id for transfer in evictions),
            ("human_c", "human_a"),
        )
        for transfer in evictions:
            self.assertEqual((transfer.layer_start, transfer.layer_end), (2, 4))
        self.assertEqual(
            manager.session_snapshot("tool_older").location,
            KVCacheManager.LOCAL_HBM,
        )

    def test_typed_eviction_halves_every_human_before_any_full_or_tool_stage(
        self,
    ) -> None:
        # Stage progression: all human sessions are halved before any human
        # full eviction, and the human class is exhausted (half + full)
        # before the tool class begins.
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=380,
            reserve_context_tokens=19,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="human_a",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
            next_request_type="human",
        )
        self._seed_local_session(
            manager,
            session_id="human_b",
            instance_index=0,
            context_tokens=10,
            completion_ns=20,
            next_request_type="human",
        )
        self._seed_local_session(
            manager,
            session_id="tool_new",
            instance_index=0,
            context_tokens=10,
            completion_ns=30,
            next_request_type="tool",
        )
        self._seed_local_session(
            manager,
            session_id="active",
            instance_index=0,
            context_tokens=5,
            completion_ns=None,
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="stage_progression",
        )
        self.assertEqual(reserve_unmet, ())
        self.assertEqual(
            [
                (transfer.session_id, transfer.layer_start, transfer.layer_end)
                for transfer in evictions
            ],
            [
                ("human_a", 2, 4),
                ("human_b", 2, 4),
                ("human_a", 0, 2),
            ],
        )
        # The watermark was met before the tool class was reached.
        self.assertEqual(
            manager.session_snapshot("tool_new").location,
            KVCacheManager.LOCAL_HBM,
        )
        self.assertEqual(
            manager.session_snapshot("human_a").location,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertEqual(
            manager.session_snapshot("human_b").location,
            KVCacheManager.PARTIAL_HBM_REMOTE,
        )

    def test_typed_eviction_stops_at_watermark_after_single_half(self) -> None:
        # One half-suffix eviction satisfies the reserve: no other session
        # (not even the remaining human prefix) is touched.
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=380,
            reserve_context_tokens=19,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="tool_old",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
            next_request_type="tool",
        )
        self._seed_local_session(
            manager,
            session_id="human_new",
            instance_index=0,
            context_tokens=10,
            completion_ns=20,
            next_request_type="human",
        )
        self._seed_local_session(
            manager,
            session_id="active",
            instance_index=0,
            context_tokens=5,
            completion_ns=None,
        )

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="early_stop",
        )
        self.assertEqual(reserve_unmet, ())
        self.assertEqual(
            [
                (transfer.session_id, transfer.layer_start, transfer.layer_end)
                for transfer in evictions
            ],
            [("human_new", 2, 4)],
        )
        human_snapshot = manager.session_snapshot("human_new")
        self.assertEqual(
            human_snapshot.location,
            KVCacheManager.PARTIAL_HBM_REMOTE,
        )
        self.assertEqual(human_snapshot.resident_prefix_layers, 2)
        self.assertEqual(
            manager.session_snapshot("tool_old").location,
            KVCacheManager.LOCAL_HBM,
        )

    def test_mark_complete_records_next_request_type_and_drives_eviction(self) -> None:
        # The type passed to mark_complete lands on the session state and is
        # the classification used by the next eviction pass; sessions with no
        # recorded type default to the human class (user ruling).
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=300,
            reserve_context_tokens=19,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="typed_tool",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
            next_request_type="tool",
        )
        self._seed_local_session(
            manager,
            session_id="untyped_old",
            instance_index=0,
            context_tokens=10,
            completion_ns=5,
        )
        self._seed_local_session(
            manager,
            session_id="active",
            instance_index=0,
            context_tokens=5,
            completion_ns=None,
        )

        self.assertEqual(manager._sessions["typed_tool"].next_request_type, "tool")
        self.assertIsNone(manager._sessions["untyped_old"].next_request_type)

        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="recorded_type",
        )
        self.assertEqual(reserve_unmet, ())
        # untyped_old (completed even earlier) belongs to the human class and
        # is evicted before the explicitly tool-typed session.
        self.assertEqual(
            [
                (transfer.session_id, transfer.layer_start, transfer.layer_end)
                for transfer in evictions
            ],
            [
                ("untyped_old", 2, 4),
                ("untyped_old", 0, 2),
                ("typed_tool", 2, 4),
            ],
        )
        with self.assertRaises(ValueError):
            manager.mark_complete("active", 99, next_request_type="voice")
        with self.assertRaises(ValueError):
            FaceRequest(0, "s", 0, "r", 2, 1, 0, None, next_trigger_type="voice")

    def test_odd_model_layer_count_derives_suffix_without_constants(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(layers=5)
        self.assertEqual(manager.partial_resident_prefix_layers, 3)

    def test_center_rank_nearest_edge_tie_and_xy_route_are_deterministic(self) -> None:
        hardware = FaceHardware(
            mesh_rows=3,
            mesh_cols=3,
            local_hbm_capacity_bytes=100,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=1.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        edges = physical_edge_ranks(hardware)
        self.assertEqual(edges, (0, 1, 2, 3, 5, 6, 7, 8))
        self.assertEqual(nearest_edge_rank(hardware, 4, edges), 1)

        nearest_route = deterministic_xy_route(hardware, 4, 1)
        self.assertEqual(nearest_route, (4, 1))
        self.assertEqual(
            len(nearest_route) - 1,
            manhattan_hops(hardware, 4, 1),
        )
        self.assertEqual(deterministic_xy_route(hardware, 4, 0), (4, 3, 0))

    def test_history_local_remote_and_other_instance_transitions(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=100,
        )
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
        )

        hbm_before_hit = manager.hbm_snapshots()
        before, prefix_transfer, local_hit, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="local_hit",
        )
        self.assertEqual(before.location, KVCacheManager.LOCAL_HBM)
        self.assertIsNone(prefix_transfer)
        self.assertEqual(local_hit.kind, "local_hit")
        self.assertEqual(local_hit.shards, ())
        self.assertEqual(evictions, ())
        self.assertEqual(manager.hbm_snapshots(), hbm_before_hit)

        manager.mark_complete("session", 20)
        before, prefix_transfer, noc_transfer, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="noc_move",
        )
        self.assertEqual(before.instance_index, 0)
        self.assertIsNone(prefix_transfer)
        self.assertEqual(noc_transfer.kind, "noc_migrate")
        self.assertEqual(evictions, ())
        self.assertEqual(
            sum(shard.bytes for shard in noc_transfer.shards),
            noc_transfer.total_bytes,
        )
        self.assertTrue(
            all(
                len(shard.noc_path) - 1
                == manhattan_hops(
                    manager.topology.hardware,
                    int(shard.source_rank),
                    int(shard.target_rank),
                )
                for shard in noc_transfer.shards
            )
        )
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 20, 20),
        )
        moved = manager.session_snapshot("session")
        self.assertEqual(moved.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(moved.instance_index, 1)
        self.assertEqual(tuple(rank for rank, _ in moved.rank_bytes), (2, 3))

        manager.mark_complete("session", 30)
        stores, reserve_unmet = manager.enforce_reserve(
            instance_index=1,
            trigger_request_id="remote_store",
        )
        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0].kind, "remote_store")
        self.assertEqual(reserve_unmet, (2, 3))
        self.assertTrue(
            all(shard.edge_rank in manager.edge_ranks for shard in stores[0].shards)
        )
        remote = manager.session_snapshot("session")
        self.assertEqual(remote.location, KVCacheManager.REMOTE_MEMORY)
        self.assertIsNone(remote.instance_index)
        self.assertEqual(remote.rank_bytes, ())
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 0, 0),
        )

        before, prefix_transfer, remote_load, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="remote_load",
        )
        self.assertEqual(before.location, KVCacheManager.REMOTE_MEMORY)
        self.assertIsNone(prefix_transfer)
        self.assertEqual(remote_load.kind, "remote_load")
        self.assertEqual(evictions, ())
        self.assertEqual(
            sum(shard.bytes for shard in remote_load.shards),
            remote_load.total_bytes,
        )
        self.assertTrue(
            all(
                shard.edge_rank in manager.edge_ranks
                and shard.source_rank == shard.edge_rank
                for shard in remote_load.shards
            )
        )
        restored = manager.session_snapshot("session")
        self.assertEqual(restored.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(restored.instance_index, 0)
        self.assertEqual(tuple(rank for rank, _ in restored.rank_bytes), (0, 1))
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (20, 20, 0, 0),
        )

    def test_partial_history_cross_instance_migrates_prefix_then_loads_suffix(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=116,
            reserve_context_tokens=4,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=4,
            completion_ns=10,
        )
        evictions, reserve_unmet = manager.enforce_reserve(
            instance_index=0,
            trigger_request_id="force_partial",
        )
        self.assertEqual(reserve_unmet, ())
        self.assertEqual(len(evictions), 1)
        partial = manager.session_snapshot("session")
        self.assertEqual(partial.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(partial.instance_index, 0)
        self.assertEqual(partial.resident_prefix_layers, 2)

        reservation_evictions = manager.reserve_request_capacity(
            request_id="turn1",
            session_id="session",
            instance_index=1,
            final_context_tokens=4,
        )
        self.assertEqual(reservation_evictions, ())
        before, prefix_transfer, suffix_transfer, prepare_evictions = (
            manager.prepare_prefill(
                session_id="session",
                target_instance_index=1,
                history_tokens=4,
                trigger_request_id="turn1",
                reservation_request_id="turn1",
            )
        )

        self.assertEqual(before, partial)
        self.assertEqual(prepare_evictions, ())
        self.assertIsNotNone(prefix_transfer)
        self.assertEqual(prefix_transfer.kind, "noc_migrate")
        self.assertEqual(prefix_transfer.phase, "history")
        self.assertEqual(
            prefix_transfer.reason,
            "history_partial_prefix_migrate",
        )
        self.assertEqual(
            (prefix_transfer.source_instance_index, prefix_transfer.target_instance_index),
            (0, 1),
        )
        self.assertEqual(
            (prefix_transfer.layer_start, prefix_transfer.layer_end),
            (0, 2),
        )
        self.assertEqual(
            (
                prefix_transfer.resident_prefix_layers_before,
                prefix_transfer.resident_prefix_layers_after,
            ),
            (2, 2),
        )
        self.assertEqual(suffix_transfer.kind, "remote_load")
        self.assertEqual(
            (suffix_transfer.source_instance_index, suffix_transfer.target_instance_index),
            (None, 1),
        )
        self.assertEqual(
            (suffix_transfer.layer_start, suffix_transfer.layer_end),
            (2, 4),
        )

        restored = manager.session_snapshot("session")
        self.assertEqual(restored.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(restored.instance_index, 1)
        self.assertEqual(restored.resident_prefix_layers, 4)
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 32, 32),
        )
        manager.release_request_capacity_reservation("turn1")



    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_default_config_uses_hbm160_edge_pool_and_one_million_reserve(self) -> None:
        config = load_face_trace_config()
        self.assertEqual(
            config.hardware.local_hbm_capacity_bytes,
            160 * 1024**3,
        )
        self.assertIn("160 GiB", config.hardware.label)
        self.assertEqual(
            config.hardware_config.name,
            "face_case5_config_c.json",
        )
        self.assertEqual(
            config.hardware_capacity_profile,
            "validation-160gib",
        )
        self.assertEqual(
            config.system_template.name,
            "llama2_7b_roofline_template.json",
        )
        self.assertEqual(
            config.system_config.name,
            "system.json",
        )
        self.assertEqual(config.network_config.name, "network.yml")
        self.assertEqual(config.comm_group_config.name, "comm_group.json")
        self.assertEqual(
            config.hardware_metadata["remote-memory"]["memory-type"],
            "PER_NPU_MEMORY_EXPANSION",
        )
        self.assertEqual(len(config.configuration_digest), 8)
        self.assertEqual(config.kv_reserve_context_tokens, 1_000_000)
        self.assertEqual(
            config.remote_memory.edge_npus,
            physical_edge_ranks(config.hardware),
        )
        self.assertEqual(len(config.remote_memory.edge_npus), 26)
        self.assertEqual(config.remote_memory.remote_mem_bw_gbps, 512.0)
        self.assertEqual(config.remote_memory.remote_mem_latency_ns, 100)
        self.assertEqual(
            config.remote_memory.logical_pool,
            "unified-kv-cache-pool",
        )
        with config.remote_memory.path.open(encoding="utf-8") as source:
            remote_raw = json.load(source)
        self.assertEqual(
            remote_raw["memory-type"],
            "PER_NPU_MEMORY_EXPANSION",
        )
        self.assertEqual(
            tuple(remote_raw["npu-ids"]),
            config.remote_memory.edge_npus,
        )
        with config.system_config.open(encoding="utf-8") as source:
            system_raw = json.load(source)
        self.assertEqual(system_raw["local-mem-bw"], 1640.0)
        self.assertEqual(system_raw["local-mem-capacity-bytes"], 160 * 1024**3)
        self.assertEqual(system_raw["remote-mem-bw"], 512)
        self.assertEqual(system_raw["remote-mem-latency"], 100)
        self.assertEqual(system_raw["peak-perf"], 261.12)
        self.assertEqual(system_raw["hbm-kv-restore-bandwidth-sharing"], 1)

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_et_kv_migration_ack_and_cross_instance_store_trigger(self) -> None:
        config = load_face_trace_config()
        group_by_index = dict(enumerate(config.inference_groups))
        builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in range(config.npus_count)
        }
        control_group = group_by_index[0]
        timer_gates = tuple(
            builders[rank].timer_gate(f"control_rank{rank}", 1_000)
            for rank in control_group.ranks
        )
        store = KVTransfer(
            kind="remote_store",
            phase="completion",
            reason="test_cross_instance_trigger",
            session_id="victim",
            trigger_request_id="trigger",
            source_instance_index=1,
            target_instance_index=None,
            total_bytes=64,
            shards=(
                KVTransferShard(
                    source_rank=2,
                    target_rank=2,
                    edge_rank=2,
                    bytes=64,
                    noc_path=(2,),
                    layer_start=16,
                    layer_end=32,
                ),
            ),
            model_layers=32,
            layer_start=16,
            layer_end=32,
            resident_prefix_layers_before=32,
            resident_prefix_layers_after=16,
        )
        store_record = _emit_kv_transfer(
            config=config,
            builders=builders,
            group_by_index=group_by_index,
            tag_allocator=TransferTagAllocator(),
            transfer=store,
            action_name="store",
            trigger_gate=TransferTriggerGate(
                control_instance_index=0,
                node_gates=timer_gates,
            ),
        )["shards"][0]

        control_rank = control_group.ranks[0]
        trigger_send = builders[control_rank].nodes[-1]
        self.assertEqual(trigger_send.type, COMM_SEND_NODE)
        self.assertIn(timer_gates[0], trigger_send.data_deps)
        self.assertIsNotNone(store_record["trigger_tag"])
        source_nodes = builders[2].nodes
        self.assertEqual(
            [node.type for node in source_nodes],
            [COMM_RECV_NODE, MEM_STORE_NODE],
        )
        self.assertIn(source_nodes[0].id, source_nodes[1].data_deps)

        noc_builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in range(config.npus_count)
        }
        noc = KVTransfer(
            kind="noc_migrate",
            phase="history",
            reason="test_source_release_ack",
            session_id="moving",
            trigger_request_id="request",
            source_instance_index=0,
            target_instance_index=1,
            total_bytes=64,
            shards=(
                KVTransferShard(
                    source_rank=20,
                    target_rank=2,
                    edge_rank=None,
                    bytes=64,
                    noc_path=deterministic_xy_route(config.hardware, 20, 2),
                    layer_start=0,
                    layer_end=32,
                ),
            ),
            model_layers=32,
            layer_start=0,
            layer_end=32,
            resident_prefix_layers_before=32,
            resident_prefix_layers_after=32,
        )
        noc_record = _emit_kv_transfer(
            config=config,
            builders=noc_builders,
            group_by_index=group_by_index,
            tag_allocator=TransferTagAllocator(),
            transfer=noc,
            action_name="noc",
        )["shards"][0]
        self.assertEqual(
            noc_record["source_release_dependency"],
            "noc_migration_ack_recv",
        )
        self.assertIsNotNone(noc_record["ack_tag"])
        self.assertEqual(
            [node.type for node in noc_builders[20].nodes],
            [COMM_SEND_NODE, COMM_RECV_NODE],
        )
        self.assertEqual(
            [node.type for node in noc_builders[2].nodes],
            [COMM_RECV_NODE, COMM_SEND_NODE],
        )
        self.assertIn(
            noc_builders[20].nodes[0].id,
            noc_builders[20].nodes[1].data_deps,
        )
        self.assertIn(
            noc_builders[2].nodes[0].id,
            noc_builders[2].nodes[1].data_deps,
        )

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_tp_readiness_barrier_waits_for_every_rank_local_predecessor(self) -> None:
        config = load_face_trace_config()
        group = config.inference_groups[0]
        builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in group.ranks
        }
        predecessor_ids: dict[int, int] = {}
        for rank in group.ranks:
            builders[rank].comp(f"rank{rank}_kv_ready", 1, 1)
            predecessor = builders[rank].previous_id
            self.assertIsNotNone(predecessor)
            predecessor_ids[rank] = int(predecessor)

        record = _emit_tp_readiness_barrier(
            builders=builders,
            group=group,
            name="request_prefill_kv_ready_barrier",
        )

        self.assertEqual(record["collective"], "all_reduce")
        self.assertEqual(record["comm_size_bytes"], 1)
        self.assertEqual(record["pg_name"], group.pg_name)
        self.assertEqual(record["ranks"], list(group.ranks))
        self.assertEqual(len(record["node_ids_by_rank"]), len(group.ranks))
        node_ids_by_rank = dict(record["node_ids_by_rank"])
        for rank in group.ranks:
            barrier = builders[rank].nodes[-1]
            self.assertEqual(barrier.type, COMM_COLL_NODE)
            self.assertEqual(barrier.name, "request_prefill_kv_ready_barrier")
            self.assertEqual(barrier.id, node_ids_by_rank[rank])
            self.assertIn(predecessor_ids[rank], barrier.data_deps)
            attrs = {attr.name: attr for attr in barrier.attr}
            self.assertEqual(attrs["comm_type"].uint64_val, ALL_REDUCE)
            self.assertEqual(attrs["comm_size"].uint64_val, 1)
            self.assertEqual(attrs["pg_name"].string_val, group.pg_name)

            builders[rank].comp(f"rank{rank}_first_prefill_compute", 1, 1)
            first_compute = builders[rank].nodes[-1]
            self.assertIn(barrier.id, first_compute.data_deps)

    @unittest.skipUnless(
        _MATERIALIZED,
        "materialized 30s input absent (request-neutral bare repo)",
    )
    def test_remote_suffix_load_emits_target_hbm_dma_and_branch_gate(self) -> None:
        config = load_face_trace_config()
        group_by_index = dict(enumerate(config.inference_groups))
        target_group = group_by_index[0]
        target_rank = target_group.ranks[0]
        edge_rank = 2
        builders = {
            rank: TraceBuilder(remote_operand_loads=False)
            for rank in range(config.npus_count)
        }
        for rank in target_group.ranks:
            builders[rank].comp(f"rank{rank}_prefix_ready", 1, 1)
        checkpoint = builders[target_rank].chain_checkpoint()
        timer_gates = tuple(
            builders[rank].previous_id for rank in target_group.ranks
        )
        transfer = KVTransfer(
            kind="remote_load",
            phase="history",
            reason="test_suffix_pipeline",
            session_id="partial",
            trigger_request_id="next_turn",
            source_instance_index=None,
            target_instance_index=0,
            total_bytes=64,
            shards=(
                KVTransferShard(
                    source_rank=edge_rank,
                    target_rank=target_rank,
                    edge_rank=edge_rank,
                    bytes=64,
                    noc_path=deterministic_xy_route(
                        config.hardware, edge_rank, target_rank
                    ),
                    layer_start=16,
                    layer_end=32,
                ),
            ),
            model_layers=32,
            layer_start=16,
            layer_end=32,
            resident_prefix_layers_before=16,
            resident_prefix_layers_after=32,
        )
        tag_allocator = TransferTagAllocator()
        record = _emit_kv_transfer(
            config=config,
            builders=builders,
            group_by_index=group_by_index,
            tag_allocator=tag_allocator,
            transfer=transfer,
            action_name="suffix_restore",
            pending_gate=PendingHistoryGate(
                source_instance_index=0,
                timer_gates=timer_gates,
                location="partial_hbm_remote",
            ),
        )["shards"][0]
        dma_node_id = record["target_hbm_completion_node_id"]
        dma_node = next(
            node for node in builders[target_rank].nodes if node.id == dma_node_id
        )
        self.assertEqual(dma_node.type, MEM_LOAD_NODE)
        attrs = {attr.name: attr for attr in dma_node.attr}
        self.assertTrue(attrs["is_local_hbm_kv_restore"].bool_val)

        for rank in target_group.ranks:
            if rank != target_rank:
                builders[rank].local_hbm_kv_restore(
                    f"rank{rank}_suffix_restore",
                    64,
                )
        readiness = _emit_tp_point_to_point_readiness_barrier(
            builders=builders,
            group=target_group,
            tag_allocator=tag_allocator,
            name="suffix_ready_barrier",
        )
        self.assertIsNone(readiness["collective"])
        barrier_tags = [
            action["tag"]
            for action in (
                *readiness["arrival_actions"],
                *readiness["release_actions"],
            )
        ]
        self.assertEqual(len(barrier_tags), len(set(barrier_tags)))
        ready_node_id = dict(readiness["node_ids_by_rank"])[target_rank]
        ready_node = next(
            node for node in builders[target_rank].nodes if node.id == ready_node_id
        )
        self.assertEqual(ready_node.type, COMM_SEND_NODE)
        first_arrival_recv_id = readiness["arrival_actions"][0][
            "control_recv_node_id"
        ]
        first_arrival_recv = next(
            node
            for node in builders[target_rank].nodes
            if node.id == first_arrival_recv_id
        )
        self.assertIn(dma_node.id, first_arrival_recv.data_deps)

        builders[target_rank].restore_chain(checkpoint)
        builders[target_rank].comp("resident_prefix_compute", 1, 1)
        prefix_compute = builders[target_rank].nodes[-1]
        self.assertNotIn(dma_node.id, prefix_compute.data_deps)
        builders[target_rank].arm_dependency(ready_node.id)
        builders[target_rank].comp("first_suffix_compute", 1, 1)
        suffix_compute = builders[target_rank].nodes[-1]
        self.assertIn(prefix_compute.id, suffix_compute.data_deps)
        self.assertIn(ready_node.id, suffix_compute.data_deps)

    def test_graph_batch_cross_instance_partial_history_pipelines_suffix_restore(self) -> None:
        hardware, _, _, _ = self._tiny_kv_manager(
            capacity_bytes=116,
            reserve_context_tokens=4,
            layers=4,
        )
        groups = (
            SimpleNamespace(name="ins0", pg_name="1", ranks=(0, 1)),
            SimpleNamespace(name="ins1", pg_name="2", ranks=(2, 3)),
        )
        config = SimpleNamespace(
            npus_count=4,
            inference_groups=groups,
            remote_operand_loads=False,
            trace_granularity="request_aggregated",
            prefill_chunk_size=4,
            layers=4,
            hidden_size=2,
            ffn_size=2,
            vocab_size=2,
            bytes_per_elem=1,
            num_heads=2,
            mlp_variant="gelu",
            hardware=hardware,
            remote_memory=SimpleNamespace(edge_npus=(0, 1, 2, 3)),
            request_queue=(
                RequestSpec("session", 1, "turn1", 2, 1, None, 1000),
            ),
        )
        prefix_transfer = KVTransfer(
            kind="noc_migrate",
            phase="history",
            reason="history_partial_prefix_migrate",
            session_id="session",
            trigger_request_id="turn1",
            source_instance_index=0,
            target_instance_index=1,
            total_bytes=32,
            shards=tuple(
                KVTransferShard(
                    source_rank=source_rank,
                    target_rank=target_rank,
                    edge_rank=None,
                    bytes=16,
                    noc_path=deterministic_xy_route(
                        hardware, source_rank, target_rank
                    ),
                    layer_start=0,
                    layer_end=2,
                )
                for source_rank, target_rank in ((0, 2), (1, 3))
            ),
            model_layers=4,
            layer_start=0,
            layer_end=2,
            resident_prefix_layers_before=2,
            resident_prefix_layers_after=2,
        )
        suffix_transfer = KVTransfer(
            kind="remote_load",
            phase="history",
            reason="history_remote_suffix_restore",
            session_id="session",
            trigger_request_id="turn1",
            source_instance_index=None,
            target_instance_index=1,
            total_bytes=32,
            shards=tuple(
                KVTransferShard(
                    source_rank=edge_rank,
                    target_rank=target_rank,
                    edge_rank=edge_rank,
                    bytes=16,
                    noc_path=deterministic_xy_route(
                        hardware, edge_rank, target_rank
                    ),
                    layer_start=2,
                    layer_end=4,
                )
                for edge_rank, target_rank in ((2, 2), (3, 3))
            ),
            model_layers=4,
            layer_start=2,
            layer_end=4,
            resident_prefix_layers_before=2,
            resident_prefix_layers_after=4,
        )
        request_plan = {
            "queue_index": 0,
            "session_id": "session",
            "turn_index": 1,
            "request_id": "turn1",
            "prefill_instance_index": 1,
            "history_tokens_before": 4,
            "prefill_context_tokens": 6,
            "history_location_before": SimpleNamespace(
                location="partial_hbm_remote",
                instance_index=0,
                resident_prefix_layers=2,
            ),
            "history_prefix_transfer": prefix_transfer,
            "history_transfer": suffix_transfer,
            "history_evictions": (),
            "prefill_evictions": (),
        }
        graph = GraphBatchBuilder(config)
        graph.begin_batch()
        graph.pending_history["turn1"] = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=(None, None),
            location="partial_hbm_remote",
        )
        members = graph.emit_prefill_batch(request_plan)
        self.assertEqual(set(members), {2, 3})

        for rank in groups[1].ranks:
            builder = graph.builders[rank]
            nodes = builder.nodes

            def node_ids(fragment: str) -> list[int]:
                return [
                    node["id"] for node in nodes
                    if fragment in node["name"]
                ]

            prefix_ready = max(node_ids("prefill_prefix_ready_barrier"))
            suffix_ready = max(node_ids("prefill_suffix_ready_barrier"))
            suffix_restore = max(node_ids("history_transfer_action"))
            prefix_nodes = node_ids("prefill_first_chunk_prefix")
            suffix_nodes = node_ids("prefill_first_chunk_suffix")
            prefix_first = min(prefix_nodes)
            prefix_last = max(prefix_nodes)
            suffix_first = min(suffix_nodes)
            parents = {
                edge["from"] for edge in builder.edges
                if edge["to"] == prefix_first
            }
            self.assertIn(prefix_ready, parents)
            self.assertNotIn(suffix_restore, parents)
            self.assertNotIn(suffix_ready, parents)

            suffix_parents = {
                edge["from"] for edge in builder.edges
                if edge["to"] == suffix_first
            }
            self.assertIn(prefix_last, suffix_parents)
            self.assertIn(suffix_ready, suffix_parents)

    def test_aggregated_transformer_pass_preserves_expanded_totals(self) -> None:
        spans = ((3, 7), (1, 11), (5, 19))
        parameters = {
            "layers": 3,
            "hidden_size": 24,
            "ffn_size": 49,
            "tensor_parallel": 4,
            "pg_name": "1",
            "vocab_size": 101,
            "bytes_per_elem": 2,
            "num_heads": 6,
            "tensor_parallel_rank": 0,
            "mlp_variant": "swiglu",
        }
        expanded = TraceBuilder(remote_operand_loads=True)
        for index, (tokens, kv_length) in enumerate(spans):
            transformer_pass(
                expanded,
                phase=f"expanded_{index}",
                tokens=tokens,
                kv_length=kv_length,
                **parameters,
            )
        aggregated = TraceBuilder(remote_operand_loads=True)
        pass_count = transformer_pass_aggregated(
            aggregated,
            phase="aggregated",
            pass_spans=spans,
            **parameters,
        )

        def attribute(node: object, name: str) -> int:
            for attr in node.attr:
                if attr.name == name:
                    return int(attr.uint64_val)
            return 0

        def totals(builder: TraceBuilder) -> tuple[int, int, int, int]:
            compute_nodes = [node for node in builder.nodes if node.type == COMP_NODE]
            collective_nodes = [
                node for node in builder.nodes if node.type == COMM_COLL_NODE
            ]
            return (
                sum(attribute(node, "num_ops") for node in compute_nodes),
                sum(attribute(node, "tensor_size") for node in compute_nodes),
                sum(attribute(node, REMOTE_WEIGHT_ATTR) for node in compute_nodes),
                sum(attribute(node, "comm_size") for node in collective_nodes),
            )

        self.assertEqual(pass_count, len(spans))
        self.assertEqual(len(aggregated.nodes), 17)
        self.assertEqual(
            sum(node.type == COMP_NODE for node in aggregated.nodes),
            15,
        )
        self.assertEqual(
            sum(node.type == COMM_COLL_NODE for node in aggregated.nodes),
            2,
        )
        self.assertEqual(totals(aggregated), totals(expanded))

    def test_prefill_task_load_order_and_deterministic_instance_tie(self) -> None:
        loads = (
            InstanceTaskLoadSnapshot(0, 10, 20, 30, 1),
            InstanceTaskLoadSnapshot(1, 0, 20, 10, 20),
            InstanceTaskLoadSnapshot(2, 5, 20, 10, None),
        )
        self.assertEqual(select_prefill_instance(loads, (True, True, True)), 1)
        tied = (
            InstanceTaskLoadSnapshot(2, 0, 0, 0, None),
            InstanceTaskLoadSnapshot(1, 0, 0, 0, 10),
        )
        self.assertEqual(select_prefill_instance(tied, (True, True, True)), 2)
        exact_tie = (
            InstanceTaskLoadSnapshot(2, 0, 0, 0, 10),
            InstanceTaskLoadSnapshot(1, 0, 0, 0, 10),
        )
        self.assertEqual(
            select_prefill_instance(exact_tie, (True, True, True)),
            1,
        )
        self.assertEqual(
            select_prefill_instance(exact_tie, (True, False, True)),
            2,
        )

    def test_roofline_task_load_accounts_for_history_and_decode_mean_cutoff(self) -> None:
        hardware, _ = line_topology()
        model = FaceModel(
            layers=2,
            hidden_size=16,
            ffn_size=32,
            num_heads=4,
            vocab_size=32,
            bytes_per_elem=2,
            mlp_variant="swiglu",
        )
        short_prefill = estimate_prefill_task_load_ns(
            hardware,
            model,
            instance_size=2,
            chunk_tokens=64,
            context_tokens=64,
        )
        history_prefill = estimate_prefill_task_load_ns(
            hardware,
            model,
            instance_size=2,
            chunk_tokens=64,
            context_tokens=4096,
        )
        self.assertGreater(history_prefill, short_prefill)

        short_decode = estimate_decode_remaining_task_load_ns(
            hardware,
            model,
            instance_size=2,
            current_context_tokens=64,
            generated_tokens=5,
            average_decode_length=10.5,
        )
        history_decode = estimate_decode_remaining_task_load_ns(
            hardware,
            model,
            instance_size=2,
            current_context_tokens=4096,
            generated_tokens=5,
            average_decode_length=10.5,
        )
        self.assertGreater(history_decode, short_decode)
        self.assertEqual(
            estimate_decode_remaining_task_load_ns(
                hardware,
                model,
                instance_size=2,
                current_context_tokens=4102,
                generated_tokens=11,
                average_decode_length=10.5,
            ),
            0,
        )


    def test_decode_uses_exact_roofline_tokens_and_is_deterministic(self) -> None:
        hardware, topology = line_topology()
        model = FaceModel(
            layers=2,
            hidden_size=16,
            ffn_size=32,
            num_heads=4,
            vocab_size=32,
            bytes_per_elem=2,
            mlp_variant="swiglu",
        )
        graph = WeightedInstanceGraph(topology)
        kwargs = {
            "hardware": hardware,
            "model": model,
            "topology": topology,
            "graph": graph,
            "fixed_p_chunk": 64,
            "prefill_instance_index": 1,
            "has_prefill_work": (True, False, False),
            "decode_token_lengths": ((383,), (), ()),
            "new_request_token_length": 384,
            "remaining_hbm_capacity_bytes": (100, 100, 100),
            "hbm_feasible_instances": (True, True, True),
        }
        selected, costs = select_decode_instance(**kwargs)
        repeated_selected, repeated_costs = select_decode_instance(**kwargs)

        self.assertEqual(selected, repeated_selected)
        self.assertEqual(costs, repeated_costs)
        first = costs[0]
        self.assertIsInstance(first.current_roofline, FaceRooflineEstimate)
        self.assertEqual(first.current_roofline.d_token, 383)
        self.assertEqual(first.updated_roofline.d_token, 384)
        self.assertEqual(first.current_roofline.p_chunk, 64)
        self.assertEqual(
            first.current_roofline.iteration_time_ns,
            estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=2,
                p_chunk=64,
                d_batch=1,
                d_token=383,
            ),
        )
        self.assertEqual(first.current_roofline.source, "analytical_roofline")

    def test_roofline_observability_bins_token_lengths(self) -> None:
        accumulator = PlannerRooflineStatsAccumulator()
        accumulator.record_roofline_iteration(
            FaceRooflineEstimate(
                instance_size=2,
                p_chunk=64,
                d_batch=1,
                d_token=383,
                iteration_time_ns=17,
            ),
            10,
            27,
        )
        records = accumulator.to_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "planner_roofline_iteration_stats")
        self.assertEqual(records[0]["source"], "planner_roofline")
        self.assertEqual(records[0]["kv_bin"], 512)
        self.assertNotIn("kv_length", records[0])
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = write_planner_roofline_stats(
                accumulator, output_dir=Path(temporary)
            )
            self.assertEqual(sidecar.name, "planner_roofline_stats.json")
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8"))["source"],
                "planner_roofline",
            )

    def test_weighted_schedulable_range_changes_after_update(self) -> None:
        hardware, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        before = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in before), (0, 1, 2))
        graph.increase_path((0, 1))
        after = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in after), (0, 1))

    def test_decode_roofline_candidate_hbm_invariants(self) -> None:
        hardware, topology = line_topology()
        model = FaceModel(
            layers=2,
            hidden_size=16,
            ffn_size=32,
            num_heads=4,
            vocab_size=32,
            bytes_per_elem=2,
            mlp_variant="swiglu",
        )
        graph = WeightedInstanceGraph(topology)
        selected, costs = select_decode_instance(
            hardware=hardware,
            model=model,
            topology=topology,
            graph=graph,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((), (), ()),
            new_request_token_length=257,
            remaining_hbm_capacity_bytes=(100, 500, 300),
            hbm_feasible_instances=(True, True, True),
        )
        self.assertEqual(selected, 1)
        self.assertEqual([cost.instance_index for cost in costs], [0, 1, 2])
        self.assertEqual(
            [cost.remaining_hbm_capacity_bytes for cost in costs],
            [100, 500, 300],
        )
        self.assertTrue(all(cost.updated_roofline.d_token == 257 for cost in costs))
        self.assertTrue(all(cost.current_roofline.d_token == 0 for cost in costs))
        self.assertEqual(
            {cost.per_die_delta_ns for cost in costs},
            {costs[0].per_die_delta_ns},
        )

        selected_with_capacity_filter, filtered_costs = select_decode_instance(
            hardware=hardware,
            model=model,
            topology=topology,
            graph=graph,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((), (), ()),
            new_request_token_length=257,
            remaining_hbm_capacity_bytes=(100, 500, 300),
            hbm_feasible_instances=(True, False, True),
        )
        self.assertEqual(selected_with_capacity_filter, 2)
        self.assertFalse(filtered_costs[1].hbm_feasible)
        with self.assertRaises(ValueError):
            select_decode_instance(
                hardware=hardware,
                model=model,
                topology=topology,
                graph=graph,
                fixed_p_chunk=64,
                prefill_instance_index=1,
                has_prefill_work=(False, False, False),
                decode_token_lengths=((), (), ()),
                new_request_token_length=257,
                remaining_hbm_capacity_bytes=(100, 500, 300),
                hbm_feasible_instances=(False, False, False),
            )

    def test_kv_local_first_offload_updates_and_release_restores_weight(self) -> None:
        _, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        allocator = KVAllocator(topology, graph, model_weight_bytes=0)
        allocation = allocator.allocate(
            request_id="r0",
            decode_instance_index=0,
            candidate_indices=(0, 1, 2),
            total_bytes=250,
        )
        self.assertEqual(
            [(piece.instance_index, piece.bytes) for piece in allocation.pieces],
            [(0, 200), (1, 50)],
        )
        self.assertEqual(graph.edge_weight(0, 1), 2)
        allocator.release(allocation)
        self.assertEqual(graph.edge_weight(0, 1), 1)
        self.assertEqual(tuple(allocator.remaining_capacity), (200, 200, 200))

        # At equal weighted distance, prefer the instance with more remaining
        # HBM before using config order.
        graph2 = WeightedInstanceGraph(topology)
        allocator2 = KVAllocator(topology, graph2, model_weight_bytes=0)
        allocator2.remaining_capacity[:] = [100, 0, 150]
        allocation2 = allocator2.allocate(
            request_id="r1",
            decode_instance_index=1,
            candidate_indices=(0, 1, 2),
            total_bytes=120,
        )
        self.assertEqual(allocation2.pieces[0].instance_index, 2)





if __name__ == "__main__":
    unittest.main()
