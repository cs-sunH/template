#!/usr/bin/env python3
"""Focused tests for the trace-generation-time FACE scheduler."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceLut,
    FaceLutEntry,
    FaceModel,
    FaceRequest,
    KVAllocator,
    PrefillQueueSnapshot,
    SessionKVCacheManager,
    WeightedInstanceGraph,
    attention_heads_by_tp_rank,
    build_instances,
    estimate_model_weight_bytes,
    kv_cache_shard_bytes_for_tokens,
    model_weight_shard_bytes_by_tp_rank,
    plan_face_requests,
    select_decode_instance,
    select_prefill_instance,
)
from generate_face_trace import (  # noqa: E402
    load_face_trace_config,
    select_first_session_requests,
)
from generate_trace import (  # noqa: E402
    COMM_COLL_NODE,
    COMP_NODE,
    REMOTE_WEIGHT_ATTR,
    RequestSpec,
    TraceBuilder,
    load_remote_memory_config,
    shard_extent,
    transformer_pass,
    transformer_pass_aggregated,
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

    def test_checked_in_three_minute_workload_configuration(self) -> None:
        config = load_face_trace_config()
        self.assertEqual(config.model_name, "llama2_7b")
        self.assertEqual(config.mlp_variant, "swiglu")
        self.assertEqual(config.npus_count, 54)
        self.assertEqual((config.hardware.mesh_rows, config.hardware.mesh_cols), (9, 6))
        self.assertEqual(config.hardware.d2d_latency_ns, 5)
        self.assertEqual(config.hardware.local_hbm_latency_ns, 100)
        self.assertEqual(config.hardware.local_hbm_capacity_bytes, 160 * 1024**3)
        self.assertEqual(config.hardware_config.name, "face_case5_config_c.json")
        self.assertEqual(config.hardware_capacity_profile, "validation-160gib")
        self.assertEqual(config.system_config.name, "system.json")
        self.assertEqual(config.network_config.name, "network.yml")
        self.assertEqual(config.comm_group_config.name, "comm_group.json")
        self.assertEqual(config.remote_memory_config.name, "remote_memory.json")
        with config.system_config.open(encoding="utf-8") as source:
            system_raw = json.load(source)
        self.assertEqual(system_raw["local-mem-bw"], 1640.0)
        self.assertEqual(system_raw["local-mem-capacity-bytes"], 160 * 1024**3)
        self.assertEqual(system_raw["remote-mem-bw"], 512)
        self.assertEqual(system_raw["peak-perf"], 261.12)
        self.assertIn("npus_count: [ 6, 9 ]", config.network_config.read_text())
        self.assertEqual(len(config.inference_groups), 9)
        self.assertEqual({len(group.ranks) for group in config.inference_groups}, {6})
        self.assertEqual(
            (config.layers, config.hidden_size, config.ffn_size, config.num_heads, config.vocab_size),
            (32, 4096, 11008, 32, 32000),
        )
        self.assertEqual(config.request_queue_session_limit, 0)
        self.assertEqual(config.trace_granularity, "request_aggregated")
        self.assertEqual(config.prefill_chunk_size, 512)
        self.assertEqual(config.kv_cache_policy, "session_lru_recompute")
        self.assertEqual(config.kv_reserve_context_tokens, 1_000_000)
        self.assertFalse(config.record_planning_iterations)
        self.assertEqual(config.source_request_count, 2091)
        self.assertEqual(config.source_session_count, 136)
        self.assertEqual(len(config.request_queue), 2091)
        self.assertEqual(
            config.selected_session_ids,
            tuple(str(index) for index in range(136)),
        )
        self.assertEqual(
            {request.session_id for request in config.request_queue},
            set(config.selected_session_ids),
        )
        prefill_lengths = [
            request.prefill_length for request in config.request_queue
        ]
        decode_lengths = [request.decode_length for request in config.request_queue]
        self.assertEqual((min(prefill_lengths), max(prefill_lengths)), (3, 158929))
        self.assertEqual((min(decode_lengths), max(decode_lengths)), (1, 32000))

    def test_exact_tp_shards_and_session_lru_recompute(self) -> None:
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        self.assertEqual(attention_heads_by_tp_rank(5, 3), (2, 2, 1))
        self.assertEqual(
            sum(kv_cache_shard_bytes_for_tokens(model, 17, 2)),
            2 * model.layers * 17 * model.hidden_size * model.bytes_per_elem,
        )
        self.assertEqual(
            sum(model_weight_shard_bytes_by_tp_rank(model, 2)),
            estimate_model_weight_bytes(model),
        )
        hardware = FaceHardware(3, 2, 400, 1.0, 2.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                FaceInstanceSpec("ins0", "1", (0, 1)),
                FaceInstanceSpec("ins1", "2", (2, 3)),
                FaceInstanceSpec("ins2", "3", (4, 5)),
            ),
        )
        manager = SessionKVCacheManager(topology, model, reserve_context_tokens=10)
        for session_id, request_id, completion in (("a", "a0", 10), ("b", "b0", 20)):
            decision = manager.prepare_history(
                session_id, 0, 0, completion, request_id, required_context_tokens=40
            )
            self.assertFalse(decision.admission_blocked)
            self.assertTrue(manager.grow_prefill(session_id, 40, completion, request_id).admitted)
            manager.mark_complete(session_id, completion, request_id)
        snapshot_a = manager.session_snapshot("a")
        self.assertIsNotNone(snapshot_a)
        self.assertEqual(snapshot_a.state, "EVICTED")
        self.assertEqual(snapshot_a.logical_context_tokens, 40)
        recompute = manager.prepare_history("a", 1, 40, 30, "a1", required_context_tokens=41)
        self.assertEqual(recompute.action, "RECOMPUTE")
        self.assertEqual(recompute.recompute_tokens, 40)

    def test_llama2_7b_tp6_partition_is_exact_without_model_padding(self) -> None:
        config = load_face_trace_config()
        attention_heads = tuple(
            shard_extent(config.num_heads, 6, index) for index in range(6)
        )
        self.assertEqual(sum(attention_heads), 32)
        self.assertEqual(attention_heads.count(6), 2)
        self.assertEqual(attention_heads.count(5), 4)
        self.assertEqual(
            sum(shard_extent(config.ffn_size, 6, index) for index in range(6)),
            config.ffn_size,
        )
        self.assertEqual(
            sum(shard_extent(config.vocab_size, 6, index) for index in range(6)),
            config.vocab_size,
        )
        self.assertEqual(
            estimate_model_weight_bytes(config.model), 13_476_831_232
        )

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

    def test_trace_builder_streams_without_retaining_nodes(self) -> None:
        streamed = []
        builder = TraceBuilder(
            remote_operand_loads=False,
            node_sink=streamed.append,
            retain_nodes=False,
        )
        builder.comp("first", num_ops=1, tensor_size=1)
        builder.comp("second", num_ops=2, tensor_size=2)
        self.assertEqual(builder.node_count, 2)
        self.assertFalse(builder.nodes)
        self.assertEqual([node.name for node in streamed], ["first", "second"])
        self.assertEqual(tuple(streamed[1].data_deps), (streamed[0].id,))

    def test_prefill_queue_order_and_deterministic_third_key(self) -> None:
        queues = (
            PrefillQueueSnapshot(0, 2, 10),
            PrefillQueueSnapshot(1, 1, 20),
            PrefillQueueSnapshot(2, 1, None),
        )
        self.assertEqual(select_prefill_instance(queues), 2)
        tied = (
            PrefillQueueSnapshot(2, 0, None),
            PrefillQueueSnapshot(1, 0, None),
        )
        self.assertEqual(select_prefill_instance(tied), 1)

    def test_lut_exact_filters_and_nearest_token_tie(self) -> None:
        lut = FaceLut(
            (
                FaceLutEntry(2, 0, 1, 256, 10),
                FaceLutEntry(2, 0, 1, 512, 20),
                FaceLutEntry(2, 64, 1, 256, 30),
            )
        )
        row = lut.lookup(instance_size=2, p_chunk=0, d_batch=1, d_token=384)
        self.assertEqual(row.d_token, 256)
        with self.assertRaises(KeyError):
            lut.lookup(instance_size=2, p_chunk=0, d_batch=2, d_token=384)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "face_lut.csv"
            lut.export_csv(output)
            self.assertIn("iteration_time_ns", output.read_text(encoding="utf-8"))

    def test_weighted_schedulable_range_changes_after_update(self) -> None:
        hardware, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        before = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in before), (0, 1, 2))
        graph.increase_path((0, 1))
        after = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in after), (0, 1))

    def test_decode_per_die_cost_and_config_order_tie(self) -> None:
        _, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        lut = FaceLut(
            (
                FaceLutEntry(2, 0, 0, 0, 0),
                FaceLutEntry(2, 0, 1, 256, 100),
                FaceLutEntry(2, 0, 2, 256, 300),
            )
        )
        selected, costs = select_decode_instance(
            topology=topology,
            graph=graph,
            lut=lut,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((256,), (), ()),
            new_request_token_length=256,
        )
        self.assertEqual(selected, 1)
        self.assertEqual([cost.instance_index for cost in costs], [0, 1, 2])
        self.assertGreater(costs[0].per_die_delta_ns, costs[1].per_die_delta_ns)
        self.assertEqual(costs[1].per_die_delta_ns, costs[2].per_die_delta_ns)

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

    def test_64_gib_fails_fast_for_the_exact_one_million_reserve(self) -> None:
        config = load_face_trace_config()
        hardware = FaceHardware(
            mesh_rows=1,
            mesh_cols=6,
            local_hbm_capacity_bytes=64 * 1024**3,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=1.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp6", "1", tuple(range(6))),),
        )
        with self.assertRaisesRegex(ValueError, r"relative_tp_rank=0"):
            SessionKVCacheManager(
                topology,
                config.model,
                reserve_context_tokens=1_000_000,
            )

    def test_manager_deletes_multiple_lru_victims_but_not_active_kv(self) -> None:
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = FaceHardware(1, 2, 200, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp2", "1", (0, 1)),),
        )
        manager = SessionKVCacheManager(topology, model, reserve_context_tokens=10)
        for session_id, completion_ns in (("a", 10), ("b", 20)):
            decision = manager.prepare_history(
                session_id,
                0,
                0,
                completion_ns,
                f"{session_id}0",
                required_context_tokens=10,
            )
            self.assertFalse(decision.admission_blocked)
            self.assertTrue(
                manager.grow_prefill(
                    session_id, 10, completion_ns, f"{session_id}0"
                ).admitted
            )
            manager.mark_complete(session_id, completion_ns, f"{session_id}0")

        c_decision = manager.prepare_history(
            "c", 0, 0, 30, "c0", required_context_tokens=30
        )
        self.assertEqual(
            tuple(record.victim_session_id for record in c_decision.evictions),
            ("a", "b"),
        )
        self.assertEqual(manager.session_snapshot("a").logical_context_tokens, 10)
        self.assertEqual(manager.session_snapshot("a").state, "EVICTED")
        self.assertTrue(manager.grow_prefill("c", 30, 30, "c0").admitted)
        deferred = manager.enforce_watermark(0, 30, "c0", ("c",))
        self.assertTrue(deferred.deferred)
        self.assertEqual(manager.session_snapshot("c").state, "RESIDENT")
        self.assertTrue(manager.session_snapshot("c").active)
        manager.mark_complete("c", 31, "c0")
        manager.assert_final_state()

    def test_session_policy_keeps_face_mapping_without_capacity_pressure(self) -> None:
        config = load_face_trace_config()
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        requests = tuple(
            FaceRequest(index, f"s{index}", 0, f"r{index}", 512, 2, 0, None)
            for index in range(4)
        )
        legacy = plan_face_requests(
            hardware=config.hardware,
            model=config.model,
            instance_specs=specs,
            requests=requests,
        )
        managed = plan_face_requests(
            hardware=config.hardware,
            model=config.model,
            instance_specs=specs,
            requests=requests,
            p_chunk=512,
            kv_cache_policy="session_lru_recompute",
            reserve_context_tokens=1_000_000,
            record_planning_iterations=False,
        )
        self.assertEqual(legacy.p_chunk, managed.p_chunk)
        self.assertEqual(
            [
                (item.prefill_instance_index, item.decode_instance_index)
                for item in legacy.requests
            ],
            [
                (item.prefill_instance_index, item.decode_instance_index)
                for item in managed.requests
            ],
        )

    def test_checked_in_shape_and_requests_plan_deterministically(self) -> None:
        config = load_face_trace_config()
        hardware = config.hardware
        model = config.model
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        requests = (
            FaceRequest(0, "session_0", 0, "session_0_request_0", 497, 42, 0, None),
            FaceRequest(
                1,
                "session_0",
                1,
                "session_0_request_1",
                494,
                58,
                None,
                1_000_000_000,
            ),
            FaceRequest(2, "session_1", 0, "session_1_request_0", 241, 57, 0, None),
            FaceRequest(
                3,
                "session_1",
                1,
                "session_1_request_1",
                152,
                46,
                None,
                1_000_000_000,
            ),
        )
        first = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=requests,
        )
        second = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=requests,
        )
        self.assertEqual(first.p_chunk, 346)
        self.assertEqual(len(first.requests), 4)
        self.assertTrue(first.iterations)
        signature = [
            (
                plan.request_id,
                plan.estimated_arrival_ns,
                plan.prefill_instance_index,
                plan.decode_instance_index,
                plan.completion_ns,
            )
            for plan in first.requests
        ]
        self.assertEqual(
            signature,
            [
                (
                    plan.request_id,
                    plan.estimated_arrival_ns,
                    plan.prefill_instance_index,
                    plan.decode_instance_index,
                    plan.completion_ns,
                )
                for plan in second.requests
            ],
        )
        by_id = {plan.request_id: plan for plan in first.requests}
        self.assertEqual(
            by_id["session_0_request_1"].estimated_arrival_ns,
            by_id["session_0_request_0"].completion_ns + 1_000_000_000,
        )
        self.assertEqual(
            by_id["session_1_request_1"].estimated_arrival_ns,
            by_id["session_1_request_0"].completion_ns + 1_000_000_000,
        )
        for plan in first.requests:
            self.assertGreaterEqual(plan.prefill_start_ns, plan.estimated_arrival_ns)
            self.assertGreater(plan.completion_ns, plan.prefill_complete_ns)
            self.assertTrue(plan.decode_candidates)
            self.assertIn(
                plan.decode_instance_index,
                [candidate.instance_index for candidate in plan.decode_candidates],
            )
            for candidate in plan.decode_candidates:
                self.assertLessEqual(
                    candidate.weighted_distance,
                    hardware.schedulable_distance_limit,
                )


if __name__ == "__main__":
    unittest.main()
