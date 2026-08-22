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
    DecodeTieCounter,
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    FaceRooflineEstimate,
    KVAllocator,
    PrefillQueueSnapshot,
    WeightedInstanceGraph,
    build_instances,
    estimate_iteration_time_ns,
    estimate_model_weight_bytes,
    select_decode_instance,
    select_prefill_instance,
)
from session_kv_manager import (  # noqa: E402
    SessionKVCacheManager,
    attention_heads_by_tp_rank,
    kv_cache_shard_bytes_for_tokens,
    model_weight_shard_bytes_by_tp_rank,
)
from generate_face_trace import (  # noqa: E402
    _candidate_dict,
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


# ------------------------------------------------------------------------
# request-neutral 合成 fixture(收尾形态,比照 wscllm 裸仓库收尾):加载数据
# 时只用本文件内手写的合成队列(绝不引用任何真实 trace 数据),通过临时
# 配置副本把 request_queue_csv 指向合成队列后加载。
# ------------------------------------------------------------------------
import tempfile

REQUEST_QUEUE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)
SYNTHETIC_QUEUE_ROWS = (
    ("fixture_s0", "0", "fixture_s0_r0", "100", "5", "0", "", "synthetic fixture"),
    ("fixture_s1", "0", "fixture_s1_r0", "200", "6", "0", "", "synthetic fixture"),
    ("fixture_s2", "0", "fixture_s2_r0", "300", "7", "0", "", "synthetic fixture"),
    ("fixture_s3", "0", "fixture_s3_r0", "400", "8", "0", "", "synthetic fixture"),
    ("fixture_s4", "0", "fixture_s4_r0", "500", "9", "0", "", "synthetic fixture"),
    ("fixture_s5", "0", "fixture_s5_r0", "600", "10", "0", "", "synthetic fixture"),
    ("fixture_s6", "0", "fixture_s6_r0", "700", "11", "0", "", "synthetic fixture"),
    ("fixture_s7", "0", "fixture_s7_r0", "800", "12", "0", "", "synthetic fixture"),
    ("fixture_s0", "1", "fixture_s0_r1", "50", "4", "", "1000", "synthetic fixture"),
    ("fixture_s1", "1", "fixture_s1_r1", "60", "5", "", "2000", "synthetic fixture"),
)

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="face_test_fixture_")


def _write_synthetic_queue(path) -> None:
    path.write_text(
        REQUEST_QUEUE_HEADER + "\n"
        + "\n".join(",".join(row) for row in SYNTHETIC_QUEUE_ROWS) + "\n",
        encoding="utf-8",
    )


def load_checked_in_config() -> object:
    """request-neutral fixture:checked-in 配置 + 手写合成队列(一次性物化)。

    相对路径(hardware/、system/ 等)仍按 SH_TEST_DIR 解析,配置字段语义与
    checked-in 配置一致;仅输入队列槽位替换为合成数据(收尾形态:仓库不
    物化任何真实 request 队列,正式入口缺失输入 fail-closed)。
    """
    from generate_face_trace import load_face_trace_config
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    _write_synthetic_queue(queue_path)
    config_csv = Path(_FIXTURE_DIR.name) / "synthetic_trace_config.csv"
    lines = (Path(__file__).parent / "trace_config.csv").read_text(
        encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("config,request_queue_csv,"):
            out.append("config,request_queue_csv,{},,,,synthetic fixture\n".format(queue_path))
        else:
            out.append(line)
    config_csv.write_text("".join(out), encoding="utf-8")
    return load_face_trace_config(config_csv)



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
        config = load_checked_in_config()
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
        # 收尾形态(2026-08-16):合成 fixture 期望(8 session / 10 request,
        # 手写数据;仓库不物化真实队列,见 load_checked_in_config)。
        self.assertEqual(config.source_request_count, 10)
        self.assertEqual(config.source_session_count, 8)
        self.assertEqual(len(config.request_queue), 10)
        self.assertEqual(
            config.selected_session_ids,
            tuple("fixture_s{}".format(index) for index in range(8)),
        )
        self.assertEqual(
            {request.session_id for request in config.request_queue},
            set(config.selected_session_ids),
        )
        prefill_lengths = [
            request.prefill_length for request in config.request_queue
        ]
        decode_lengths = [request.decode_length for request in config.request_queue]
        self.assertEqual((min(prefill_lengths), max(prefill_lengths)), (50, 800))
        self.assertEqual((min(decode_lengths), max(decode_lengths)), (4, 12))

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
        config = load_checked_in_config()
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

    def test_roofline_estimate_uses_exact_tokens_and_validates_prefill_context(self) -> None:
        hardware, topology = line_topology()
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        self.assertGreater(
            estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=2,
                p_chunk=8,
                d_batch=1,
                d_token=257,
                p_context_tokens=64,
            ),
            0,
        )
        with self.assertRaisesRegex(ValueError, "p_context_tokens must be zero"):
            estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=2,
                p_chunk=0,
                d_batch=1,
                d_token=257,
                p_context_tokens=1,
            )
        with self.assertRaisesRegex(ValueError, "must include the current Prefill chunk"):
            estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=2,
                p_chunk=8,
                d_batch=1,
                d_token=257,
                p_context_tokens=7,
            )

        _, costs = select_decode_instance(
            topology=topology,
            graph=WeightedInstanceGraph(topology),
            hardware=hardware,
            model=model,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((257,), (), ()),
            new_request_token_length=385,
        )
        loaded = next(cost for cost in costs if cost.instance_index == 0)
        self.assertIsInstance(loaded.current_roofline, FaceRooflineEstimate)
        self.assertEqual(loaded.current_roofline.d_token, 257)
        self.assertEqual(loaded.updated_roofline.d_token, 385)
        serialized = _candidate_dict(loaded)
        self.assertEqual(
            set(serialized),
            {
                "instance_index",
                "weighted_distance",
                "current_roofline",
                "updated_roofline",
                "delta_time_ns",
                "per_die_delta_ns",
            },
        )

    def test_weighted_schedulable_range_changes_after_update(self) -> None:
        hardware, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        before = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in before), (0, 1, 2))
        graph.increase_path((0, 1))
        after = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in after), (0, 1))

    def test_decode_roofline_cost_uses_formula_and_config_order_tie(self) -> None:
        hardware, topology = line_topology()
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        graph = WeightedInstanceGraph(topology)
        selected, costs = select_decode_instance(
            topology=topology,
            graph=graph,
            hardware=hardware,
            model=model,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((256,), (), ()),
            new_request_token_length=256,
        )
        self.assertEqual([cost.instance_index for cost in costs], [0, 1, 2])
        expected = min(costs, key=lambda cost: (cost.per_die_delta_ns, cost.instance_index))
        self.assertEqual(selected, expected.instance_index)
        for cost in costs:
            self.assertEqual(
                cost.delta_time_ns,
                cost.updated_roofline.iteration_time_ns
                - cost.current_roofline.iteration_time_ns,
            )

    def _decode_tie_fixture(self):
        """所有候选工作负载相同，因此形成可重放的精确三方平局。"""
        hardware, topology = line_topology()
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        graph = WeightedInstanceGraph(topology)
        return hardware, model, topology, graph

    def test_decode_tie_round_robin_rotation(self) -> None:
        # 中-1 裁决(2026-08-20):真实平局下共享 counter 轮流取 tied 元素。
        hardware, model, topology, graph = self._decode_tie_fixture()
        counter = DecodeTieCounter()
        picks = []
        for _ in range(4):
            selected, _ = select_decode_instance(
                topology=topology,
                graph=graph,
                hardware=hardware,
                model=model,
                fixed_p_chunk=64,
                prefill_instance_index=1,
                has_prefill_work=(False, False, False),
                decode_token_lengths=((), (), ()),
                new_request_token_length=256,
                tie_counter=counter,
            )
            picks.append(selected)
        self.assertEqual(picks, [0, 1, 2, 0])

    def test_decode_tie_counter_not_advanced_on_unique_min(self) -> None:
        # 唯一可调度候选不触发计数器；随后恢复精确平局仍取 tied[0]。
        hardware, model, topology, graph = self._decode_tie_fixture()
        counter = DecodeTieCounter()
        graph.increase_path((0, 1), amount=2)
        graph.increase_path((1, 2), amount=2)
        selected, _ = select_decode_instance(
            topology=topology,
            graph=graph,
            hardware=hardware,
            model=model,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((), (), ()),
            new_request_token_length=256,
            tie_counter=counter,
        )
        self.assertEqual(selected, 1)
        self.assertEqual(counter.value, 0)
        graph.decrease_path((0, 1), amount=2)
        graph.decrease_path((1, 2), amount=2)
        selected, _ = select_decode_instance(
            topology=topology,
            graph=graph,
            hardware=hardware,
            model=model,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((), (), ()),
            new_request_token_length=256,
            tie_counter=counter,
        )
        self.assertEqual(selected, 0)
        self.assertEqual(counter.value, 1)

    def test_decode_tie_without_counter_keeps_config_order(self) -> None:
        # 不传 counter:单次调用完全旧行为,连调两次均取 tied[0]。
        hardware, model, topology, graph = self._decode_tie_fixture()
        picks = []
        for _ in range(2):
            selected, _ = select_decode_instance(
                topology=topology,
                graph=graph,
                hardware=hardware,
                model=model,
                fixed_p_chunk=64,
                prefill_instance_index=1,
                has_prefill_work=(False, False, False),
                decode_token_lengths=((), (), ()),
                new_request_token_length=256,
            )
            picks.append(selected)
        self.assertEqual(picks, [0, 0])

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
        config = load_checked_in_config()
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




if __name__ == "__main__":
    unittest.main()
