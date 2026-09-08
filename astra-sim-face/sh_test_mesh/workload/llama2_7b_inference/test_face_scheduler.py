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
    SH_TEST_DIR,
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


def _load_edited_config(tag, edit, *, source_name="trace_config.csv"):
    """加载经逐行编辑的 trace 配置副本(D1 loader 三态/坏值用例)。

    edit(line) 返回替换行,返回 None 表示删除该行;request_queue_csv 行
    无条件重定向到合成队列(与 load_checked_in_config 同款 fixture)。
    """
    from generate_face_trace import load_face_trace_config
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    _write_synthetic_queue(queue_path)
    config_csv = Path(_FIXTURE_DIR.name) / f"synthetic_trace_config_{tag}.csv"
    lines = (Path(__file__).parent / source_name).read_text(
        encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("config,request_queue_csv,"):
            out.append("config,request_queue_csv,{},,,,synthetic fixture\n".format(queue_path))
        else:
            edited = edit(line)
            if edited is not None:
                out.append(edited)
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

    def test_hardware_remote_memory_expansion_fails_closed_at_parse(self) -> None:
        """A.2 (2026-09-05): the remote memory backend was removed; a hardware
        config declaring any expansion must fail closed at parse time."""

        from config_resolver import load_hardware_config

        checked_in = json.loads(
            (SH_TEST_DIR / "hardware" / "face_case5_config_c.json").read_text(
                encoding="utf-8"
            )
        )
        expansion = dict(checked_in)
        expansion["remote-memory"] = {
            "memory-type": "PER_NPU_MEMORY_EXPANSION",
            "bandwidth-gbps": 512.0,
            "latency-ns": 100,
            "npu-selection": "mesh-boundary",
            "logical-pool": "unified-kv-cache-pool",
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "face_case5_expansion.json"
            source.write_text(json.dumps(expansion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NO_MEMORY_EXPANSION"):
                load_hardware_config(source, "validation-160gib")

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
        # Passive 语义手算:容量 400B/rank、权重 72B/rank -> 可用 328B/rank;
        # a/b 各 40 token(160B/rank),完成后剩 8B/rank;完成期零逐出。
        manager = SessionKVCacheManager(topology, model)
        for session_id, request_id, completion in (("a", "a0", 10), ("b", "b0", 20)):
            decision = manager.prepare_history(
                session_id, 0, 0, completion, request_id, required_context_tokens=40
            )
            self.assertFalse(decision.admission_blocked)
            self.assertTrue(manager.grow_prefill(session_id, 40, completion, request_id).admitted)
            self.assertEqual(manager.mark_complete(session_id, completion, request_id), ())
        snapshot_a = manager.session_snapshot("a")
        self.assertIsNotNone(snapshot_a)
        self.assertEqual(snapshot_a.state, "RESIDENT")
        self.assertEqual(snapshot_a.logical_context_tokens, 40)
        # 41 token 增量 4B/rank <= 余 8B/rank:同实例原地放行(LOCAL_HIT,
        # 零重算——被动形态下完成边界保留 KV)。
        recompute = manager.prepare_history("a", 0, 40, 30, "a1", required_context_tokens=41)
        self.assertEqual(recompute.action, "LOCAL_HIT")
        self.assertEqual(recompute.recompute_tokens, 0)

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

    def test_manager_deletes_multiple_lru_victims_but_not_active_kv(self) -> None:
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = FaceHardware(1, 2, 200, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp2", "1", (0, 1)),),
        )
        manager = SessionKVCacheManager(topology, model)
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

        candidate_calls = 0
        original_candidate_sessions = manager._candidate_sessions

        def snapshot_candidates(*args, **kwargs):
            nonlocal candidate_calls
            candidate_calls += 1
            return original_candidate_sessions(*args, **kwargs)

        manager._candidate_sessions = snapshot_candidates
        try:
            c_decision = manager.prepare_history(
                "c", 0, 0, 30, "c0", required_context_tokens=30
            )
        finally:
            manager._candidate_sessions = original_candidate_sessions

        # The admission needs two deletions, but one stage-local LRU snapshot.
        self.assertEqual(candidate_calls, 1)
        self.assertEqual(
            tuple(record.victim_session_id for record in c_decision.evictions),
            ("a", "b"),
        )
        self.assertEqual(
            [
                (event.event_type, event.session_id, event.phase,
                 event.reason, event.trigger_request_id)
                for event in manager.events
            ],
            [
                ("no_history", "a", "history", "window_first_request", "a0"),
                ("retain_complete", "a", "completion", "request_completed_keep_kv", "a0"),
                ("no_history", "b", "history", "window_first_request", "b0"),
                ("retain_complete", "b", "completion", "request_completed_keep_kv", "b0"),
                ("evict_delete", "a", "history", "history_and_prefill_admission", "c0"),
                ("evict_delete", "b", "history", "history_and_prefill_admission", "c0"),
                ("no_history", "c", "history", "window_first_request", "c0"),
            ],
        )
        self.assertEqual(
            tuple(snapshot.remaining_bytes for snapshot in manager.hbm_snapshots(0)),
            (128, 128),
        )
        self.assertEqual(manager.session_snapshot("a").logical_context_tokens, 10)
        self.assertEqual(manager.session_snapshot("a").state, "EVICTED")
        self.assertTrue(manager.grow_prefill("c", 30, 30, "c0").admitted)
        self.assertEqual(manager.session_snapshot("c").state, "RESIDENT")
        self.assertTrue(manager.session_snapshot("c").active)
        manager.mark_complete("c", 31, "c0")
        manager.assert_final_state()

    def test_terminal_retirement_releases_local_kv_and_fails_closed(self) -> None:
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = FaceHardware(1, 2, 200, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp2", "1", (0, 1)),),
        )
        manager = SessionKVCacheManager(topology, model)
        self.assertFalse(
            manager.prepare_history(
                "terminal", 0, 0, 10, "terminal_r0",
                required_context_tokens=10,
            ).admission_blocked
        )
        self.assertTrue(manager.grow_prefill("terminal", 10, 10, "terminal_r0").admitted)
        with self.assertRaisesRegex(RuntimeError, "inactive completed"):
            manager.retire_terminal_session("terminal", 10, "terminal_r0")

        manager.mark_complete("terminal", 10, "terminal_r0")
        completion_actions = manager.events
        self.assertEqual(
            manager.retire_terminal_session("terminal", 10, "terminal_r0"),
            0,
        )
        self.assertEqual(manager.events, completion_actions)
        self.assertEqual(manager.session_ids, ())
        self.assertIsNone(manager.session_snapshot("terminal"))
        self.assertTrue(
            all(snapshot.resident_kv_bytes == 0
                for snapshot in manager.hbm_snapshots())
        )
        with self.assertRaises(KeyError):
            manager.retire_terminal_session("terminal", 10, "terminal_r0")

    def test_terminal_retirement_bounds_many_single_turn_sessions(self) -> None:
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = FaceHardware(1, 2, 200, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp2", "1", (0, 1)),),
        )
        manager = SessionKVCacheManager(topology, model)
        for index in range(32):
            session_id = f"single_{index}"
            request_id = f"{session_id}_r0"
            self.assertFalse(
                manager.prepare_history(
                    session_id, 0, 0, index, request_id,
                    required_context_tokens=10,
                ).admission_blocked
            )
            self.assertTrue(manager.grow_prefill(session_id, 10, index, request_id).admitted)
            manager._pressure_event(
                now_ns=index,
                phase="admission",
                event_type="admission_blocked",
                reason="test_pressure_ownership",
                trigger_request_id=request_id,
                target_instance_index=0,
                before=(),
                after=(),
                insufficient_ranks=(),
            )
            self.assertIn(request_id, manager._pressure_event_keys_by_request)
            manager.mark_complete(session_id, index, request_id)
            self.assertNotIn(request_id, manager._pressure_event_keys_by_request)
            self.assertFalse(manager._pressure_event_keys)
            manager.retire_terminal_session(session_id, index, request_id)
            self.assertEqual(manager.session_ids, ())

    def test_pressure_event_dedup_is_request_scoped_and_retired(self) -> None:
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = FaceHardware(1, 2, 1_000, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp2", "1", (0, 1)),),
        )
        manager = SessionKVCacheManager(topology, model)
        requests = (("first", "first_r0"), ("second", "second_r0"))
        for index, (session_id, request_id) in enumerate(requests):
            self.assertFalse(
                manager.prepare_history(
                    session_id,
                    0,
                    0,
                    index,
                    request_id,
                    required_context_tokens=10,
                ).admission_blocked
            )
            self.assertTrue(
                manager.grow_prefill(session_id, 10, index, request_id).admitted
            )

        def record_pressure(request_id: str, now_ns: int) -> None:
            manager._pressure_event(
                now_ns=now_ns,
                phase="admission",
                event_type="admission_blocked",
                reason="test_request_scoped_pressure",
                trigger_request_id=request_id,
                target_instance_index=0,
                before=(),
                after=(),
                insufficient_ranks=(),
            )

        for now_ns in range(3):
            record_pressure("first_r0", now_ns)
        for now_ns in range(3, 6):
            record_pressure("second_r0", now_ns)

        pressure_events = [
            event
            for event in manager.events
            if event.reason in {
                "test_request_scoped_pressure",
                "retry_after_previous_capacity_block",
            }
        ]
        self.assertEqual(
            [(event.event_type, event.trigger_request_id) for event in pressure_events],
            [
                ("admission_blocked", "first_r0"),
                ("admission_retry", "first_r0"),
                ("admission_blocked", "second_r0"),
                ("admission_retry", "second_r0"),
            ],
        )
        self.assertEqual(
            set(manager._pressure_event_keys_by_request),
            {"first_r0", "second_r0"},
        )
        self.assertEqual(
            manager._pressure_event_keys,
            {
                ("admission_blocked", "admission", "first_r0", 0),
                ("admission_retry", "admission", "first_r0", 0),
                ("admission_blocked", "admission", "second_r0", 0),
                ("admission_retry", "admission", "second_r0", 0),
            },
        )

        manager.mark_complete("first", 10, "first_r0")
        manager.retire_terminal_session("first", 10, "first_r0")
        self.assertEqual(set(manager._pressure_event_keys_by_request), {"second_r0"})
        self.assertEqual(
            manager._pressure_event_keys,
            {
                ("admission_blocked", "admission", "second_r0", 0),
                ("admission_retry", "admission", "second_r0", 0),
            },
        )

        manager.mark_complete("second", 11, "second_r0")
        manager.retire_terminal_session("second", 11, "second_r0")
        self.assertFalse(manager._pressure_event_keys)
        self.assertFalse(manager._pressure_event_keys_by_request)
        manager.assert_final_state()


class KvEvictionModeConfigTests(unittest.TestCase):
    """R3 (2026-09-05): kv_eviction_mode 门控随 reserve 档一并物理清除——
    loader 对该键恢复"未知键 fail-closed 拒绝"(回归钉子,防口子重开)。"""

    def test_kv_eviction_mode_key_rejected(self) -> None:
        def edit(line: str):
            if line.startswith("config,kv_reserve_context_tokens,"):
                return (
                    line
                    + "config,kv_eviction_mode,passive_only,,,,synthetic fixture\n"
                )
            return line

        with self.assertRaisesRegex(
            ValueError, "unsupported config key.*kv_eviction_mode"
        ):
            _load_edited_config("eviction_mode_row", edit)


class PassiveEvictionModeTests(unittest.TestCase):
    """R7 (2026-09-05): 被动逐出语义回归——完成边界零逐出、准入按需逐到
    刚好够、深缺口两出口台账(主动水位逐出已物理清除)。

    算术口径(FaceModel(1,4,4,2,4,1,"gelu") + tp2 + 容量 200):每 rank
    模型权重 72 字节 -> 空 rank 每 token KV 4 字节、初始余 128 字节/rank。
    """

    @staticmethod
    def _manager():
        model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = FaceHardware(1, 2, 200, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (FaceInstanceSpec("tp2", "1", (0, 1)),),
        )
        return SessionKVCacheManager(topology, model)

    def _admit_and_complete(self, manager, session_id, request_id, now_ns,
                            tokens=10):
        decision = manager.prepare_history(
            session_id, 0, 0, now_ns, request_id, required_context_tokens=tokens
        )
        self.assertFalse(decision.admission_blocked)
        self.assertTrue(
            manager.grow_prefill(session_id, tokens, now_ns, request_id).admitted
        )
        return manager.mark_complete(session_id, now_ns, request_id)

    def test_passive_completion_retains_kv_and_final_state_passes(self) -> None:
        manager = self._manager()  # 两会话各 10 token -> 40B/rank,余 48B/rank
        self.assertEqual(self._admit_and_complete(manager, "a", "a0", 10), ())
        self.assertEqual(self._admit_and_complete(manager, "b", "b0", 20), ())

        # 完成边界零逐出——retain_complete 事件在场,会话 RESIDENT。
        event_types = [
            (event.event_type, event.session_id) for event in manager.events
        ]
        self.assertIn(("retain_complete", "a"), event_types)
        self.assertIn(("retain_complete", "b"), event_types)
        self.assertFalse(
            [event for event in manager.events if event.event_type == "evict_delete"]
        )
        for session_id, completion_ns in (("a", 10), ("b", 20)):
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.state, "RESIDENT")
            self.assertFalse(snapshot.active)
            self.assertEqual(snapshot.last_completion_ns, completion_ns)
        # 两会话各 40 字节/rank -> 余 48:低余量终态合法(完成期不存在
        # 逐出路径,mark_complete 恒返回空元组)。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (48, 48),
        )
        manager.assert_final_state()

        manager.retire_terminal_session("a", 10, "a0")
        manager.retire_terminal_session("b", 20, "b0")
        manager.assert_final_state()

    def test_passive_admission_evicts_exactly_enough(self) -> None:
        manager = self._manager()  # a/b 各 10 token;a+b 完成后余 48B/rank
        self.assertEqual(self._admit_and_complete(manager, "a", "a0", 10), ())
        self.assertEqual(self._admit_and_complete(manager, "b", "b0", 20), ())

        # c 需 80B/rank(20 token):仅逐最老 a(+40 -> 88)即够,b 不动。
        decision = manager.prepare_history(
            "c", 0, 0, 30, "c0", required_context_tokens=20
        )
        self.assertFalse(decision.admission_blocked)
        self.assertEqual(
            tuple(record.victim_session_id for record in decision.evictions),
            ("a",),
        )
        # 逐出来自按需 fit 路径(reason 钉死来源)。
        self.assertEqual(
            decision.evictions[0].reason, "history_and_prefill_admission"
        )
        self.assertEqual(manager.session_snapshot("b").state, "RESIDENT")
        self.assertTrue(manager.grow_prefill("c", 20, 30, "c0").admitted)
        # 逐到刚好够即停:不追加额外逐出(b 保留,余 8B/rank)。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (8, 8),
        )
        self.assertEqual(manager.mark_complete("c", 40, "c0"), ())
        manager.assert_final_state()

    def test_deep_gap_records_structural_infeasibility_before_failing(self) -> None:
        manager = self._manager()
        with self.assertRaisesRegex(ValueError, "request cannot fit an empty instance"):
            manager.ensure_physical_fit(0, (280, 280), 5, "r9")
        self.assertEqual(manager.deep_gap_events, 1)
        deep_gaps = [
            event for event in manager.events if event.event_type == "deep_gap"
        ]
        self.assertEqual(len(deep_gaps), 1)
        self.assertEqual(
            deep_gaps[0].reason, "request_exceeds_empty_instance"
        )
        self.assertEqual(deep_gaps[0].insufficient_ranks, (0, 1))
        # fail-closed raise 不留部分变异。
        self.assertEqual(manager.session_ids, ())
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (128, 128),
        )

    def test_deep_gap_records_exhausted_candidates_on_admission_block(self) -> None:
        manager = self._manager()
        # z 保持 ACTIVE(不可逐,占 40B/rank);a/b 完成后共占 80B/rank -> 余 8。
        self.assertFalse(
            manager.prepare_history(
                "z", 0, 0, 5, "z0", required_context_tokens=10
            ).admission_blocked
        )
        self.assertTrue(manager.grow_prefill("z", 10, 5, "z0").admitted)
        self.assertEqual(self._admit_and_complete(manager, "a", "a0", 10), ())
        self.assertEqual(self._admit_and_complete(manager, "b", "b0", 20), ())

        # c 需 120B/rank(30 token):逐光 a,b(->88)仍不够 -> 优雅推迟。
        decision = manager.prepare_history(
            "c", 0, 0, 30, "c0", required_context_tokens=30
        )
        self.assertTrue(decision.admission_blocked)
        self.assertEqual(
            tuple(record.victim_session_id for record in decision.evictions),
            ("a", "b"),
        )
        self.assertEqual(decision.insufficient_ranks, (0, 1))
        self.assertEqual(manager.deep_gap_events, 1)
        deep_gaps = [
            event for event in manager.events if event.event_type == "deep_gap"
        ]
        self.assertEqual(len(deep_gaps), 1)
        self.assertEqual(
            deep_gaps[0].reason, "exhausted_completed_candidates"
        )
        self.assertEqual(manager.session_snapshot("a").state, "EVICTED")
        self.assertEqual(manager.session_snapshot("b").state, "EVICTED")
        self.assertEqual(manager.session_snapshot("z").state, "RESIDENT")
        self.assertTrue(manager.session_snapshot("z").active)
        # 台账守恒:仅剩 z 的 40/rank -> 余 88。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (88, 88),
        )
        # 收尾:清掉被阻塞请求的去重键后终态检查通过。
        manager._forget_pressure_event_keys("c0")
        self.assertEqual(manager.mark_complete("z", 40, "z0"), ())
        for session_id, now_ns, request_id in (
            ("a", 10, "a0"),
            ("b", 20, "b0"),
            ("z", 40, "z0"),
        ):
            manager.retire_terminal_session(session_id, now_ns, request_id)
        self.assertEqual(manager.session_ids, ())
        manager.assert_final_state()




if __name__ == "__main__":
    unittest.main()
