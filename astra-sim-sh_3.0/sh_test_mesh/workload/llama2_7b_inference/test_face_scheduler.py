#!/usr/bin/env python3
"""Focused tests for the trace-generation-time FACE scheduler."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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
    estimate_model_weight_bytes,
    estimate_prefill_task_load_ns,
    kv_cache_bytes_for_tokens,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
    manhattan_hops,
    model_weight_shard_bytes_by_tp_rank,
    nearest_edge_rank,
    physical_edge_ranks,
    plan_face_requests,
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
    main as generate_face_trace_main,
    load_face_trace_config,
    order_plans_for_static_emission,
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


# ---------------------------------------------------------------------------
# request-neutral 合成 fixture（裸仓库不物化输入；计数只反映 fixture 自身，
# 不依赖任何真实 trace 数据——方案 §3 步骤 0-1 / face/wscllm 收尾先例）。
# sh_3.0 = sidecar_restore 变体：合成队列之外还需 context sidecar（每请求
# prefix_tokens/input_tokens_total，满足 input_tokens_total ==
# prefix_tokens + prefill_length）。
# ---------------------------------------------------------------------------
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
# context sidecar：turn-0 prefix 常驻；turn>0 prefix = 该 session 上一请求
# 的 final context（prefill+decode）——与真实派生规则同构。
SYNTHETIC_CONTEXT_ROWS = (
    ("fixture_s0", "0", "fixture_s0_r0", "100", "200"),
    ("fixture_s1", "0", "fixture_s1_r0", "200", "400"),
    ("fixture_s2", "0", "fixture_s2_r0", "300", "600"),
    ("fixture_s3", "0", "fixture_s3_r0", "400", "800"),
    ("fixture_s4", "0", "fixture_s4_r0", "500", "1000"),
    ("fixture_s5", "0", "fixture_s5_r0", "600", "1200"),
    ("fixture_s6", "0", "fixture_s6_r0", "700", "1400"),
    ("fixture_s7", "0", "fixture_s7_r0", "800", "1600"),
    ("fixture_s0", "1", "fixture_s0_r1", "105", "155"),
    ("fixture_s1", "1", "fixture_s1_r1", "206", "266"),
)

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="sh30_test_fixture_")


def _write_synthetic_inputs() -> tuple[Path, Path]:
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    queue_path.write_text(
        "session_id,turn_index,request_id,prefill_length,decode_length,"
        "session_arrival_time_ns,inter_request_interval_ns,description\n"
        + "\n".join(",".join(row) for row in SYNTHETIC_QUEUE_ROWS)
        + "\n",
        encoding="utf-8",
    )
    context_path = Path(_FIXTURE_DIR.name) / "synthetic_request_context.csv"
    context_path.write_text(
        "session_id,turn_index,request_id,prefix_tokens,input_tokens_total\n"
        + "\n".join(",".join(row) for row in SYNTHETIC_CONTEXT_ROWS)
        + "\n",
        encoding="utf-8",
    )
    return queue_path, context_path


def load_checked_in_fixture_config() -> object:
    """request-neutral fixture：checked-in 配置 + 手写合成队列/上下文
    sidecar（一次性物化）。相对路径解析与 checked-in 配置一致；仅输入
    队列与 sidecar 槽位替换为合成数据。"""
    queue_path, context_path = _write_synthetic_inputs()
    config_path = Path(_FIXTURE_DIR.name) / "trace_config.csv"
    lines = []
    for raw_line in (MODULE_DIR / "trace_config.csv").read_text(
        encoding="utf-8"
    ).splitlines():
        if raw_line.startswith("config,request_queue_csv,"):
            lines.append(
                "config,request_queue_csv,"
                + str(queue_path)
                + ",,,,request-neutral synthetic fixture queue (unit test)"
            )
        elif raw_line.startswith("config,request_queue_context_csv,"):
            lines.append(
                "config,request_queue_context_csv,"
                + str(context_path)
                + ",,,,request-neutral synthetic fixture sidecar (unit test)"
            )
        else:
            lines.append(raw_line)
    if not config_path.exists():
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_face_trace_config(config_path)


def center_topology(capacity_bytes: int = 10_000) -> tuple[FaceHardware, object]:
    """3×3 mesh tiled as nine single-NPU instances.

    Only rank 4 (the mesh center) is off the boundary, so instance 4 is
    the sole edge-free instance; instances 0-3 and 5-8 each hold one
    boundary rank.
    """
    hardware = FaceHardware(
        mesh_rows=3,
        mesh_cols=3,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        tuple(
            FaceInstanceSpec(f"ins{rank}", str(rank + 1), (rank,))
            for rank in range(9)
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


    def test_prefill_work_derivation_matches_prefix_reuse_rules(self) -> None:
        requests = (
            RequestSpec("s0", 0, "r0", 10, 5, 0, None, 0, 10),
            RequestSpec("s0", 1, "r1", 4, 1, None, 100, 20, 24),
            RequestSpec("s0", 2, "r2", 4, 1, None, 100, 8, 12),
        )
        self.assertEqual(derive_prefill_work_tokens(requests), (10, 9, 4))

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
    ) -> None:
        before, transfer, evictions = manager.prepare_prefill(
            session_id=session_id,
            target_instance_index=instance_index,
            history_tokens=0,
            trigger_request_id=f"{session_id}_initial",
        )
        if before is not None or transfer is not None or evictions:
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
            manager.mark_complete(session_id, completion_ns)

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

    def test_exact_prefix_metadata_reuses_truncates_and_recomputes(self) -> None:
        hardware, model, topology, _ = self._tiny_kv_manager(
            capacity_bytes=10_000,
            reserve_context_tokens=1,
        )
        specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=(
                FaceRequest(
                    0, "session", 0, "r0", 5, 1, 0, None, 10, 15
                ),
                FaceRequest(
                    1, "session", 1, "r1", 3, 1, None, 1_000, 5, 8
                ),
                FaceRequest(
                    2, "session", 2, "r2", 2, 1, None, 1_000, 12, 14
                ),
            ),
            reserve_context_tokens=1,
            record_iterations=False,
        )
        first, second, third = plan.requests
        self.assertEqual(
            (first.history_tokens_before, first.prefill_context_tokens),
            (0, 15),
        )
        self.assertEqual(
            (second.history_tokens_before, second.prefill_context_tokens),
            (5, 8),
        )
        self.assertEqual(second.history_tokens_discarded, 11)
        self.assertEqual(
            (third.history_tokens_before, third.prefill_context_tokens),
            (9, 14),
        )
        self.assertEqual(third.history_tokens_discarded, 0)
        self.assertFalse(plan.iterations_recorded)
        self.assertEqual(plan.iterations, ())

    def test_static_emission_orders_kv_producer_before_store_trigger(self) -> None:
        store = SimpleNamespace(
            kind="remote_store",
            phase="completion",
            session_id="victim",
        )

        def request(
            request_id: str,
            session_id: str,
            turn_index: int,
            queue_index: int,
            prefill_start_ns: int,
            estimated_arrival_ns: int,
            completion_ns: int,
            completion_evictions: tuple[object, ...] = (),
        ) -> SimpleNamespace:
            return SimpleNamespace(
                request_id=request_id,
                session_id=session_id,
                turn_index=turn_index,
                queue_index=queue_index,
                prefill_start_ns=prefill_start_ns,
                estimated_arrival_ns=estimated_arrival_ns,
                prefill_complete_ns=prefill_start_ns + 10,
                completion_ns=completion_ns,
                history_evictions=(),
                prefill_evictions=(),
                decode_evictions=(),
                completion_evictions=completion_evictions,
            )

        trigger = request("trigger", "other", 0, 0, 0, 0, 300, (store,))
        producer = request("producer", "victim", 0, 1, 100, 100, 200)
        following = request("following", "victim", 1, 2, 310, 310, 320)
        ordered = order_plans_for_static_emission(
            SimpleNamespace(requests=(trigger, producer, following))
        )
        self.assertEqual(
            [item.request_id for item in ordered],
            ["producer", "trigger", "following"],
        )

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

    def test_checked_in_config_is_request_neutral_and_fails_closed_without_input(
        self,
    ) -> None:
        """checked-in 配置 = request-neutral（占位队列路径，不绑定任何
        默认 request 队列）+ 缺失输入 fail-closed + 配置字段语义（拓扑/
        模型/硬件；计数只反映合成 fixture，见方案 §3 步骤 0-1 收尾先例）。"""
        config = load_checked_in_fixture_config()
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
        self.assertEqual(config.request_queue_csv.name,
                         "synthetic_request_queue.csv")
        self.assertEqual(config.request_queue_context_csv.name,
                         "synthetic_request_context.csv")
        self.assertEqual(config.request_queue_session_limit, 0)
        self.assertEqual(config.trace_granularity, "request_aggregated")
        self.assertEqual(config.source_request_count, 10)
        self.assertEqual(config.source_session_count, 8)
        self.assertEqual(len(config.request_queue), 10)
        self.assertEqual(len(config.selected_session_ids), 8)
        self.assertEqual(
            config.selected_session_ids[:5],
            ("fixture_s0", "fixture_s1", "fixture_s2", "fixture_s3",
             "fixture_s4"),
        )
        self.assertEqual(
            config.selected_session_ids[-5:],
            ("fixture_s3", "fixture_s4", "fixture_s5", "fixture_s6",
             "fixture_s7"),
        )
        self.assertEqual(
            {request.session_id for request in config.request_queue},
            {f"fixture_s{index}" for index in range(8)},
        )
        self.assertTrue(
            all(request.prefix_tokens is not None for request in config.request_queue)
        )
        self.assertTrue(
            all(request.input_tokens_total is not None for request in config.request_queue)
        )
        self.assertTrue(
            all(
                request.input_tokens_total
                == request.prefix_tokens + request.prefill_length
                for request in config.request_queue
            )
        )
        first_request_indexes: dict[str, int] = {}
        for index, request in enumerate(config.request_queue):
            first_request_indexes.setdefault(request.session_id, index)
        self.assertEqual(len(first_request_indexes), 8)
        for index in first_request_indexes.values():
            request = config.request_queue[index]
            self.assertEqual(request.turn_index, 0)
            self.assertIsNotNone(request.session_arrival_time_ns)
        prefill_work_tokens = derive_prefill_work_tokens(config.request_queue)
        for index in first_request_indexes.values():
            self.assertEqual(
                prefill_work_tokens[index],
                config.request_queue[index].input_tokens_total,
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
            7.7,
        )
        self.assertEqual((min(prefill_lengths), max(prefill_lengths)), (50, 800))
        self.assertEqual((min(decode_lengths), max(decode_lengths)), (4, 12))
        self.assertEqual(len(arrival_times), 8)
        self.assertEqual(min(arrival_times), 0)
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
        self.assertEqual(max(request_arrivals.values()), 2000)
        self.assertLess(max(request_arrivals.values()), 10**9)
        # checked-in 配置本体 = 双态（backport re-verification 2026-08-16
        # 改为状态感知，与 sh_2.0 仓 skipUnless fixture 化同款语义）：
        #  - 裸仓库态：request_queue_csv 是占位路径（不物化任何默认队列），
        #    且占位路径不存在时正式入口 sys.exit(1)（fail-closed，不落随机
        #    stub 队列）；
        #  - 20.csv 前 30s 物化输入在位（traces/ 三件套，PROVENANCE 重建
        #    入口）：:12/:13 指向物化 queue + context sidecar，正式入口
        #    正常解析 1177 请求 / 112 session（sidecar_restore 生效）。
        checked_in_text = (MODULE_DIR / "trace_config.csv").read_text(
            encoding="utf-8")
        queue_line = next(
            line for line in checked_in_text.splitlines()
            if line.startswith("config,request_queue_csv,")
        )
        materialized_queue = (
            MODULE_DIR / "traces"
            / "astra_compute_20_first_30_seconds_request_queue.csv"
        )
        if materialized_queue.is_file():
            self.assertIn("traces/astra_compute_20_first_30_seconds_request_queue.csv", queue_line)
            materialized_config = load_face_trace_config(
                MODULE_DIR / "trace_config.csv")
            self.assertEqual(materialized_config.source_request_count, 1177)
            self.assertEqual(materialized_config.source_session_count, 112)
        else:
            self.assertIn("request_queue_placeholder.csv", queue_line)
            # fail-closed：占位路径不存在时正式入口 sys.exit(1)（不落随机
            # stub 队列）。
            with self.assertRaises(SystemExit) as caught:
                load_face_trace_config(MODULE_DIR / "trace_config.csv")
            self.assertNotEqual(caught.exception.code, 0)

    def test_llama2_7b_tp6_partition_is_exact_without_model_padding(self) -> None:
        config = load_checked_in_fixture_config()
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

        before, restore, restore_evictions = manager.prepare_prefill(
            session_id="second",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="restore_suffix",
        )
        self.assertEqual(before.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(restore_evictions, ())
        self.assertEqual(restore.kind, "remote_load")
        self.assertEqual((restore.layer_start, restore.layer_end), (2, 4))
        self.assertEqual(restore.total_bytes, before.remote_bytes)
        restored = manager.session_snapshot("second")
        self.assertEqual(restored.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(restored.resident_prefix_layers, 4)

    def test_odd_model_layer_count_derives_suffix_without_constants(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(layers=5)
        self.assertEqual(manager.partial_resident_prefix_layers, 3)

    def test_zero_token_truncation_normalizes_partial_cache_to_local(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=1,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
        )
        transfer = manager._evict_suffix(
            manager._sessions["session"],
            phase="completion",
            reason="test",
            trigger_request_id="r0",
        )
        self.assertEqual(transfer.kind, "remote_store")
        self.assertEqual(
            manager.session_snapshot("session").location,
            KVCacheManager.PARTIAL_HBM_REMOTE,
        )
        self.assertEqual(manager.truncate_history("session", 0), 10)
        snapshot = manager.session_snapshot("session")
        self.assertEqual(snapshot.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(snapshot.resident_prefix_layers, 4)
        self.assertEqual(snapshot.total_bytes, 0)

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
        before, local_hit, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="local_hit",
        )
        self.assertEqual(before.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(local_hit.kind, "local_hit")
        self.assertEqual(local_hit.shards, ())
        self.assertEqual(evictions, ())
        self.assertEqual(manager.hbm_snapshots(), hbm_before_hit)

        manager.mark_complete("session", 20)
        before, noc_transfer, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="noc_move",
        )
        self.assertEqual(before.instance_index, 0)
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

        before, remote_load, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="remote_load",
        )
        self.assertEqual(before.location, KVCacheManager.REMOTE_MEMORY)
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

    def test_plan_uses_hbm_aware_decode_tie_and_records_kv_state(self) -> None:
        hardware, model, topology, _ = self._tiny_kv_manager(
            capacity_bytes=200,
            reserve_context_tokens=80,
        )
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(
                    0,
                    "session",
                    0,
                    "request",
                    10,
                    1,
                    0,
                    None,
                ),
            ),
            reserve_context_tokens=80,
        )
        request = plan.requests[0]

        expected_prefill = select_prefill_instance(
            (
                InstanceTaskLoadSnapshot(0, 0, 0, 0, None),
                InstanceTaskLoadSnapshot(1, 0, 0, 0, None),
            ),
            (True, True),
        )
        self.assertEqual(request.prefill_instance_index, expected_prefill)
        # 2×2 玩具拓扑全边缘：首请求走 §7-B8 全集 fallback
        self.assertEqual(
            request.prefill_affinity_reason,
            "first_request_edge_fallback",
        )
        self.assertEqual(request.prefill_assignment_key, (0, -1, 0))
        self.assertEqual(
            [snapshot.total_task_load_ns for snapshot in request.prefill_instance_loads],
            [0, 0],
        )
        self.assertEqual(request.decode_instance_index, request.prefill_instance_index)
        self.assertEqual(request.decode_candidates, ())

        self.assertIsNone(request.history_location_before)
        self.assertIsNone(request.history_transfer)
        self.assertEqual(request.prefill_decode_transfer.kind, "local_hit")
        self.assertEqual(
            request.prefill_decode_transfer.reason,
            "prefill_decode_local_reuse",
        )
        self.assertEqual(len(request.completion_evictions), 1)
        self.assertEqual(request.completion_evictions[0].kind, "remote_store")
        self.assertEqual(
            sum(
                shard.bytes
                for shard in request.completion_evictions[0].shards
            ),
            request.completion_evictions[0].total_bytes,
        )
        self.assertEqual(
            request.kv_location_after_completion,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertIsNone(request.kv_instance_after_completion)
        self.assertEqual(request.reserve_unmet_ranks, ())
        self.assertEqual(len(request.hbm_after_completion), 4)
        for snapshot in request.hbm_after_completion:
            self.assertEqual(
                snapshot.used_bytes,
                snapshot.model_weight_bytes + snapshot.kv_cache_bytes,
            )
            self.assertEqual(
                snapshot.remaining_bytes,
                snapshot.capacity_bytes - snapshot.used_bytes,
            )
        self.assertEqual(
            plan.final_session_states[0].location,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertEqual(plan.reserve_context_tokens, 80)

    def test_plan_pins_partial_history_to_resident_prefix_instance(self) -> None:
        hardware, model, topology, _ = self._tiny_kv_manager(
            capacity_bytes=116,
            reserve_context_tokens=4,
            layers=4,
        )
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "session", 0, "turn0", 2, 1, 0, None),
                FaceRequest(1, "session", 1, "turn1", 2, 1, None, 1000),
            ),
            reserve_context_tokens=4,
        )
        first, second = plan.requests
        self.assertEqual(
            first.kv_location_after_completion,
            KVCacheManager.PARTIAL_HBM_REMOTE,
        )
        self.assertEqual(first.completion_evictions[0].layer_start, 2)
        self.assertEqual(second.history_location_before.location,
                         KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(second.prefill_affinity_reason, "resident_prefix_layers")
        self.assertEqual(
            second.prefill_instance_index,
            first.prefill_instance_index,
        )
        self.assertEqual(
            second.decode_instance_index,
            second.prefill_instance_index,
        )
        self.assertEqual(
            (second.history_transfer.layer_start,
             second.history_transfer.layer_end),
            (2, 4),
        )

    def test_default_config_uses_hbm160_edge_pool_and_one_million_reserve(self) -> None:
        config = load_checked_in_fixture_config()
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

    def test_et_kv_migration_ack_and_cross_instance_store_trigger(self) -> None:
        config = load_checked_in_fixture_config()
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

    def test_tp_readiness_barrier_waits_for_every_rank_local_predecessor(self) -> None:
        config = load_checked_in_fixture_config()
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

    def test_remote_suffix_load_emits_target_hbm_dma_and_branch_gate(self) -> None:
        config = load_checked_in_fixture_config()
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

    def test_plan_assigns_prefill_by_running_and_queued_roofline_load(self) -> None:
        hardware = FaceHardware(
            mesh_rows=2,
            mesh_cols=2,
            local_hbm_capacity_bytes=1_000_000_000,
            local_hbm_bandwidth_gbps=100.0,
            d2d_bandwidth_gbps=200.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        model = FaceModel(
            layers=2,
            hidden_size=16,
            ffn_size=32,
            num_heads=4,
            vocab_size=32,
            bytes_per_elem=2,
            mlp_variant="swiglu",
        )
        specs = (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=(
                FaceRequest(0, "s0", 0, "r0", 9000, 2, 0, None),
                FaceRequest(1, "s1", 0, "r1", 1000, 2, 1, None),
                FaceRequest(2, "s2", 0, "r2", 1000, 2, 2, None),
            ),
            reserve_context_tokens=1,
            average_decode_length=10.0,
        )
        first, second, third = plan.requests
        self.assertEqual(first.prefill_instance_index, 0)
        self.assertEqual(second.prefill_instance_index, 1)
        third_loads = third.prefill_instance_loads
        self.assertGreater(third_loads[0].running_prefill_task_load_ns, 0)
        self.assertGreater(third_loads[0].queued_prefill_task_load_ns, 0)
        self.assertGreater(third_loads[1].running_prefill_task_load_ns, 0)
        self.assertEqual(
            third.prefill_instance_index,
            min(third_loads, key=lambda load: load.ordering_key).instance_index,
        )

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

    def test_decode_per_die_cost_and_hbm_capacity_tie(self) -> None:
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
            remaining_hbm_capacity_bytes=(1_000, 100, 300),
            hbm_feasible_instances=(True, True, True),
        )
        self.assertEqual(selected, 2)
        self.assertEqual([cost.instance_index for cost in costs], [0, 1, 2])
        self.assertEqual(
            [cost.remaining_hbm_capacity_bytes for cost in costs],
            [1_000, 100, 300],
        )
        self.assertGreater(costs[0].per_die_delta_ns, costs[1].per_die_delta_ns)
        self.assertEqual(costs[1].per_die_delta_ns, costs[2].per_die_delta_ns)

        selected_equal_hbm, _ = select_decode_instance(
            topology=topology,
            graph=graph,
            lut=lut,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((256,), (), ()),
            new_request_token_length=256,
            remaining_hbm_capacity_bytes=(1_000, 300, 300),
            hbm_feasible_instances=(True, True, True),
        )
        self.assertEqual(selected_equal_hbm, 1)

        selected_with_capacity_filter, filtered_costs = select_decode_instance(
            topology=topology,
            graph=graph,
            lut=lut,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((256,), (), ()),
            new_request_token_length=256,
            remaining_hbm_capacity_bytes=(1_000, 100, 300),
            hbm_feasible_instances=(True, True, False),
        )
        self.assertEqual(selected_with_capacity_filter, 1)
        self.assertFalse(filtered_costs[2].hbm_feasible)

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

    def test_hbm_blocked_request_waits_until_active_session_completes(self) -> None:
        hardware = FaceHardware(
            mesh_rows=1,
            mesh_cols=2,
            local_hbm_capacity_bytes=60,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        model = FaceModel(
            layers=1,
            hidden_size=2,
            ffn_size=2,
            num_heads=2,
            vocab_size=2,
            bytes_per_elem=1,
            mlp_variant="gelu",
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=(FaceInstanceSpec("only", "1", (0, 1)),),
            requests=(
                FaceRequest(0, "active", 0, "active_0", 10, 5, 0, None),
                FaceRequest(1, "waiting", 0, "waiting_0", 10, 5, 0, None),
            ),
            reserve_context_tokens=0,
        )
        active, waiting = plan.requests
        self.assertEqual(active.admission_time_ns, 0)
        self.assertEqual(waiting.estimated_arrival_ns, 0)
        self.assertEqual(waiting.admission_time_ns, active.completion_ns)
        self.assertEqual(waiting.hbm_wait_ns, active.completion_ns)
        self.assertGreaterEqual(waiting.prefill_start_ns, waiting.admission_time_ns)
        self.assertTrue(
            any(
                transfer.kind == "remote_store"
                and transfer.session_id == "active"
                for transfer in waiting.history_evictions
            )
        )
        self.assertTrue(
            all(snapshot.remaining_bytes >= 0 for snapshot in plan.final_hbm_states)
        )

    def test_request_larger_than_empty_instance_does_not_wait_forever(self) -> None:
        hardware = FaceHardware(1, 2, 60, 1.0, 2.0, 1.0, 0, 0)
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        with self.assertRaisesRegex(ValueError, "cannot fit on any eligible empty"):
            plan_face_requests(
                hardware=hardware,
                model=model,
                instance_specs=(FaceInstanceSpec("only", "1", (0, 1)),),
                requests=(
                    FaceRequest(0, "oversized", 0, "oversized_0", 30, 1, 0, None),
                ),
                reserve_context_tokens=0,
            )

    def test_checked_in_shape_and_requests_plan_deterministically(self) -> None:
        config = load_checked_in_fixture_config()
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
        self.assertEqual(first.p_chunk, 512)
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
            self.assertEqual(plan.decode_candidates, ())
            self.assertEqual(plan.decode_instance_index, plan.prefill_instance_index)

    def test_first_request_prefill_excludes_edge_instances(self) -> None:
        hardware, topology = center_topology()
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        edge_ranks = set(physical_edge_ranks(hardware))
        self.assertEqual(edge_ranks, set(range(9)) - {4})

        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "session", 0, "r0", 10, 2, 0, None),
            ),
            reserve_context_tokens=40,
        )
        request = plan.requests[0]
        # 仅 instance 4（中心 rank）不含边缘 rank，首请求必须选中它
        self.assertEqual(request.prefill_instance_index, 4)
        self.assertEqual(request.prefill_affinity_reason, "first_request_non_edge")
        self.assertFalse(
            set(topology.instances[request.prefill_instance_index].ranks) & edge_ranks
        )
        self.assertEqual(request.decode_instance_index, request.prefill_instance_index)

    def test_local_hbm_history_sticks_to_resident_instance(self) -> None:
        hardware, topology = center_topology()
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "session", 0, "turn0", 10, 2, 0, None),
                FaceRequest(1, "session", 1, "turn1", 8, 2, None, 1000),
            ),
            reserve_context_tokens=40,
        )
        first, second = plan.requests
        self.assertEqual(first.prefill_affinity_reason, "first_request_non_edge")
        self.assertEqual(first.prefill_instance_index, 4)
        self.assertEqual(
            second.history_location_before.location,
            KVCacheManager.LOCAL_HBM,
        )
        # 全集负载均衡本会选中从未使用的 instance 0；sticky 必须覆盖它
        self.assertEqual(
            select_prefill_instance(
                second.prefill_instance_loads,
                (True,) * len(topology.instances),
            ),
            0,
        )
        self.assertEqual(second.prefill_instance_index, first.prefill_instance_index)
        self.assertEqual(second.prefill_affinity_reason, "resident_local_hbm")
        self.assertEqual(second.history_transfer.reason, "history_local_reuse")
        self.assertEqual(second.history_transfer_bytes, 0)
        self.assertTrue(
            all(
                transfer.reason != "history_other_instance"
                for transfer in second.history_evictions
            )
        )

    def test_remote_memory_history_uses_full_load_balance(self) -> None:
        hardware, topology = center_topology(capacity_bytes=200)
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "session", 0, "turn0", 10, 5, 0, None),
                FaceRequest(1, "session", 1, "turn1", 8, 5, None, 1000),
            ),
            reserve_context_tokens=80,
        )
        first, second = plan.requests
        self.assertEqual(first.prefill_instance_index, 4)
        self.assertEqual(
            first.kv_location_after_completion,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertEqual(
            second.history_location_before.location,
            KVCacheManager.REMOTE_MEMORY,
        )
        # REMOTE_MEMORY 走全集负载均衡：从未使用且下标最小的 instance 0
        # 含边缘 rank，证明远端命中可以落入边缘实例
        self.assertIsNone(second.prefill_affinity_reason)
        self.assertEqual(second.prefill_instance_index, 0)
        self.assertTrue(
            set(topology.instances[second.prefill_instance_index].ranks)
            & set(physical_edge_ranks(hardware))
        )
        self.assertEqual(
            second.prefill_instance_index,
            select_prefill_instance(
                second.prefill_instance_loads,
                second.prefill_hbm_feasible_instances,
            ),
        )
        self.assertEqual(second.history_transfer.reason, "history_remote_restore")

    def test_decode_matches_prefill_instance_and_empty_candidates(self) -> None:
        hardware, topology = center_topology()
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "session", 0, "turn0", 10, 2, 0, None),
                FaceRequest(1, "session", 1, "turn1", 8, 2, None, 1000),
            ),
            reserve_context_tokens=40,
        )
        for request in plan.requests:
            self.assertEqual(
                request.decode_instance_index,
                request.prefill_instance_index,
            )
            self.assertEqual(request.decode_candidates, ())
            self.assertEqual(
                request.prefill_decode_transfer.reason,
                "prefill_decode_local_reuse",
            )

    def test_first_request_waits_when_edge_free_instance_infeasible(self) -> None:
        # 单 rank 容量 120：恰好容纳一个 session（权重 40 + KV 60），
        # 两个并存需 160 —— 非边缘集合非空但暂不可行，必须等待而非回退。
        hardware, topology = center_topology(capacity_bytes=120)
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "active", 0, "active_0", 10, 5, 0, None),
                FaceRequest(1, "waiting", 0, "waiting_0", 10, 5, 0, None),
            ),
            reserve_context_tokens=0,
        )
        active, waiting = plan.requests
        self.assertEqual(active.prefill_instance_index, 4)
        self.assertEqual(active.prefill_affinity_reason, "first_request_non_edge")
        self.assertEqual(waiting.estimated_arrival_ns, 0)
        self.assertEqual(waiting.admission_time_ns, active.completion_ns)
        self.assertEqual(waiting.prefill_instance_index, 4)
        self.assertEqual(waiting.prefill_affinity_reason, "first_request_non_edge")
        self.assertEqual(waiting.decode_instance_index, waiting.prefill_instance_index)

    def test_first_request_edge_fallback_on_all_edge_topology(self) -> None:
        hardware, model, topology, _ = self._tiny_kv_manager(
            capacity_bytes=10_000,
            reserve_context_tokens=40,
        )
        instance_specs = tuple(
            FaceInstanceSpec(instance.name, instance.pg_name, instance.ranks)
            for instance in topology.instances
        )
        # 2×2 玩具拓扑全部 rank 均在边界：非边缘集合为空，走 §7-B8 全集 fallback
        self.assertEqual(set(physical_edge_ranks(hardware)), {0, 1, 2, 3})
        plan = plan_face_requests(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=(
                FaceRequest(0, "session", 0, "r0", 10, 2, 0, None),
            ),
            reserve_context_tokens=40,
        )
        request = plan.requests[0]
        self.assertEqual(
            request.prefill_affinity_reason,
            "first_request_edge_fallback",
        )
        self.assertEqual(
            request.prefill_instance_index,
            select_prefill_instance(
                request.prefill_instance_loads,
                (True, True),
            ),
        )
        self.assertEqual(request.decode_instance_index, request.prefill_instance_index)


if __name__ == "__main__":
    unittest.main()
