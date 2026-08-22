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
    estimate_iteration_time_ns,
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
# sh_3.0 = 折入 recompute 口径：turn-0 prefill_length 已折入 source prefix
#（== prefix + 新 token，全量重算），无 context sidecar。
# ---------------------------------------------------------------------------
# next_trigger_type（2026-08-18 类型感知逐出）：每 session 末行 "human"（无
# 后继，边界裁决）、其余 "tool"——与物化器同式。
SYNTHETIC_QUEUE_ROWS = (
    ("fixture_s0", "0", "fixture_s0_r0", "200", "5", "0", "", "tool", "synthetic fixture"),
    ("fixture_s1", "0", "fixture_s1_r0", "400", "6", "0", "", "tool", "synthetic fixture"),
    ("fixture_s2", "0", "fixture_s2_r0", "600", "7", "0", "", "human", "synthetic fixture"),
    ("fixture_s3", "0", "fixture_s3_r0", "800", "8", "0", "", "human", "synthetic fixture"),
    ("fixture_s4", "0", "fixture_s4_r0", "1000", "9", "0", "", "human", "synthetic fixture"),
    ("fixture_s5", "0", "fixture_s5_r0", "1200", "10", "0", "", "human", "synthetic fixture"),
    ("fixture_s6", "0", "fixture_s6_r0", "1400", "11", "0", "", "human", "synthetic fixture"),
    ("fixture_s7", "0", "fixture_s7_r0", "1600", "12", "0", "", "human", "synthetic fixture"),
    ("fixture_s0", "1", "fixture_s0_r1", "50", "4", "", "1000", "human", "synthetic fixture"),
    ("fixture_s1", "1", "fixture_s1_r1", "60", "5", "", "2000", "human", "synthetic fixture"),
)

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="sh30_test_fixture_")


def _write_synthetic_inputs() -> Path:
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    queue_path.write_text(
        "session_id,turn_index,request_id,prefill_length,decode_length,"
        "session_arrival_time_ns,inter_request_interval_ns,"
        "next_trigger_type,description\n"
        + "\n".join(",".join(row) for row in SYNTHETIC_QUEUE_ROWS)
        + "\n",
        encoding="utf-8",
    )
    return queue_path


def load_checked_in_fixture_config() -> object:
    """request-neutral fixture：checked-in 配置 + 手写合成折入队列
    （一次性物化）。相对路径解析与 checked-in 配置一致；仅输入队列
    槽位替换为合成数据。"""
    queue_path = _write_synthetic_inputs()
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
        # 折入 recompute 口径：turn-0 prefill_length 已含 prefix（全量重算，
        # work == prefix+新 token）；后续 turn work == 新 token（history 由
        # KV 账本驻留复用）。等价于旧 sidecar 口径 (10, 9, 4)。
        requests = (
            RequestSpec("s0", 0, "r0", 10, 5, 0, None),
            RequestSpec("s0", 1, "r1", 9, 1, None, 100),
            RequestSpec("s0", 2, "r2", 4, 1, None, 100),
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
        next_request_type: str | None = None,
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
        # 折入等价断言（sidecar 口径 input_tokens_total == prefix +
        # prefill_length 的折入形式）：turn-0 prefill_length 已折入 source
        # prefix（全量重算），turn>0 仅含新 token。
        self.assertEqual(
            {request.prefill_length for request in config.request_queue
             if request.turn_index == 0},
            {200, 400, 600, 800, 1000, 1200, 1400, 1600},
        )
        self.assertEqual(
            {request.prefill_length for request in config.request_queue
             if request.turn_index > 0},
            {50, 60},
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
            # turn-0 work == 折入全量 prefill_length（== prefix + 新 token）
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
            7.7,
        )
        self.assertEqual((min(prefill_lengths), max(prefill_lengths)), (50, 1600))
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
        #  - 20.csv 前 30s 物化输入在位（traces/ 折入 queue + canonical
        #    digest，重建入口 = traces/materialize_20_30s.py）：:12 指向物化
        #    折入 queue，正式入口正常解析 1177 请求 / 112 session（recompute
        #    折入口径生效）。
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

    def test_odd_model_layer_count_derives_suffix_without_constants(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(layers=5)
        self.assertEqual(manager.partial_resident_prefix_layers, 3)

    def test_suffix_eviction_leaves_partial_cache_resident(self) -> None:
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


    def test_roofline_estimate_keeps_exact_decode_token_length(self) -> None:
        hardware, _ = line_topology()
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        estimate = FaceRooflineEstimate(
            instance_size=2,
            p_chunk=0,
            d_batch=1,
            d_token=384,
            iteration_time_ns=estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=2,
                p_chunk=0,
                d_batch=1,
                d_token=384,
            ),
        )
        self.assertEqual(estimate.d_token, 384)
        self.assertEqual(
            estimate.iteration_time_ns,
            estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=2,
                p_chunk=0,
                d_batch=1,
                d_token=384,
            ),
        )

    def test_weighted_schedulable_range_changes_after_update(self) -> None:
        hardware, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        before = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in before), (0, 1, 2))
        graph.increase_path((0, 1))
        after = graph.schedulable_instances(0, hardware.schedulable_distance_limit)
        self.assertEqual(tuple(index for index, _ in after), (0, 1))

    def test_decode_per_die_cost_and_hbm_capacity_tie(self) -> None:
        hardware, topology = line_topology()
        graph = WeightedInstanceGraph(topology)
        model = FaceModel(1, 2, 2, 2, 2, 1, "gelu")
        selected, costs = select_decode_instance(
            hardware=hardware,
            model=model,
            topology=topology,
            graph=graph,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((256,), (), ()),
            new_request_token_length=256,
            remaining_hbm_capacity_bytes=(1_000, 100, 300),
            hbm_feasible_instances=(True, True, True),
        )
        # Exact Roofline replaces the old nearest-bin score: the occupied
        # candidate is now genuinely cheaper, and capacity remains secondary.
        self.assertEqual(selected, 0)
        self.assertEqual([cost.instance_index for cost in costs], [0, 1, 2])
        self.assertEqual(
            [cost.remaining_hbm_capacity_bytes for cost in costs],
            [1_000, 100, 300],
        )
        self.assertLess(costs[0].per_die_delta_ns, costs[1].per_die_delta_ns)
        self.assertEqual(costs[1].per_die_delta_ns, costs[2].per_die_delta_ns)
        self.assertEqual(costs[0].current_roofline.d_token, 256)
        self.assertEqual(costs[0].updated_roofline.d_token, 256)

        selected_equal_hbm, _ = select_decode_instance(
            hardware=hardware,
            model=model,
            topology=topology,
            graph=graph,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((), (), ()),
            new_request_token_length=256,
            remaining_hbm_capacity_bytes=(100, 300, 300),
            hbm_feasible_instances=(True, True, True),
        )
        self.assertEqual(selected_equal_hbm, 1)

        selected_with_capacity_filter, filtered_costs = select_decode_instance(
            hardware=hardware,
            model=model,
            topology=topology,
            graph=graph,
            fixed_p_chunk=64,
            prefill_instance_index=1,
            has_prefill_work=(False, False, False),
            decode_token_lengths=((), (), ()),
            new_request_token_length=256,
            remaining_hbm_capacity_bytes=(1_000, 100, 300),
            hbm_feasible_instances=(False, True, True),
        )
        self.assertEqual(selected_with_capacity_filter, 2)
        self.assertFalse(filtered_costs[0].hbm_feasible)

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
