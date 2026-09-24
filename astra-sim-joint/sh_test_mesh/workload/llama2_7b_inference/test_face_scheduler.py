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
    KVCapacityError,
    KVCacheManager,
    KVTransfer,
    KVTransferShard,
    KV_DELTA_JOURNAL_FIELDS,
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
            KVCacheManager(topology, model),
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
        before, transfers, evictions = manager.prepare_prefill(
            session_id=session_id,
            target_instance_index=instance_index,
            history_tokens=0,
            trigger_request_id=f"{session_id}_initial",
        )
        if before is not None or transfers or evictions:
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

    def test_passive_only_completion_short_circuits_but_records_type(self) -> None:
        # D-clear (2026-09-05): 完成边界零逐出（主动驱逐已物理删除），
        # mark_complete 的 inactive + last_completion_ns + next_request_type
        # 记录无条件保留。
        # 手算口径（capacity 200、权重 20/rank、2 bytes/token/rank）：
        # 3 个 10-token 已完成会话 = 60 B + 活跃 25-token = 50 B → 每 rank
        # 剩余 70；完成路径不做任何逐出，全部会话保持 LOCAL_HBM。
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
        )
        for session_id, completion_ns in (
            ("session_a", 10),
            ("session_b", 20),
            ("session_c", 30),
        ):
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=10,
                completion_ns=completion_ns,
            )
        self._seed_local_session(
            manager,
            session_id="active_trigger",
            instance_index=0,
            context_tokens=25,
            completion_ns=None,
        )

        manager.mark_complete(
            "active_trigger", 50, next_request_type="tool"
        )
        snapshot = manager.session_snapshot("active_trigger")
        self.assertEqual(snapshot.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(snapshot.instance_index, 0)
        self.assertFalse(snapshot.active)
        self.assertEqual(snapshot.last_completion_ns, 50)
        # D3：类型记录保留（准入期四层类型序的输入）。
        self.assertEqual(
            manager._sessions["active_trigger"].next_request_type, "tool")
        for session_id in ("session_a", "session_b", "session_c"):
            self.assertEqual(
                manager.session_snapshot(session_id).location,
                KVCacheManager.LOCAL_HBM,
            )

    def test_passive_only_admission_pressure_evicts_exactly_enough(self) -> None:
        # D4（主动驱逐退役后，D-clear 2026-09-05）：唯一逐出路径是准入期
        # _ensure_capacity——逐到恰好够即停（FIFO 队头先逐，最年轻不动）。
        # 手算口径：3 个 10-token 已完成会话占 60 B/rank → 剩余 120；
        # 新会话增长 delta = 2*65 = 130 > 120 → 恰好只需逐出队头
        # session_old（20 B/rank）→ 140 >= 130。
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
        )
        for session_id, completion_ns in (
            ("session_old", 10),
            ("session_mid", 20),
            ("session_young", 30),
        ):
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=10,
                completion_ns=completion_ns,
            )
        self._seed_local_session(
            manager,
            session_id="newcomer",
            instance_index=0,
            context_tokens=0,
            completion_ns=None,
        )

        evictions = manager.expand_prefill(
            session_id="newcomer",
            instance_index=0,
            context_tokens=65,
            trigger_request_id="admission_pressure",
        )
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].session_id, "session_old")
        self.assertEqual(evictions[0].kind, "remote_store")
        self.assertEqual(
            manager.session_snapshot("session_old").location,
            KVCacheManager.REMOTE_MEMORY,
        )
        for session_id in ("session_mid", "session_young"):
            untouched = manager.session_snapshot(session_id)
            self.assertEqual(untouched.location, KVCacheManager.LOCAL_HBM)
            self.assertEqual(untouched.instance_index, 0)
        newcomer = manager.session_snapshot("newcomer")
        self.assertEqual(newcomer.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(newcomer.context_tokens, 65)
        # 恰好够：每 rank 剩余 = 200 - 20(权重) - (20+20+130) = 10。
        self.assertEqual(
            tuple(snapshot.remaining_bytes for snapshot in manager.hbm_snapshots(0)),
            (10, 10),
        )
        self.assertEqual(manager.deep_gap_events, [])

    def test_admission_pressure_keeps_two_stage_sequence(self) -> None:
        # 主线两阶段逐出顺序回归（迁自 test_ablation_switch.py 的 none 档
        # 对照用例）：容量压力下半层化先行、耗尽后才整体外迁。
        # 手算口径：4 层模型每 token 每 rank 8B（全层）；权重 68B/rank；
        # capacity 300 → 种子后余 32B；active 5→20 增量 120B：
        # suffix(oldest)+40=72 <120 → suffix(second)+40=112 <120 →
        # full(oldest 前缀 0..2)+40=152 ≥120 恰好停。
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=300, layers=4)
        self._seed_local_session(
            manager, session_id="oldest", instance_index=0,
            context_tokens=10, completion_ns=10,
            next_request_type="human")
        self._seed_local_session(
            manager, session_id="second", instance_index=0,
            context_tokens=10, completion_ns=20,
            next_request_type="human")
        self._seed_local_session(
            manager, session_id="active", instance_index=0,
            context_tokens=5, completion_ns=None)

        evictions = manager.expand_prefill(
            session_id="active", instance_index=0,
            context_tokens=20, trigger_request_id="grow")

        self.assertEqual(
            [(t.session_id, t.layer_start, t.layer_end,
              t.resident_prefix_layers_after)
             for t in evictions],
            [("oldest", 2, 4, 2), ("second", 2, 4, 2),
             ("oldest", 0, 2, 0)],
        )
        self.assertEqual(
            manager.session_snapshot("oldest").location,
            KVCacheManager.REMOTE_MEMORY)
        self.assertEqual(
            manager.session_snapshot("second").location,
            KVCacheManager.PARTIAL_HBM_REMOTE)

    def test_ensure_capacity_records_deep_gap_events_before_failing(self) -> None:
        # D4 (2026-09-05) + K6 (2026-09-14 kimi 复审)：逐无可循且结构不可行
        # （需求 > 空实例容量）→ fail-closed raise 携带逐 rank 缺口记录
        # （deep_gap_records）；raise 时**不落** deep_gap_events 台账（joint
        # 下容量类失败有多条可恢复捕获路径，落账 = 确认终态时经
        # commit_deep_gap_records 提交）。手算：空实例每 rank 剩余 =
        # 200 - 20(权重) = 180 < 400。
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
        )
        with self.assertRaises(ValueError) as ctx:
            manager._ensure_capacity(
                0,
                (400, 400),
                phase="prefill",
                reason="deep_gap_probe",
                trigger_request_id="deep_gap_trigger",
            )
        self.assertIsInstance(ctx.exception, KVCapacityError)
        records = ctx.exception.deep_gap_records
        self.assertEqual(len(records), 2)
        self.assertEqual([record["rank"] for record in records], [0, 1])
        for record in records:
            self.assertEqual(record["instance_index"], 0)
            self.assertEqual(record["phase"], "prefill")
            self.assertEqual(record["reason"], "deep_gap_probe")
            self.assertEqual(record["trigger_request_id"], "deep_gap_trigger")
            self.assertEqual(record["remaining_bytes"], 180)
            self.assertEqual(record["required_bytes"], 400)
            self.assertEqual(record["gap_bytes"], 220)
        # 可恢复路径不落账；确认终态提交后台账与旧口径逐字段一致。
        self.assertEqual(manager.deep_gap_events, [])
        manager.commit_deep_gap_records(records)
        self.assertEqual(len(manager.deep_gap_events), 2)
        self.assertEqual(
            [event["rank"] for event in manager.deep_gap_events],
            [0, 1],
        )

    def test_mark_complete_records_next_request_type(self) -> None:
        # The type passed to mark_complete lands on the session state; it is
        # the classification consumed by the typed eviction order at the next
        # admission boundary. Sessions with no recorded type default to the
        # human class (user ruling).
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=300,
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

        with self.assertRaises(ValueError):
            manager.mark_complete("active", 99, next_request_type="voice")

    def test_odd_model_layer_count_derives_suffix_without_constants(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(layers=5)
        self.assertEqual(manager.partial_resident_prefix_layers, 3)

    def test_suffix_eviction_leaves_partial_cache_resident(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
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
        )
        self._seed_local_session(
            manager,
            session_id="session",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
        )

        hbm_before_hit = manager.hbm_snapshots()
        before, transfers, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="local_hit",
        )
        local_hit = transfers[0]
        self.assertEqual(before.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(local_hit.kind, "local_hit")
        self.assertEqual(local_hit.shards, ())
        self.assertEqual(evictions, ())
        self.assertEqual(manager.hbm_snapshots(), hbm_before_hit)

        manager.mark_complete("session", 20)
        # joint（§2.2）：跨实例 = copy 工作副本——基础历史保留 home 实例
        # 0，执行端 1 持工作副本；merge_back v2 零字节翻转后权威驻留
        # 翻转到执行端（少并多，home := exec）。
        before, transfers, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="noc_copy",
            action="copy",
        )
        noc_transfer = transfers[0]
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
        # 双持有：home 基础副本（ranks 0/1）+ 执行端工作副本（ranks 2/3）
        # （每 rank KV = 10 token × 2 头 × 2 (K+V) × 2 elem / 2 rank = 20B）。
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (20, 20, 20, 20),
        )
        working = manager.session_snapshot("session")
        self.assertEqual(working.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(working.instance_index, 1)
        self.assertEqual(working.working_kind, "copy")
        self.assertEqual(working.home_instance, 0)
        # compute_done 后合并 v2（少并多，2026-09-17）：copy 的 exec 侧
        # 恒持并集 ⊇ home 侧 → 零字节翻转——无传输、home 侧基础释放、
        # home 迁移到 exec；无双份驻留。
        merge_transfers = manager.merge_back(
            session_id="session",
            trigger_request_id="noc_copy",
            new_tokens=0,
        )
        self.assertEqual(merge_transfers, ())
        merged = manager.session_snapshot("session")
        self.assertEqual(merged.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(merged.instance_index, 1)
        self.assertIsNone(merged.working_kind)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 20, 20),
        )
        self.assertEqual(
            manager.last_merge_outcome,
            {
                "session_id": "session",
                "direction": "reverse",
                "zero_byte_flip": True,
                "winner_instance": 1,
                "loser_instance": 0,
                "transferred_bytes": 0,
                "home_flipped": True,
            },
        )

        manager.mark_complete("session", 30)
        # D-clear (2026-09-05): 完成边界主动驱逐已删除，此处用保留的
        # _evict_session 守卫（fixture 手法，同 terminal-retirement 用例）
        # 造出 REMOTE_MEMORY 形态供 remote_load 段消费。
        store = manager._evict_session(
            manager._sessions["session"],
            phase="completion",
            reason="fixture_remote",
            trigger_request_id="remote_store",
        )
        self.assertEqual(store.kind, "remote_store")
        self.assertTrue(
            all(shard.edge_rank in manager.edge_ranks for shard in store.shards)
        )
        remote = manager.session_snapshot("session")
        self.assertEqual(remote.location, KVCacheManager.REMOTE_MEMORY)
        self.assertIsNone(remote.instance_index)
        self.assertEqual(remote.rank_bytes, ())
        self.assertEqual(
            tuple(snapshot.kv_cache_bytes for snapshot in manager.hbm_snapshots()),
            (0, 0, 0, 0),
        )

        before, transfers, evictions = manager.prepare_prefill(
            session_id="session",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="remote_load",
            action="copy",
        )
        remote_load = transfers[0]
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
        # P11 死键清除回归钉（2026-09-23）：local-mem-capacity-bytes
        # 写入链已退役（C++ 零读者），system.json 不再含该键。
        self.assertNotIn("local-mem-capacity-bytes", system_raw)
        self.assertEqual(system_raw["remote-mem-bw"], 512)
        self.assertEqual(system_raw["remote-mem-latency"], 100)
        self.assertEqual(system_raw["peak-perf"], 261.12)
        self.assertEqual(system_raw["hbm-kv-restore-bandwidth-sharing"], 1)

    def test_kv_eviction_mode_key_rejected(self) -> None:
        # R3 fail-closed 回归钉子（D-clear 2026-09-05）：kv_eviction_mode
        # 门控整体退役后，残留该键的配置必须被拒绝启动而非静默忽略。
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trace_config.csv"
            source.write_text(
                "\n".join(
                    (
                        "kind,key,value,group_name,pg_name,ranks,description",
                        "config,kv_eviction_mode,passive_only,,,,retired gate",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported config key"):
                load_face_trace_config(source)

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

    def test_terminal_retirement_releases_local_partial_and_remote_kv(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            layers=4,
        )
        self._seed_local_session(
            manager,
            session_id="active",
            instance_index=0,
            context_tokens=10,
            completion_ns=None,
        )
        with self.assertRaisesRegex(RuntimeError, "inactive completed"):
            manager.retire_terminal_session("active", 10, "active_r0")
        manager.mark_complete("active", 10)
        self.assertEqual(
            manager.retire_terminal_session("active", 10, "active_r0"), 0
        )
        self.assertEqual(manager.session_ids, ())
        with self.assertRaises(KeyError):
            manager.retire_terminal_session("active", 10, "active_r0")

        _, _, _, partial_manager = self._tiny_kv_manager(
            capacity_bytes=200,
            layers=4,
        )
        self._seed_local_session(
            partial_manager,
            session_id="partial",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
        )
        partial_manager._evict_suffix(
            partial_manager._sessions["partial"],
            phase="completion",
            reason="fixture_partial",
            trigger_request_id="partial_r0",
        )
        self.assertEqual(
            partial_manager.session_snapshot("partial").location,
            KVCacheManager.PARTIAL_HBM_REMOTE,
        )
        self.assertEqual(
            partial_manager.retire_terminal_session("partial", 10, "partial_r0"),
            0,
        )
        self.assertTrue(
            all(snapshot.kv_cache_bytes == 0
                for snapshot in partial_manager.hbm_snapshots())
        )

        _, _, _, remote_manager = self._tiny_kv_manager(
            capacity_bytes=200,
            layers=4,
        )
        self._seed_local_session(
            remote_manager,
            session_id="remote",
            instance_index=0,
            context_tokens=10,
            completion_ns=10,
        )
        remote_manager._evict_session(
            remote_manager._sessions["remote"],
            phase="completion",
            reason="fixture_remote",
            trigger_request_id="remote_r0",
        )
        before = remote_manager.hbm_snapshots()
        self.assertIsNone(
            remote_manager.retire_terminal_session("remote", 10, "remote_r0")
        )
        self.assertEqual(remote_manager.session_ids, ())
        self.assertEqual(remote_manager.hbm_snapshots(), before)

    def test_terminal_retirement_bounds_many_single_turn_sessions(self) -> None:
        _, _, _, manager = self._tiny_kv_manager(
            capacity_bytes=200,
            layers=4,
        )
        for index in range(32):
            session_id = f"single_{index}"
            self._seed_local_session(
                manager,
                session_id=session_id,
                instance_index=0,
                context_tokens=10,
                completion_ns=None,
            )
            manager.mark_complete(session_id, index)
            manager.retire_terminal_session(session_id, index, f"{session_id}_r0")
            self.assertEqual(manager.session_ids, ())

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













def _merge_v2_manager(
    *,
    capacity_bytes: int = 2000,
    layers: int = 4,
) -> tuple[FaceModel, KVCacheManager]:
    """merge v2 真值表 fixture：2×2 mesh、双 2-rank 实例。

    每层每 token 每 rank 4B（全层 16B/token/rank），逐字节可手算。
    """
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
        hidden_size=4,
        ffn_size=4,
        num_heads=2,
        vocab_size=4,
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
    return model, KVCacheManager(topology, model)


def _seed_completed_session(
    manager: KVCacheManager,
    *,
    session_id: str,
    instance_index: int,
    context_tokens: int,
) -> None:
    before, transfers, evictions = manager.prepare_prefill(
        session_id=session_id,
        target_instance_index=instance_index,
        history_tokens=0,
        trigger_request_id=f"{session_id}_seed",
    )
    assert before is None and not transfers and not evictions
    manager.expand_prefill(
        session_id=session_id,
        instance_index=instance_index,
        context_tokens=context_tokens,
        trigger_request_id=f"{session_id}_seed",
    )
    manager.mark_complete(session_id, context_tokens)


def _kv_snapshot_bytes(
    manager: KVCacheManager,
    instance_index: int,
) -> tuple[int, ...]:
    return tuple(
        snapshot.kv_cache_bytes
        for snapshot in manager.hbm_snapshots(instance_index)
    )


class MergeV2DirectionTests(unittest.TestCase):
    """merge v2（少并多，2026-09-17 用户裁定）方向真值表与披露契约。

    每方向断言：传输字节/层区间/reason、败者释放、胜者终态全层 LOCAL、
    home 迁移（I7）、守恒（context/total/shard 按合并后全量）、
    ``last_merge_outcome`` 披露快照、I6（本会话零 remote_store）。
    """

    def test_remote_read_local_base_forward_matches_v1_increment_bytes(self):
        # B(=kv(10)) ≥ I(=kv(5)) → 前向：W 搬 home。前向腿逐字节与旧 v1
        # 增量腿一致（LOCAL 基两口径恒等的回归锚）：bytes == kv(new)。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        _, transfers, evictions = manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="remote-read",
        )
        # LOCAL 基不变锚：零化起算、无传输无逐出。
        self.assertEqual(transfers, ())
        self.assertEqual(evictions, ())
        session = manager.session_snapshot("s")
        self.assertEqual(session.shard_bytes, (0, 0))
        self.assertEqual(session.context_tokens, 0)
        self.assertEqual(session.working_kind, "remote-read")
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(len(merge_transfers), 1)
        noc = merge_transfers[0]
        self.assertEqual(noc.kind, "noc_migrate")
        self.assertEqual(noc.reason, "merge_working_copy_to_home")
        self.assertEqual(noc.source_instance_index, 1)
        self.assertEqual(noc.target_instance_index, 0)
        self.assertEqual((noc.layer_start, noc.layer_end), (0, 4))
        expected_increment = kv_cache_shard_bytes_for_tokens(model, 5, 2)
        self.assertEqual(
            tuple(shard.bytes for shard in noc.shards), expected_increment)
        self.assertEqual(noc.total_bytes, sum(expected_increment))
        # I6：本会话零池写。
        self.assertFalse(any(
            t.kind == "remote_store" and t.session_id == "s"
            for t in merge_transfers))
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.home_instance, 0)  # home 不迁移
        self.assertIsNone(merged.working_kind)
        self.assertEqual(merged.context_tokens, 15)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 15, 2))
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (240, 240))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (0, 0))
        self.assertEqual(
            manager.last_merge_outcome,
            {
                "session_id": "s",
                "direction": "forward",
                "zero_byte_flip": False,
                "winner_instance": 0,
                "loser_instance": 1,
                "transferred_bytes": 160,
                "home_flipped": False,
            },
        )

    def test_remote_read_local_base_reverse_moves_base_and_flips_home(self):
        # B(=kv(5)=80) < I(=kv(10)=160) → 翻转：B 搬 exec（home→exec）、
        # home 释放、home 迁移到 exec、终态 LOCAL@exec。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=5)
        manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=5,
            trigger_request_id="r1",
            action="remote-read",
        )
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=10,
            trigger_request_id="r1")

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=10)

        self.assertEqual(len(merge_transfers), 1)
        noc = merge_transfers[0]
        self.assertEqual(noc.kind, "noc_migrate")
        self.assertEqual(noc.reason, "merge_base_to_exec")
        self.assertEqual(noc.source_instance_index, 0)
        self.assertEqual(noc.target_instance_index, 1)
        self.assertEqual((noc.layer_start, noc.layer_end), (0, 4))
        expected_base = kv_cache_shard_bytes_for_tokens(model, 5, 2)
        self.assertEqual(
            tuple(shard.bytes for shard in noc.shards), expected_base)
        self.assertEqual(noc.total_bytes, sum(expected_base))
        self.assertFalse(any(
            t.kind == "remote_store" and t.session_id == "s"
            for t in merge_transfers))
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)  # home 迁移到 exec
        self.assertIsNone(merged.working_kind)
        self.assertEqual(merged.context_tokens, 15)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 15, 2))
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (0, 0))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (240, 240))
        self.assertEqual(
            manager.last_merge_outcome,
            {
                "session_id": "s",
                "direction": "reverse",
                "zero_byte_flip": False,
                "winner_instance": 1,
                "loser_instance": 0,
                "transferred_bytes": 160,
                "home_flipped": True,
            },
        )

    def _prepare_partial_hybrid(
        self,
        manager: KVCacheManager,
        *,
        layer_start: int,
        history_tokens: int,
        increment_tokens: int,
    ) -> None:
        _seed_completed_session(
            manager, session_id="s", instance_index=0,
            context_tokens=history_tokens)
        manager._evict_suffix(
            manager._sessions["s"],
            phase="completion",
            reason="fixture_partial",
            trigger_request_id="fixture",
            layer_start=layer_start,
        )
        manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=history_tokens,
            trigger_request_id="r1",
            action="remote-read",
        )
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=increment_tokens,
            trigger_request_id="r1")

    def test_remote_read_partial_hybrid_forward_uses_shard_truth(self):
        # 混合形态（p=3）：H(=kv(20)@[0,3)=240) ≥ S+I(=80+80=160) → 前向。
        # 前向腿逐 rank 字节必须取 shard_bytes 真值（S+I=160），而非从
        # context_tokens(=增量 5，80) 派生——D1 两口径分离锚。
        model, manager = _merge_v2_manager()
        self._prepare_partial_hybrid(
            manager, layer_start=3, history_tokens=20, increment_tokens=5)

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(len(merge_transfers), 1)
        noc = merge_transfers[0]
        self.assertEqual(noc.kind, "noc_migrate")
        self.assertEqual(noc.reason, "merge_working_copy_to_home")
        self.assertEqual(noc.source_instance_index, 1)
        self.assertEqual(noc.target_instance_index, 0)
        self.assertEqual(
            tuple(shard.bytes for shard in noc.shards), (160, 160))
        self.assertEqual(noc.total_bytes, 320)
        self.assertNotEqual(
            tuple(shard.bytes for shard in noc.shards),
            kv_cache_shard_bytes_for_tokens(model, 5, 2),
        )
        self.assertFalse(any(
            t.kind == "remote_store" and t.session_id == "s"
            for t in merge_transfers))
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.home_instance, 0)
        self.assertEqual(merged.context_tokens, 25)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 25, 2))
        # home 终态 = H + W = 240 + 160 = 400 = kv(25)。
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (400, 400))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (0, 0))
        self.assertEqual(manager.last_merge_outcome["direction"], "forward")
        self.assertFalse(manager.last_merge_outcome["home_flipped"])
        self.assertEqual(manager.last_merge_outcome["transferred_bytes"], 320)

    def test_remote_read_partial_hybrid_reverse_moves_home_prefix(self):
        # 混合形态（p=2 对半）：H(=160) < S+I(=160+80=240) → 翻转搬 H。
        model, manager = _merge_v2_manager()
        self._prepare_partial_hybrid(
            manager, layer_start=2, history_tokens=20, increment_tokens=5)

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(len(merge_transfers), 1)
        noc = merge_transfers[0]
        self.assertEqual(noc.reason, "merge_base_to_exec")
        self.assertEqual(noc.source_instance_index, 0)
        self.assertEqual(noc.target_instance_index, 1)
        self.assertEqual((noc.layer_start, noc.layer_end), (0, 2))
        expected_home_prefix = kv_cache_shard_bytes_for_layer_range(
            model, 20, 2, layer_start=0, layer_end=2)
        self.assertEqual(
            tuple(shard.bytes for shard in noc.shards), expected_home_prefix)
        self.assertFalse(any(
            t.kind == "remote_store" and t.session_id == "s"
            for t in merge_transfers))
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(merged.context_tokens, 25)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 25, 2))
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (0, 0))
        # exec 终态 = S+I + H = 240 + 160 = 400 = kv(25)。
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (400, 400))
        outcome = manager.last_merge_outcome
        self.assertEqual(outcome["direction"], "reverse")
        self.assertFalse(outcome["zero_byte_flip"])
        self.assertTrue(outcome["home_flipped"])
        self.assertEqual(outcome["transferred_bytes"], 320)

    def test_copy_partial_base_merges_as_zero_byte_flip(self):
        # copy：exec 恒持并集 ⊇ home 侧 → 零字节翻转（无传输、home 释放）。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager._evict_suffix(
            manager._sessions["s"],
            phase="completion",
            reason="fixture_partial",
            trigger_request_id="fixture",
        )
        _, transfers, _ = manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="copy",
        )
        # 两腿：前缀 NoC [0,p) + 后缀池恢复 [p,L)。
        self.assertEqual(
            [(t.kind, t.layer_start, t.layer_end) for t in transfers],
            [("noc_migrate", 0, 2), ("remote_load", 2, 4)],
        )
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=15,
            trigger_request_id="r1")

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(merge_transfers, ())
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(merged.context_tokens, 15)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 15, 2))
        # home 侧基础前缀（80/rank）释放；exec 并集（240/rank）转正。
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (0, 0))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (240, 240))
        self.assertEqual(
            manager.last_merge_outcome,
            {
                "session_id": "s",
                "direction": "reverse",
                "zero_byte_flip": True,
                "winner_instance": 1,
                "loser_instance": 0,
                "transferred_bytes": 0,
                "home_flipped": True,
            },
        )

    def test_recompute_at_remote_instance_merges_as_zero_byte_flip(self):
        # recompute@异地（LOCAL 基）：重算复份＋增量 ⊇ home 基础 → 零字节
        # 翻转（merge_transfers 为空、home 侧释放、home := exec）。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        _, transfers, evictions = manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="recompute",
        )
        self.assertEqual((transfers, evictions), ((), ()))
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=15,
            trigger_request_id="r1")

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(merge_transfers, ())
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(merged.context_tokens, 15)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 15, 2))
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (0, 0))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (240, 240))
        outcome = manager.last_merge_outcome
        self.assertEqual(outcome["direction"], "reverse")
        self.assertTrue(outcome["zero_byte_flip"])
        self.assertEqual(outcome["transferred_bytes"], 0)
        self.assertTrue(outcome["home_flipped"])

    def test_remote_base_copy_and_recompute_merge_in_place(self):
        # REMOTE 基（无主，裁定③）：in_place——零传输零池写、home := exec、
        # 终态 LOCAL@exec。copy（池恢复全量）与 recompute（重算全量）两形。
        for action in ("copy", "recompute"):
            with self.subTest(action=action):
                model, manager = _merge_v2_manager()
                _seed_completed_session(
                    manager, session_id="s", instance_index=0,
                    context_tokens=10)
                manager._evict_session(
                    manager._sessions["s"],
                    phase="completion",
                    reason="fixture_remote",
                    trigger_request_id="fixture",
                )
                self.assertEqual(
                    manager.session_snapshot("s").location,
                    KVCacheManager.REMOTE_MEMORY,
                )
                _, transfers, _ = manager.prepare_prefill(
                    session_id="s",
                    target_instance_index=1,
                    history_tokens=10,
                    trigger_request_id="r1",
                    action=action,
                )
                if action == "copy":
                    self.assertEqual(
                        [(t.kind, t.layer_start, t.layer_end)
                         for t in transfers],
                        [("remote_load", 0, 4)],
                    )
                else:
                    self.assertEqual(transfers, ())
                manager.expand_prefill(
                    session_id="s", instance_index=1, context_tokens=15,
                    trigger_request_id="r1")

                merge_transfers = manager.merge_back(
                    session_id="s", trigger_request_id="r1", new_tokens=5)

                self.assertEqual(merge_transfers, ())
                merged = manager.session_snapshot("s")
                self.assertEqual(merged.location, KVCacheManager.LOCAL_HBM)
                self.assertEqual(merged.instance_index, 1)
                self.assertEqual(merged.home_instance, 1)
                self.assertIsNone(merged.working_kind)
                self.assertEqual(merged.context_tokens, 15)
                self.assertEqual(
                    merged.shard_bytes,
                    kv_cache_shard_bytes_for_tokens(model, 15, 2),
                )
                self.assertEqual(_kv_snapshot_bytes(manager, 0), (0, 0))
                self.assertEqual(_kv_snapshot_bytes(manager, 1), (240, 240))
                self.assertEqual(
                    manager.last_merge_outcome,
                    {
                        "session_id": "s",
                        "direction": "in_place",
                        "zero_byte_flip": False,
                        "winner_instance": 1,
                        "loser_instance": 0,
                        "transferred_bytes": 0,
                        "home_flipped": True,
                    },
                )

    def test_stay_merge_sets_outcome_and_version_key_blocks_duplicates(self):
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="stay_r",
        )
        self.assertIsNone(manager.last_merge_outcome)
        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="stay_r", new_tokens=5)
        self.assertEqual(merge_transfers, ())
        self.assertEqual(
            manager.last_merge_outcome,
            {
                "session_id": "s",
                "direction": "stay",
                "zero_byte_flip": False,
                "winner_instance": None,
                "loser_instance": None,
                "transferred_bytes": 0,
                "home_flipped": False,
            },
        )
        # 版本键（恰好一次）：同请求重复 merge = 合同类违规。
        with self.assertRaisesRegex(
                RuntimeError, "duplicate merge transaction"):
            manager.merge_back(
                session_id="s", trigger_request_id="stay_r", new_tokens=5)
        # stay 后会话仍在 home，无任何字节/形态变化。
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.home_instance, 0)
        self.assertEqual(
            merged.shard_bytes, kv_cache_shard_bytes_for_tokens(model, 10, 2))

    def test_remote_read_rejects_exec_equal_to_resident_instance(self):
        _, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        with self.assertRaisesRegex(RuntimeError, "exec == resident"):
            manager.prepare_prefill(
                session_id="s",
                target_instance_index=0,
                history_tokens=10,
                trigger_request_id="r1",
                action="remote-read",
            )


class MergeV2CapacityTests(unittest.TestCase):
    """方向回退兜底与双侧深缺口（I8 统一逐出；合同变更 2026-09-17）。

    单层模型（layers=1，权重 20B/rank、2B/token/rank、容量 200B/rank）
    ——单层无 PARTIAL，逐出只剩整会话外迁，手算口径简单。
    """

    @staticmethod
    def _tiny_manager() -> KVCacheManager:
        # 复用 FaceSchedulerTests 的单层微型模型（权重 20B/rank、
        # 2B/token/rank、容量 200B/rank）——手算口径与既有用例同源。
        return FaceSchedulerTests._tiny_kv_manager(
            capacity_bytes=200, layers=1)[3]

    def test_forward_capacity_failure_falls_back_to_reverse(self):
        # home(0) 被活跃 blocker 占满 → 前向 KVCapacityError → 改试反向
        # （exec 容纳 B）成功：翻转搬 B、home 迁移到 exec。
        manager = self._tiny_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        # blocker：活跃（未完成）不可逐 → ins0 剩余 = 200-20-20-160 = 0。
        FaceSchedulerTests._seed_local_session(
            manager, session_id="blocker", instance_index=0,
            context_tokens=80, completion_ns=None)
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (180, 180))
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r1", action="remote-read",
        )
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(len(merge_transfers), 1)
        noc = merge_transfers[0]
        self.assertEqual(noc.reason, "merge_base_to_exec")
        self.assertEqual(noc.source_instance_index, 0)
        self.assertEqual(noc.target_instance_index, 1)
        self.assertEqual(tuple(shard.bytes for shard in noc.shards), (20, 20))
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(merged.context_tokens, 15)
        # home 仅剩 blocker；exec = I + B = 10 + 20 = 30 = kv(15)。
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (160, 160))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (30, 30))
        outcome = manager.last_merge_outcome
        self.assertEqual(outcome["direction"], "reverse")
        self.assertFalse(outcome["zero_byte_flip"])
        self.assertTrue(outcome["home_flipped"])
        self.assertEqual(manager.deep_gap_events, [])

    def test_dual_sided_deep_gap_fails_closed_and_commits_records(self):
        # 双侧均被活跃 blocker 占满 → 双向都 KVCapacityError → RuntimeError
        #（消息含逐 rank 缺口），两个 exc 的 deep_gap_records 都落账。
        manager = self._tiny_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        FaceSchedulerTests._seed_local_session(
            manager, session_id="blocker_home", instance_index=0,
            context_tokens=80, completion_ns=None)
        FaceSchedulerTests._seed_local_session(
            manager, session_id="blocker_exec", instance_index=1,
            context_tokens=80, completion_ns=None)
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r1", action="remote-read",
        )
        # exec 剩余 20，增量 10 放得下（剩余 10）；反向需 B=20 > 10 失败。
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")

        with self.assertRaisesRegex(RuntimeError, "dual-sided deep gap"):
            manager.merge_back(
                session_id="s", trigger_request_id="r1", new_tokens=5)
        records = manager.deep_gap_events
        # 前向@ins0（ranks 0,1）+ 反向@ins1（ranks 2,3）各 2 条。
        self.assertEqual(len(records), 4)
        self.assertEqual(
            sorted(record["rank"] for record in records), [0, 1, 2, 3])
        by_instance = {
            record["instance_index"] for record in records}
        self.assertEqual(by_instance, {0, 1})
        for record in records:
            self.assertEqual(record["phase"], "completion")
            self.assertEqual(record["reason"], "merge_winner_capacity")
        # fail-closed：未结算（版本键未消耗、无 outcome）。
        self.assertIsNone(manager.last_merge_outcome)
        self.assertIsNone(manager._sessions["s"].last_merged_request_id)

    def test_forward_unified_eviction_victims_are_disclosed_not_self(self):
        # I6 精确口径：统一逐出 victim 的 remote_store 允许出现在返回值
        #（session_id ≠ 本会话）；本会话的 remote_store 不得出现。
        manager = self._tiny_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        # filler：已完成可逐 → 前向空间准备触发统一逐出（整会话外迁）。
        FaceSchedulerTests._seed_local_session(
            manager, session_id="filler", instance_index=0,
            context_tokens=80, completion_ns=10)
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (180, 180))
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r1", action="remote-read",
        )
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")

        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        victim_stores = [
            t for t in merge_transfers
            if t.kind == "remote_store"
        ]
        self.assertEqual(len(victim_stores), 1)
        self.assertEqual(victim_stores[0].session_id, "filler")
        self.assertEqual(
            manager.session_snapshot("filler").location,
            KVCacheManager.REMOTE_MEMORY,
        )
        self.assertFalse(any(
            t.kind == "remote_store" and t.session_id == "s"
            for t in merge_transfers))
        self.assertEqual(
            [t.kind for t in merge_transfers if t.session_id == "s"],
            ["noc_migrate"],
        )
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.home_instance, 0)
        self.assertEqual(merged.context_tokens, 15)
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (30, 30))
        self.assertEqual(manager.deep_gap_events, [])


class PrepareRemoteReadHybridTests(unittest.TestCase):
    """prepare_prefill remote-read 分支：LOCAL 基不变锚 + PARTIAL 混合形态。"""

    def test_partial_base_restores_suffix_as_hot_kv(self):
        # 混合形态：remote_load [p,L) + shard_bytes == 后缀 S + base 快照 p
        # + context_tokens = 0（增量口径）。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=20)
        manager._evict_suffix(
            manager._sessions["s"],
            phase="completion",
            reason="fixture_partial",
            trigger_request_id="fixture",
            layer_start=3,
        )
        partial = manager.session_snapshot("s")
        self.assertEqual(partial.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(partial.resident_prefix_layers, 3)

        before, transfers, evictions = manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=20,
            trigger_request_id="r1",
            action="remote-read",
        )

        self.assertEqual(evictions, ())
        self.assertEqual(len(transfers), 1)
        load = transfers[0]
        self.assertEqual(load.kind, "remote_load")
        self.assertEqual(
            load.reason, "history_suffix_pool_restore_working_copy")
        self.assertEqual(load.target_instance_index, 1)
        self.assertEqual((load.layer_start, load.layer_end), (3, 4))
        expected_suffix = kv_cache_shard_bytes_for_layer_range(
            model, 20, 2, layer_start=3, layer_end=4)
        self.assertEqual(
            tuple(shard.bytes for shard in load.shards), expected_suffix)
        self.assertTrue(all(
            shard.edge_rank in manager.edge_ranks
            and shard.source_rank == shard.edge_rank
            for shard in load.shards))
        # 工作副本：后缀物化入账（热 KV），context = 增量口径 0。
        session = manager._sessions["s"]
        self.assertEqual(session.shard_bytes, expected_suffix)
        self.assertEqual(session.total_bytes, sum(expected_suffix))
        self.assertEqual(session.context_tokens, 0)
        self.assertEqual(session.working_kind, "remote-read")
        self.assertEqual(session.location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(session.instance_index, 1)
        self.assertEqual(session.resident_prefix_layers, 4)
        # base 快照（分支前执行）：PARTIAL p=3。
        self.assertEqual(session.base_location,
                         KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(session.base_history_tokens, 20)
        self.assertEqual(session.base_resident_prefix_layers, 3)
        self.assertEqual(session.home_instance, 0)
        # 物理账：home 前缀 + exec 后缀。
        self.assertEqual(_kv_snapshot_bytes(manager, 0), (240, 240))
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (80, 80))
        snapshot = manager.session_snapshot("s")
        self.assertEqual(snapshot.local_shard_bytes, expected_suffix)
        self.assertEqual(snapshot.rank_bytes, ((2, 80), (3, 80)))

    def test_local_base_zeroes_working_copy_unchanged_anchor(self):
        # LOCAL 基：逐字节不变（零化：shard=0/total=0/context=0、无传输）。
        _, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        _, transfers, evictions = manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="remote-read",
        )
        self.assertEqual(transfers, ())
        self.assertEqual(evictions, ())
        session = manager._sessions["s"]
        self.assertEqual(session.shard_bytes, (0, 0))
        self.assertEqual(session.total_bytes, 0)
        self.assertEqual(session.context_tokens, 0)
        self.assertEqual(session.base_location, KVCacheManager.LOCAL_HBM)
        self.assertEqual(session.base_resident_prefix_layers, 4)
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (0, 0))

    def test_hybrid_expand_grows_increment_and_keeps_frozen_suffix(self):
        # 混合形态 expand：context = 增量口径，shard = S + kv(增量)——
        # delta 恰为增量增长（D1 两口径分离）。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=20)
        manager._evict_suffix(
            manager._sessions["s"],
            phase="completion",
            reason="fixture_partial",
            trigger_request_id="fixture",
            layer_start=2,
        )
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=20,
            trigger_request_id="r1", action="remote-read",
        )
        evictions = manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")
        self.assertEqual(evictions, ())
        session = manager._sessions["s"]
        expected_suffix = kv_cache_shard_bytes_for_layer_range(
            model, 20, 2, layer_start=2, layer_end=4)
        expected_increment = kv_cache_shard_bytes_for_tokens(model, 5, 2)
        self.assertEqual(
            session.shard_bytes,
            tuple(
                suffix + increment
                for suffix, increment in zip(
                    expected_suffix, expected_increment)
            ),
        )
        self.assertEqual(session.context_tokens, 5)
        self.assertEqual(_kv_snapshot_bytes(manager, 1), (240, 240))
        # 不变量全量审计（strict 口径）在混合形态下保持一致。
        manager._check_invariants()


class I10ActiveProtectionTest(unittest.TestCase):
    """I10 显式断言（kimi 收尾批 2026-09-17）：混合形态在飞工作副本
    （含准入相恢复的热后缀 S）不受统一 T+E 逐出——victim 资格 fail-closed
    （face_scheduler `_evict_suffix` 的 active 守卫）。混合形态下 S 被误逐
    = 前缀读流失效，本不变量首次成为正确性依赖（方案 §4.1 I10）。"""

    def test_active_hybrid_working_copy_not_evictable(self):
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=20)
        manager._evict_suffix(
            manager._sessions["s"],
            phase="completion",
            reason="fixture_partial",
            trigger_request_id="fixture",
            layer_start=3,
        )
        manager.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=20, trigger_request_id="r1",
            action="remote-read")
        session = manager._sessions["s"]
        self.assertTrue(session.active)
        self.assertGreater(sum(session.shard_bytes), 0)  # 热后缀 S 在账
        # 直接对在飞工作副本触发统一逐出 → fail-closed raise（victim
        # 资格硬保护，非容量类可恢复失败）。
        with self.assertRaises(RuntimeError) as ctx:
            manager._evict_suffix(
                session, phase="decode", reason="probe_evict_active",
                trigger_request_id="probe", layer_start=2)
        self.assertIn(
            "only completed inactive sessions may be evicted",
            str(ctx.exception))
        # 经 _ensure_capacity 亦不得静默驱逐活跃混合会话：需求超出可用
        # （活跃会话非合法 victim）→ KVCapacityError，工作副本原样在账。
        with self.assertRaises(KVCapacityError):
            manager._ensure_capacity(
                1, tuple(b + 1 << 40 for b in session.shard_bytes),
                phase="decode", reason="probe_capacity_active",
                trigger_request_id="probe")
        self.assertEqual(
            manager._sessions["s"].shard_bytes, session.shard_bytes)
        self.assertTrue(manager._sessions["s"].active)


class KVDeltaJournalExportTests(unittest.TestCase):
    """C14 kv_delta_journal 序列化导出接口（F3：§20.3/§22.7-1 移交收口）。

    字段完整：导出行字段面 = KV_DELTA_JOURNAL_FIELDS 冻结口径（C14
    _append_kv_delta_row 的构造序，不发明新字段）。字节守恒：导出的
    transferred/两侧 retained 字节与结算真值逐位一致（与 merge 返回的
    KVTransfer total_bytes、last_merge_outcome 披露快照对账）。
    """

    def test_forward_merge_export_fields_and_byte_conservation(self):
        # 前向（W=kv(5) ≤ B=kv(10)）：导出行与结算真值逐位对账。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="remote-read",
        )
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")
        merge_transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        rows = manager.kv_delta_journal_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # 字段完整：键集与构造序 = C14 冻结口径。
        self.assertEqual(tuple(row), KV_DELTA_JOURNAL_FIELDS)
        self.assertEqual(
            set(row), {
                "seq", "session_id", "trigger_request_id", "working_kind",
                "direction", "zero_byte_flip", "winner_instance",
                "loser_instance", "home_before", "home_after",
                "home_migration", "transferred_bytes",
                "home_side_retained_bytes", "exec_side_retained_bytes",
                "new_tokens", "staging_return_bytes"})
        self.assertEqual(row["seq"], 0)
        self.assertEqual(row["session_id"], "s")
        self.assertEqual(row["trigger_request_id"], "r1")
        self.assertEqual(row["working_kind"], "remote-read")
        self.assertEqual(row["direction"], "forward")
        self.assertFalse(row["zero_byte_flip"])
        self.assertEqual(row["winner_instance"], 0)
        self.assertEqual(row["loser_instance"], 1)
        self.assertEqual(row["home_before"], 0)
        self.assertEqual(row["home_after"], 0)
        self.assertFalse(row["home_migration"])
        # 字节守恒：传输字节 ≡ merge 返回传输的 total_bytes ≡ 披露快照。
        transferred_total = sum(t.total_bytes for t in merge_transfers)
        self.assertEqual(row["transferred_bytes"], transferred_total)
        self.assertEqual(
            row["transferred_bytes"],
            manager.last_merge_outcome["transferred_bytes"])
        # 两侧实际保留量 = 结算时刻账本真值（home 基 B=kv(10)、exec
        # 工作副本 W=kv(5)；remote-read 无 copy journal → journal 残量
        # 口径不适用，取 base 前缀推导值）。
        self.assertEqual(
            row["home_side_retained_bytes"],
            sum(kv_cache_shard_bytes_for_tokens(model, 10, 2)))
        self.assertEqual(
            row["exec_side_retained_bytes"],
            sum(kv_cache_shard_bytes_for_tokens(model, 5, 2)))
        self.assertEqual(row["new_tokens"], 5)
        self.assertEqual(row["staging_return_bytes"], 0)  # F14 恒 0 披露位
        # find 访问器与导出行同源（最新命中）。
        self.assertEqual(manager.kv_delta_find("r1"), row)

    def test_stay_row_seq_chain_and_frozen_copy_semantics(self):
        # stay 早退本地提交也恰一行（行存在 ⇔ 结算完成）；两行 seq 链
        # 0/1；导出为冻结事实的拷贝——消费方改写不回写账本。
        model, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="stay_r",
        )
        manager.merge_back(
            session_id="s", trigger_request_id="stay_r", new_tokens=5)

        rows = manager.kv_delta_journal_rows()
        self.assertEqual([row["seq"] for row in rows], [0])
        stay = rows[0]
        self.assertEqual(stay["direction"], "stay")
        self.assertEqual(stay["transferred_bytes"], 0)
        self.assertEqual(stay["winner_instance"], None)
        self.assertEqual(stay["home_before"], 0)
        # A12'：stay 无胜者 ⇒ home 不变——home_after = home_before
        # （原 None 会被消费端当独立 home 计入集合，乒乓指标假阳性）。
        self.assertEqual(stay["home_after"], 0)
        self.assertFalse(stay["home_migration"])
        self.assertEqual(
            stay["home_side_retained_bytes"],
            sum(kv_cache_shard_bytes_for_tokens(model, 10, 2)))
        self.assertEqual(stay["exec_side_retained_bytes"], 0)

        # 第二次结算（另一会话的 forward）→ seq 链单调推进。
        _seed_completed_session(
            manager, session_id="t", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="t",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="t_r1",
            action="remote-read",
        )
        manager.expand_prefill(
            session_id="t", instance_index=1, context_tokens=5,
            trigger_request_id="t_r1")
        manager.merge_back(
            session_id="t", trigger_request_id="t_r1", new_tokens=5)
        rows = manager.kv_delta_journal_rows()
        self.assertEqual([row["seq"] for row in rows], [0, 1])
        self.assertEqual([row["direction"] for row in rows],
                         ["stay", "forward"])

        # 冻结拷贝语义：改写导出行不回写账本；重复导出逐位一致。
        rows[0]["transferred_bytes"] = -1
        rows[1]["direction"] = "tampered"
        fresh = manager.kv_delta_journal_rows()
        self.assertEqual(fresh[0]["transferred_bytes"], 0)
        self.assertEqual(fresh[1]["direction"], "forward")
        self.assertIsNot(fresh[0], manager.kv_delta_journal[0])

    def test_export_fails_closed_on_field_drift_and_seq_break(self):
        # 字段集漂移/seq 断链 = 账本破损（构造面唯一入口保证不变量，
        # 到达即破损）→ 导出 fail-closed，不得静默降级。
        _, manager = _merge_v2_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s",
            target_instance_index=0,
            history_tokens=10,
            trigger_request_id="stay_r",
        )
        manager.merge_back(
            session_id="s", trigger_request_id="stay_r", new_tokens=5)
        # 注入破损行（模拟账本被外力改写）。
        manager.kv_delta_journal[0]["extra_field"] = 1
        with self.assertRaisesRegex(RuntimeError, "field set drifted"):
            manager.kv_delta_journal_rows()
        del manager.kv_delta_journal[0]["extra_field"]
        manager.kv_delta_journal[0]["seq"] = 7  # seq 断链
        with self.assertRaisesRegex(RuntimeError, "seq chain broken"):
            manager.kv_delta_journal_rows()

    def test_dump_sidecar_per_key_sentinel(self):
        # A12'：dump_joint_kv_ledgers 逐键独立导出——kv_delta_journal_
        # rows() 抛（字段漂移/seq 链 fail-closed）⇒ 哨兵键
        # kv_delta_journal_export_error 落盘、其余三键不受连坐（旧行
        # 为：兜底 except 吞成 sidecar 整体失落 + 消费端误判零结算）。
        import importlib
        online_dir = str(Path(__file__).resolve().parent / "online")
        # A14'（H9，2026-09-22 第三轮复审）：运行时插路径/动态导入在
        # finally 回收（sys.modules 残留 + 路径序滞留会让跨目录收集
        # 的 pytest 会话命中缓存、错载兄弟仓同名 online_service 模块）。
        saved_path = list(sys.path)
        try:
            if online_dir not in sys.path:
                sys.path.insert(0, online_dir)
            online_service = importlib.import_module("online_service")
            _, manager = _merge_v2_manager()
            _seed_completed_session(
                manager, session_id="s", instance_index=0,
                context_tokens=10)
            manager.prepare_prefill(
                session_id="s",
                target_instance_index=0,
                history_tokens=10,
                trigger_request_id="stay_r",
            )
            manager.merge_back(
                session_id="s", trigger_request_id="stay_r", new_tokens=5)
            # 注入字段集漂移（导出 fail-closed 的触发器）。
            manager.kv_delta_journal[0]["extra_field"] = 1
            scheduler = SimpleNamespace(kv_manager=manager)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "joint_kv_ledgers.json")
                online_service.dump_joint_kv_ledgers(scheduler, path)
                payload = json.loads(
                    Path(path).read_text(encoding="utf-8"))
            self.assertNotIn("kv_delta_journal", payload)
            self.assertIn("kv_delta_journal_export_error", payload)
            self.assertIn(
                "RuntimeError", payload["kv_delta_journal_export_error"])
            # 其余键不受连坐。
            self.assertIn("deep_gap_events", payload)
            self.assertIn("copy_handoff_events", payload)
            self.assertIn("merge_degrade_events", payload)
        finally:
            sys.path[:] = saved_path
            sys.modules.pop("online_service", None)

    def test_empty_journal_exports_empty_tuple(self):
        # 零结算 manager：导出 = 空元组（GREEN run 亦保留的披露通道）。
        _, manager = _merge_v2_manager()
        self.assertEqual(manager.kv_delta_journal_rows(), ())
        self.assertIsNone(manager.kv_delta_find("any"))

    def test_kv_delta_index_tracks_latest_and_matches_scan(self):
        # O5：_kv_delta_index 与账本同源——重复 trigger 取最新命中，且
        # 与线性反扫全量一致（原热路径 reversed() 扫描的 O(1) 化）。
        _, manager = _merge_v2_manager()
        for index in range(8):
            manager._append_kv_delta_row(
                session_id="s",
                trigger_request_id="dup" if index % 2 == 0 else f"r{index}",
                working_kind=None,
                direction="in_place",
                zero_byte_flip=False,
                winner_instance=None,
                loser_instance=None,
                home_before=0,
                transferred_bytes=index,
                home_side_retained_bytes=0,
                exec_side_retained_bytes=0,
                new_tokens=0,
            )
        for row in manager.kv_delta_journal:
            self.assertIs(
                manager.kv_delta_find(row["trigger_request_id"]),
                manager._kv_delta_index[row["trigger_request_id"]])
        latest = max(
            (row for row in manager.kv_delta_journal
             if row["trigger_request_id"] == "dup"),
            key=lambda row: row["seq"])
        self.assertIs(manager.kv_delta_find("dup"), latest)
        self.assertEqual(manager.kv_delta_find("dup")["transferred_bytes"], 6)


if __name__ == "__main__":
    unittest.main()
