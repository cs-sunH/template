#!/usr/bin/env python3
"""Focused tests for WSC-LLM PD-disaggregated trace-time request mapping."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    InstanceGraph,
    KVAllocation,
    PrefillQueueSnapshot,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    WscLlmRequest,
    WscLlmTimingEntry,
    WscLlmTimingLut,
    WscRelevantKvAllocator,
    SessionKVCacheManager,
    build_instances,
    build_static_pd_mapping,
    estimate_model_weight_bytes,
    plan_wsc_llm_requests,
    select_prefill_instance,
)
from generate_wsc_llm_trace import (  # noqa: E402
    IDLE_SENTINEL_DURATION_NS,
    _add_idle_rank_sentinels,
    _validate_all_rank_dags,
    _validate_rank_dag,
    load_wsc_llm_trace_config,
    select_first_session_requests,
)
from generate_trace import (  # noqa: E402
    COMM_COLL_NODE,
    COMP_NODE,
    ChakraNode,
    REMOTE_WEIGHT_ATTR,
    RequestSpec,
    TraceBuilder,
    load_remote_memory_config,
    shard_extent,
    transformer_pass,
    transformer_pass_aggregated,
)


EXPECTED_RANKS = (
    (20, 21, 26, 27, 32, 33),
    (2, 3, 8, 9, 14, 15),
    (18, 19, 24, 25, 30, 31),
    (22, 23, 28, 29, 34, 35),
    (38, 39, 44, 45, 50, 51),
    (0, 1, 6, 7, 12, 13),
    (4, 5, 10, 11, 16, 17),
    (36, 37, 42, 43, 48, 49),
    (40, 41, 46, 47, 52, 53),
)
EXPECTED_ROLES = (
    DECODE_ROLE,
    PREFILL_ROLE,
    DECODE_ROLE,
    DECODE_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
    PREFILL_ROLE,
)
EXPECTED_STATIC_ROUTES = {
    1: (0, (1, 0)),
    4: (0, (4, 0)),
    5: (2, (5, 2)),
    6: (3, (6, 3)),
    7: (2, (7, 2)),
    8: (3, (8, 3)),
}
# 真实 9 实例布局（trace_config 顺序，roles = [D,P,D,D,P,P,P,P,P]）。
# 中-1 裁决（2026-08-20）：实测该布局下每个 Prefill 实例的最近 Decode 实例
# 唯一——实例级平局不存在；布局若改动引入平局，
# test_real_layout_nearest_decode_is_unique 失败提示需重新裁决。
REAL_LAYOUT_INSTANCE_SPECS = (
    WscLlmInstanceSpec("decode_0", "1", (20, 21, 26, 27, 32, 33), DECODE_ROLE),
    WscLlmInstanceSpec("prefill_1", "2", (2, 3, 8, 9, 14, 15), PREFILL_ROLE),
    WscLlmInstanceSpec("decode_2", "3", (18, 19, 24, 25, 30, 31), DECODE_ROLE),
    WscLlmInstanceSpec("decode_3", "4", (22, 23, 28, 29, 34, 35), DECODE_ROLE),
    WscLlmInstanceSpec("prefill_4", "5", (38, 39, 44, 45, 50, 51), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_5", "6", (0, 1, 6, 7, 12, 13), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_6", "7", (4, 5, 10, 11, 16, 17), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_7", "8", (36, 37, 42, 43, 48, 49), PREFILL_ROLE),
    WscLlmInstanceSpec("prefill_8", "9", (40, 41, 46, 47, 52, 53), PREFILL_ROLE),
)


# ------------------------------------------------------------------------
# request-neutral 合成 fixture(方案 §0.3 / §3 步骤 0-1):checked-in 配置的
# request_queue_csv 为占位路径,正式入口缺失输入 fail-closed。测试需要队列
# 数据时只用本文件内手写的合成队列(绝不引用任何真实 trace 数据),通过
# 临时配置副本把 request_queue_csv 指向合成队列后加载。
# ------------------------------------------------------------------------
REQUEST_QUEUE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)
# 8 个 turn-0 槽位 + 2 个 turn>0 槽位(槽位口径与 legacy 在线测试共用;
# turn-0 行必须带 session_arrival_time_ns,turn>0 行必须带 interval)。
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

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="wscllm_test_fixture_")


def _write_synthetic_queue(path: Path) -> None:
    path.write_text(
        REQUEST_QUEUE_HEADER
        + "\n"
        + "\n".join(",".join(row) for row in SYNTHETIC_QUEUE_ROWS)
        + "\n",
        encoding="utf-8",
    )


def load_checked_in_config() -> object:
    """request-neutral fixture:checked-in 配置 + 手写合成队列(一次性物化)。

    相对路径(hardware/、system/ 等)仍按 SH_TEST_DIR 解析,配置字段语义与
    checked-in 配置一致;仅输入队列槽位替换为合成数据。
    """
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    if not queue_path.exists():
        _write_synthetic_queue(queue_path)
    config_path = Path(_FIXTURE_DIR.name) / "trace_config.csv"
    if not config_path.exists():
        lines = []
        for raw_line in (MODULE_DIR / "trace_config.csv").read_text(
            encoding="utf-8"
        ).splitlines():
            if raw_line.startswith("config,request_queue_csv,"):
                lines.append(
                    "config,request_queue_csv,"
                    + str(queue_path)
                    + ",,,,,request-neutral synthetic fixture queue (unit test)"
                )
            else:
                lines.append(raw_line)
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_wsc_llm_trace_config(config_path)


def checked_in_topology() -> tuple[object, object]:
    config = load_checked_in_config()
    specs = tuple(
        WscLlmInstanceSpec(
            name=group.name,
            pg_name=group.pg_name,
            ranks=group.ranks,
            phase_role=group.phase_role,
        )
        for group in config.inference_groups
    )
    return config, build_instances(config.hardware, specs)


def small_model() -> WscLlmModel:
    return WscLlmModel(
        layers=2,
        hidden_size=16,
        ffn_size=32,
        num_heads=4,
        vocab_size=64,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def line_topology() -> tuple[WscLlmHardware, object]:
    hardware = WscLlmHardware(
        mesh_rows=5,
        mesh_cols=2,
        local_hbm_capacity_bytes=50,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            WscLlmInstanceSpec("p0", "1", (0, 1), PREFILL_ROLE),
            WscLlmInstanceSpec("p1", "2", (2, 3), PREFILL_ROLE),
            WscLlmInstanceSpec("d2", "3", (4, 5), DECODE_ROLE),
            WscLlmInstanceSpec("p3", "4", (6, 7), PREFILL_ROLE),
            WscLlmInstanceSpec("p4", "5", (8, 9), PREFILL_ROLE),
        ),
    )
    return hardware, topology


class WscLlmSchedulerTests(unittest.TestCase):
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

    def test_checked_in_config_is_request_neutral_and_fails_closed_without_input(self) -> None:
        """checked-in 配置 = request-neutral(占位队列路径,不绑定任何默认
        request 队列)+ 缺失输入 fail-closed + 配置字段语义(拓扑/角色/模型/
        硬件;不依赖任何真实 trace 计数,见方案 §0.3 与 §3 步骤 0-1)。"""
        config, topology = checked_in_topology()
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
        self.assertEqual(tuple(group.ranks for group in config.inference_groups), EXPECTED_RANKS)
        self.assertEqual(
            tuple(group.phase_role for group in config.inference_groups),
            EXPECTED_ROLES,
        )
        self.assertEqual(topology.indices_for_role(PREFILL_ROLE), (1, 4, 5, 6, 7, 8))
        self.assertEqual(topology.indices_for_role(DECODE_ROLE), (0, 2, 3))
        self.assertEqual({len(group.ranks) for group in config.inference_groups}, {6})
        self.assertEqual(
            (
                config.layers,
                config.hidden_size,
                config.ffn_size,
                config.num_heads,
                config.vocab_size,
            ),
            (32, 4096, 11008, 32, 32000),
        )
        self.assertEqual(config.request_queue_session_limit, 0)
        self.assertEqual(config.trace_granularity, "request_aggregated")
        self.assertEqual(config.prefill_chunk_size, 512)
        self.assertEqual(config.kv_cache_policy, "session_lru_recompute")
        self.assertEqual(config.kv_reserve_context_tokens, 1_000_000)
        self.assertFalse(config.record_planning_iterations)
        # request-neutral:合成 fixture 队列的计数只反映 fixture 自身,不依赖
        # 任何真实 trace 数据(裸仓库不物化输入,见方案 §3 步骤 0-1)。
        self.assertEqual(config.source_request_count, len(SYNTHETIC_QUEUE_ROWS))
        self.assertEqual(config.source_session_count, 8)
        self.assertEqual(len(config.request_queue), len(SYNTHETIC_QUEUE_ROWS))
        self.assertEqual(
            config.selected_session_ids,
            tuple(f"fixture_s{index}" for index in range(8)),
        )
        # checked-in 配置本体 = request-neutral:request_queue_csv 是占位路径。
        checked_in_text = (MODULE_DIR / "trace_config.csv").read_text(encoding="utf-8")
        queue_line = next(
            line
            for line in checked_in_text.splitlines()
            if line.startswith("config,request_queue_csv,")
        )
        self.assertEqual(
            queue_line.split(",")[2],
            "llama2_7b_inference/request_queue_placeholder.csv",
        )
        self.assertIn("request-neutral", queue_line)
        # 缺失输入 fail-closed:正式入口(checked-in 配置,无物化队列)必须
        # exit 非 0 并打印 missing request queue,不得回退到任何 stub 队列。
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exit_context:
                load_wsc_llm_trace_config()
        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("missing request queue", stderr.getvalue())

    def test_session_lru_mode_keeps_terminal_kv_and_uses_static_route(self) -> None:
        config, _ = checked_in_topology()
        specs = tuple(
            WscLlmInstanceSpec(group.name, group.pg_name, group.ranks, group.phase_role)
            for group in config.inference_groups
        )
        requests = (
            WscLlmRequest(0, "s0", 0, "s0r0", 10, 2, 0, None),
            WscLlmRequest(1, "s1", 0, "s1r0", 10, 2, 0, None),
            WscLlmRequest(2, "s0", 1, "s0r1", 8, 2, None, 1_000),
        )
        plan = plan_wsc_llm_requests(
            hardware=config.hardware,
            model=config.model,
            instance_specs=specs,
            requests=requests,
            p_chunk=512,
            kv_cache_policy="session_lru_recompute",
            record_planning_iterations=False,
        )
        self.assertEqual(plan.p_chunk, 512)
        self.assertEqual(tuple(item.history_action for item in plan.requests), ("NO_HISTORY", "NO_HISTORY", "NOC_MIGRATE"))
        self.assertTrue(all(not item.terminal_kv_release_at_completion for item in plan.requests))
        self.assertTrue(all(item.state == "RESIDENT" for item in plan.final_session_snapshots))

    def test_decode_instances_are_center_prioritized_and_validation_rejects_inverse(self) -> None:
        _, topology = checked_in_topology()
        decode_distances = [
            topology.instance(index).wafer_center_manhattan_distance
            for index in topology.indices_for_role(DECODE_ROLE)
        ]
        prefill_distances = [
            topology.instance(index).wafer_center_manhattan_distance
            for index in topology.indices_for_role(PREFILL_ROLE)
        ]
        self.assertLessEqual(max(decode_distances), min(prefill_distances))

        hardware = WscLlmHardware(3, 2, 100, 1.0, 1.0, 1.0, 0, 0)
        with self.assertRaisesRegex(ValueError, "center-prioritized"):
            build_instances(
                hardware,
                (
                    WscLlmInstanceSpec("outer_decode", "1", (0, 1), DECODE_ROLE),
                    WscLlmInstanceSpec("center_prefill", "2", (2, 3), PREFILL_ROLE),
                    WscLlmInstanceSpec("outer_prefill", "3", (4, 5), PREFILL_ROLE),
                ),
            )

    def test_exact_static_routes_are_one_hop_and_edge_disjoint(self) -> None:
        _, topology = checked_in_topology()
        mapping = build_static_pd_mapping(topology, alpha=4.0)
        actual = {
            route.prefill_instance_index: (
                route.decode_instance_index,
                route.path,
            )
            for route in mapping.routes
        }
        self.assertEqual(actual, EXPECTED_STATIC_ROUTES)
        self.assertEqual(mapping.total_hops, 6)
        self.assertEqual(mapping.shared_edge_occurrences, 0)
        self.assertEqual(mapping.adjusted_transfer_cost, 6.0)
        self.assertEqual({count for _, _, count in mapping.edge_use_counts}, {1})
        self.assertTrue(all(not route.shared_edges for route in mapping.routes))
        expected_domains = {
            1: (0, 1, 4),
            4: (0, 4, 1),
            5: (2, 5, 7),
            7: (2, 7, 5),
            6: (3, 6, 8),
            8: (3, 8, 6),
        }
        for prefill_index, expected_domain in expected_domains.items():
            allocator = WscRelevantKvAllocator(
                topology,
                mapping,
                model_weight_bytes=0,
            )
            allocation = allocator.allocate(
                request_id=f"domain_{prefill_index}",
                route=mapping.route_for_prefill(prefill_index),
                total_bytes=1,
            )
            self.assertEqual(
                allocation.relevant_instance_indices,
                expected_domain,
            )

    def test_real_layout_nearest_decode_is_unique(self) -> None:
        """钉住"真实布局无实例平局"（中-1 裁决 2026-08-20）。

        本仓 decode 为静态最近实例映射；实测真实 9 实例布局下每个 Prefill
        实例的最近 Decode 实例唯一——实例级平局不存在。布局若改动引入
        平局，本测试失败提示需重新裁决（勿静默改断言迁就新布局）。
        """
        hardware = WscLlmHardware(
            mesh_rows=9,
            mesh_cols=6,
            local_hbm_capacity_bytes=50,
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        topology = build_instances(hardware, REAL_LAYOUT_INSTANCE_SPECS)
        graph = InstanceGraph(topology)
        decode_indices = topology.indices_for_role(DECODE_ROLE)
        self.assertEqual(decode_indices, (0, 2, 3))
        expected_nearest_decode = {1: 0, 4: 0, 5: 2, 6: 3, 7: 2, 8: 3}
        for prefill_index in topology.indices_for_role(PREFILL_ROLE):
            decode_distances = {
                decode_index: graph.shortest_distance(prefill_index, decode_index)
                for decode_index in decode_indices
            }
            nearest_distance = min(decode_distances.values())
            nearest_decode_indices = tuple(
                decode_index
                for decode_index in decode_indices
                if decode_distances[decode_index] == nearest_distance
            )
            self.assertEqual(
                len(nearest_decode_indices),
                1,
                msg=(
                    f"real-layout nearest-Decode tie for Prefill instance "
                    f"{prefill_index} at distance {nearest_distance} "
                    f"(decode distances {decode_distances}); the 2026-08-20 "
                    "decode tie-break ruling requires re-adjudication"
                ),
            )
            self.assertEqual(
                nearest_decode_indices[0],
                expected_nearest_decode[prefill_index],
            )

    def test_prefill_selection_uses_request_count_then_config_order(self) -> None:
        queues = (
            PrefillQueueSnapshot(1, 2),
            PrefillQueueSnapshot(4, 1),
            PrefillQueueSnapshot(5, 1),
        )
        self.assertEqual(select_prefill_instance(queues), 4)
        tied_out_of_order = (
            PrefillQueueSnapshot(8, 0),
            PrefillQueueSnapshot(1, 0),
        )
        self.assertEqual(select_prefill_instance(tied_out_of_order), 1)

    def test_timing_lut_is_phase_exclusive_and_not_a_mixed_pd_lut(self) -> None:
        hardware, _ = line_topology()
        lut = WscLlmTimingLut.build(
            hardware,
            small_model(),
            instance_sizes=(2,),
            p_chunk=64,
            request_count=2,
            max_d_token=500,
        )
        self.assertTrue(lut.entries)
        for entry in lut.entries:
            self.assertNotEqual(entry.p_chunk > 0, entry.d_batch > 0)
            self.assertEqual(
                entry.phase_role,
                PREFILL_ROLE if entry.p_chunk > 0 else DECODE_ROLE,
            )
        nearest = lut.lookup(
            phase_role=DECODE_ROLE,
            instance_size=2,
            p_chunk=0,
            d_batch=1,
            d_token=384,
        )
        self.assertEqual(nearest.d_token, 256)
        with self.assertRaises(ValueError):
            WscLlmTimingEntry(2, PREFILL_ROLE, 64, 1, 256, 10)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "wsc_llm_timing_lut.csv"
            lut.export_csv(output)
            contents = output.read_text(encoding="utf-8")
            self.assertIn("phase_role", contents)
            self.assertIn("iteration_time_ns", contents)

    def test_wsc_relevant_kv_priority_release_and_capacity_error(self) -> None:
        _, topology = line_topology()
        mapping = build_static_pd_mapping(topology)
        route = mapping.route_for_prefill(0)
        self.assertEqual(route.path, (0, 1, 2))
        allocator = WscRelevantKvAllocator(
            topology,
            mapping,
            model_weight_bytes=0,
        )
        allocator.remaining_capacity[:] = [60, 80, 100, 100, 100]
        allocation = allocator.allocate(
            request_id="r0",
            route=route,
            total_bytes=220,
        )
        self.assertIsInstance(allocation, KVAllocation)
        self.assertEqual(allocation.relevant_instance_indices, (2, 1, 0, 3, 4))
        self.assertEqual(
            [
                (
                    piece.instance_index,
                    piece.bytes,
                    piece.location_priority,
                    piece.path,
                )
                for piece in allocation.pieces
            ],
            [
                (2, 100, "decode", (2,)),
                (1, 80, "selected_path_intermediate", (2, 1)),
                (0, 40, "selected_prefill", (2, 1, 0)),
            ],
        )
        self.assertEqual(allocator.remaining_capacity[3:], [100, 100])
        allocator.release(allocation)
        self.assertEqual(tuple(allocator.remaining_capacity), (60, 80, 100, 100, 100))

        allocator2 = WscRelevantKvAllocator(
            topology,
            mapping,
            model_weight_bytes=0,
        )
        with self.assertRaisesRegex(
            ValueError,
            r"Relevant\(P,D\).*Decode remapping.*disabled",
        ):
            allocator2.allocate(request_id="too_large", route=route, total_bytes=501)

    def test_64_gib_fails_fast_for_the_exact_one_million_reserve(self) -> None:
        config = load_checked_in_config()
        hardware = WscLlmHardware(2, 6, 64 * 1024**3, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d_tp6", "1", tuple(range(6)), DECODE_ROLE),
                WscLlmInstanceSpec("p_tp6", "2", tuple(range(6, 12)), PREFILL_ROLE),
            ),
        )
        with self.assertRaisesRegex(ValueError, r"relative_tp_rank=0"):
            SessionKVCacheManager(
                topology,
                config.model,
                reserve_context_tokens=1_000_000,
            )

    def test_manager_deletes_multiple_lru_victims_but_not_active_kv(self) -> None:
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        hardware = WscLlmHardware(1, 2, 400, 1.0, 1.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
                WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
            ),
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
        self.assertTrue(manager.session_snapshot("c").active)
        manager.mark_complete("c", 31, "c0")
        manager.assert_final_state()

    def test_dedicated_phase_iterations_least_occupied_queues_and_fixed_decode(self) -> None:
        config, topology = checked_in_topology()
        specs = tuple(
            WscLlmInstanceSpec(
                group.name,
                group.pg_name,
                group.ranks,
                group.phase_role,
            )
            for group in config.inference_groups
        )
        requests = tuple(
            WscLlmRequest(
                queue_index=index,
                session_id=f"s{index}",
                turn_index=0,
                request_id=f"r{index}",
                prefill_length=12,
                decode_length=2,
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            )
            for index in range(7)
        )
        plan = plan_wsc_llm_requests(
            hardware=config.hardware,
            model=small_model(),
            instance_specs=specs,
            requests=requests,
        )
        self.assertEqual(
            [request.prefill_instance_index for request in plan.requests],
            [1, 4, 5, 6, 7, 8, 1],
        )
        self.assertEqual(plan.requests[-1].prefill_assignment_key, (1, 1))
        for request in plan.requests:
            expected_decode, expected_path = EXPECTED_STATIC_ROUTES[
                request.prefill_instance_index
            ]
            self.assertEqual(request.decode_instance_index, expected_decode)
            self.assertEqual(request.static_route.path, expected_path)
            self.assertTrue(request.terminal_kv_release_at_completion)
        for iteration in plan.iterations:
            self.assertEqual(
                topology.instance(iteration.instance_index).phase_role,
                iteration.phase_role,
            )
            if iteration.phase_role == PREFILL_ROLE:
                self.assertIsNotNone(iteration.prefill_request_id)
                self.assertFalse(iteration.decode_request_ids)
            else:
                self.assertIsNone(iteration.prefill_request_id)
                self.assertTrue(iteration.decode_request_ids)

    def test_capacity_pressure_stops_prefill_fcfs_until_kv_release(self) -> None:
        model = small_model()
        usable_bytes_per_instance = 2_000
        hardware = WscLlmHardware(
            mesh_rows=1,
            mesh_cols=2,
            local_hbm_capacity_bytes=(
                estimate_model_weight_bytes(model) + usable_bytes_per_instance
            ),
            local_hbm_bandwidth_gbps=1.0,
            d2d_bandwidth_gbps=2.0,
            peak_perf_tflops=1.0,
            d2d_latency_ns=0,
            local_hbm_latency_ns=0,
        )
        specs = (
            WscLlmInstanceSpec("d0", "1", (0,), DECODE_ROLE),
            WscLlmInstanceSpec("p1", "2", (1,), PREFILL_ROLE),
        )
        requests = (
            WscLlmRequest(0, "s0", 0, "r0", 4, 20, 0, None),
            WscLlmRequest(1, "s1", 0, "r1", 4, 20, 0, None),
        )

        plan = plan_wsc_llm_requests(
            hardware=hardware,
            model=model,
            instance_specs=specs,
            requests=requests,
        )

        first, second = plan.requests
        self.assertEqual(first.static_route.path, (1, 0))
        self.assertEqual(second.static_route.path, (1, 0))
        self.assertEqual(first.kv_allocation.total_bytes, 3_072)
        self.assertEqual(second.kv_allocation.total_bytes, 3_072)
        self.assertEqual(
            first.completion_ns,
            second.prefill_start_ns,
            "the blocked FCFS head should resume at the first capacity-release event",
        )
        self.assertGreater(second.prefill_start_ns, first.prefill_complete_ns)
        self.assertTrue(first.terminal_kv_release_at_completion)
        self.assertTrue(second.terminal_kv_release_at_completion)

    def test_multi_turn_planning_is_deterministic_and_preserves_session_timing(self) -> None:
        config, _ = checked_in_topology()
        specs = tuple(
            WscLlmInstanceSpec(
                group.name,
                group.pg_name,
                group.ranks,
                group.phase_role,
            )
            for group in config.inference_groups
        )
        requests = (
            WscLlmRequest(0, "s0", 0, "s0r0", 20, 3, 0, None),
            WscLlmRequest(1, "s0", 1, "s0r1", 11, 2, None, 1000),
            WscLlmRequest(2, "s1", 0, "s1r0", 18, 2, 0, None),
            WscLlmRequest(3, "s1", 1, "s1r1", 9, 2, None, 2000),
        )
        first = plan_wsc_llm_requests(
            hardware=config.hardware,
            model=small_model(),
            instance_specs=specs,
            requests=requests,
        )
        second = plan_wsc_llm_requests(
            hardware=config.hardware,
            model=small_model(),
            instance_specs=specs,
            requests=requests,
        )
        signature = lambda plan: [
            (
                request.request_id,
                request.estimated_arrival_ns,
                request.prefill_instance_index,
                request.decode_instance_index,
                request.static_route.path,
                request.completion_ns,
            )
            for request in plan.requests
        ]
        self.assertEqual(signature(first), signature(second))
        by_id = {request.request_id: request for request in first.requests}
        self.assertFalse(by_id["s0r0"].terminal_kv_release_at_completion)
        self.assertTrue(by_id["s0r1"].terminal_kv_release_at_completion)
        self.assertFalse(by_id["s1r0"].terminal_kv_release_at_completion)
        self.assertTrue(by_id["s1r1"].terminal_kv_release_at_completion)
        self.assertEqual(
            by_id["s0r1"].estimated_arrival_ns,
            by_id["s0r0"].completion_ns + 1000,
        )
        self.assertEqual(
            by_id["s1r1"].estimated_arrival_ns,
            by_id["s1r0"].completion_ns + 2000,
        )
        for request in first.requests:
            self.assertGreaterEqual(request.prefill_start_ns, request.estimated_arrival_ns)
            self.assertGreater(request.completion_ns, request.prefill_complete_ns)
            self.assertEqual(
                request.decode_instance_index,
                EXPECTED_STATIC_ROUTES[request.prefill_instance_index][0],
            )
        expected_free = tuple(
            instance.size * config.hardware.local_hbm_capacity_bytes
            - estimate_model_weight_bytes(small_model())
            for instance in first.topology.instances
        )
        self.assertEqual(first.final_remaining_capacity_bytes, expected_free)

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
        self.assertEqual(estimate_model_weight_bytes(config.model), 13_476_831_232)

    def test_idle_rank_sentinel_makes_every_builder_nonempty(self) -> None:
        builders = {
            0: TraceBuilder(remote_operand_loads=False),
            1: TraceBuilder(remote_operand_loads=False),
            2: TraceBuilder(remote_operand_loads=False),
        }
        builders[1].comp("rank_1_real_work", num_ops=1, tensor_size=1)
        idle_ranks = _add_idle_rank_sentinels(builders)

        self.assertEqual(idle_ranks, (0, 2))
        self.assertEqual([len(builders[rank].nodes) for rank in builders], [1, 1, 1])
        self.assertEqual(
            builders[0].nodes[0].name,
            "wsc_llm_idle_rank_0000_sentinel",
        )
        self.assertEqual(builders[0].nodes[0].type, COMP_NODE)
        self.assertEqual(
            builders[0].nodes[0].duration_micros,
            IDLE_SENTINEL_DURATION_NS // 1000,
        )
        self.assertEqual(tuple(builders[0].nodes[0].data_deps), ())
        timer_attrs = {
            attr.name: attr.bool_val for attr in builders[0].nodes[0].attr
        }
        self.assertTrue(timer_attrs["is_cpu_op"])
        self.assertTrue(timer_attrs["is_timer_op"])
        self.assertEqual(builders[1].nodes[0].name, "rank_1_real_work")
        _validate_all_rank_dags(builders)
        self.assertTrue(all(builder.nodes for builder in builders.values()))

    def test_rank_dag_validation_accepts_a_legal_small_dag(self) -> None:
        builder = TraceBuilder(remote_operand_loads=False)
        builder.comp("root", num_ops=1, tensor_size=1)
        builder.comp("child", num_ops=1, tensor_size=1)
        _validate_rank_dag(7, builder)

    def test_rank_dag_validation_rejects_missing_dependency_and_cycle(self) -> None:
        missing = TraceBuilder(remote_operand_loads=False)
        missing_node = ChakraNode()
        missing_node.id = 1
        missing_node.name = "missing_dep"
        missing_node.type = COMP_NODE
        missing_node.data_deps.append(99)
        missing.nodes.append(missing_node)
        with self.assertRaisesRegex(
            RuntimeError,
            r"rank 3: node 1 references missing dependency 99",
        ):
            _validate_rank_dag(3, missing)

        cyclic = TraceBuilder(remote_operand_loads=False)
        first = ChakraNode()
        first.id = 1
        first.name = "cycle_1"
        first.type = COMP_NODE
        first.data_deps.append(2)
        second = ChakraNode()
        second.id = 2
        second.name = "cycle_2"
        second.type = COMP_NODE
        second.data_deps.append(1)
        cyclic.nodes.extend((first, second))
        with self.assertRaisesRegex(
            RuntimeError,
            r"rank 4: Chakra ET DAG contains a cycle",
        ):
            _validate_rank_dag(4, cyclic)

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
        self.assertEqual(sum(node.type == COMP_NODE for node in aggregated.nodes), 15)
        self.assertEqual(sum(node.type == COMM_COLL_NODE for node in aggregated.nodes), 2)
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


if __name__ == "__main__":
    unittest.main()
