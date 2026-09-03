#!/usr/bin/env python3
"""test_wsc_relevant_memory_scheduler.py -- relevant_distributed 分配器单测。

覆盖（总文档 §8.1 / 执行文档 §5.1）：
  - 优先序 `(2,0,1)` 与总文档 §7 算例 B 的贪心过程逐项断言（钉 D / P 账本
    拆分 / 守恒 80+60+60=200 / drain 释放 scratch 140 而 own 60 不在此释放）；
  - 背压两档：空域不可行 ValueError（配置非法）与当前压力 None（FCFS 阻塞），
    ②③各自空域/当前档触发用例；①被②③结构性蕴含——回归断言其永不独立触发；
  - 逐 rank 不等 shard 对拍（实例级账本误判场景：总容量够但单 rank 不够必须失败）；
  - scratch 扣/放与 resident 精确合覆盖整段、区间与字节守恒、release 精确还原；
  - 跨重叠 D′ 的全局账本无重复预订；
  - journal 行兼容性（注入 MemoryActionRecorder，四字段 schema 零改动）与
    不装 recorder 时零行为；KV 事件流三类；
  - 多源历史拉回规划、缺省容量路径（权重预载）、增量/严格不变量校验。

算例 B 折算说明：仓库 KV 公式每 token 每实例最少 2 B（2*layers*hidden*
bytes_per_elem，正数维度下界），故 fixture 取 2 B/token 模型——算例 B 的
全部字节量（容量 [260,80,100,100,100] B、prefill 200 B、decode 20 B、
piece 80/60/60 B、scratch 140 B）逐字一致，token 数为算例的一半。

运行：在 llama2_7b_inference 目录
    python3 test_wsc_relevant_memory_scheduler.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from metrics_integration import MemoryActionRecorder  # noqa: E402
from metrics_schema import MemoryMetricsObserver  # noqa: E402
from session_kv_manager import (  # noqa: E402
    kv_cache_shard_bytes_for_tokens,
    model_weight_shard_bytes_by_tp_rank,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    StaticPdRoute,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    build_instances,
    build_static_pd_mapping,
    kv_cache_bytes_for_tokens,
)
from wsc_relevant_memory_scheduler import (  # noqa: E402
    HistoryPullSource,
    KVPiece,
    KVPlacement,
    RelevantKvRequest,
    WscDistributedKvAllocator,
)


# 算例 B 拓扑：line 0-1-2-3-4，TP1 实例，P=0、D=2、path=(0,1,2)。
EXAMPLE_B_FREE_BYTES = (260, 80, 100, 100, 100)


def example_b_topology():
    hardware = WscLlmHardware(
        mesh_rows=5,
        mesh_cols=1,
        local_hbm_capacity_bytes=1000,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            WscLlmInstanceSpec("p0", "1", (0,), PREFILL_ROLE),
            WscLlmInstanceSpec("p1", "2", (1,), PREFILL_ROLE),
            WscLlmInstanceSpec("d2", "3", (2,), DECODE_ROLE),
            WscLlmInstanceSpec("p3", "4", (3,), PREFILL_ROLE),
            WscLlmInstanceSpec("p4", "5", (4,), PREFILL_ROLE),
        ),
    )
    return hardware, topology


def example_b_model() -> WscLlmModel:
    # 2 B/token（layers=1、hidden=1、heads=1、bpe=1 → 2*1*1*1）。
    return WscLlmModel(1, 1, 2, 1, 4, 1)


def example_b_allocator(**kwargs) -> WscDistributedKvAllocator:
    _, topology = example_b_topology()
    return WscDistributedKvAllocator(
        topology,
        example_b_model(),
        rank_kv_free_bytes={
            rank: free for rank, free in enumerate(EXAMPLE_B_FREE_BYTES)
        },
        **kwargs,
    )


def example_b_request(
    request_id: str,
    session_id: str,
    *,
    prefill_tokens: int,
    decode_tokens: int,
    turn_index: int = 0,
) -> RelevantKvRequest:
    return RelevantKvRequest(
        request_id=request_id,
        session_id=session_id,
        turn_index=turn_index,
        prefill_context_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
    )


def ledger_view(allocator: WscDistributedKvAllocator):
    return tuple(
        (s.rank, s.resident_kv_bytes, s.staging_scratch_bytes, s.kv_free_bytes)
        for s in allocator.rank_ledger_snapshots()
    )


class WorkedExampleBTests(unittest.TestCase):
    """算例 B（总文档 §7）：贪心过程追踪 + 背压两档。"""

    def setUp(self) -> None:
        self.hardware, self.topology = example_b_topology()
        self.mapping = build_static_pd_mapping(self.topology)
        self.route = self.mapping.route_for_prefill(0)

    def test_route_and_priority_order_is_two_zero_one(self) -> None:
        self.assertEqual(self.route.path, (0, 1, 2))
        allocator = example_b_allocator()
        placement = allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        self.assertIsInstance(placement, KVPlacement)
        self.assertEqual(placement.static_route, (0, 1, 2))
        self.assertEqual(placement.total_tokens, 110)
        # 散布序（贪心 prefill 段）= (2, 0, 1)：先 D 剩余、再本请求 P、最后
        # 中间 die——legacy 的 (2,1,0,3,4) 序钉子不适用（裁决 #4/#26）。
        scatter_order = tuple(
            piece.instance_index
            for piece in placement.pieces
            if piece.tier != "decode_local"
        )
        self.assertEqual(scatter_order, (2, 0, 1))

    def test_greedy_trace_pieces_and_ledger(self) -> None:
        allocator = example_b_allocator()
        placement = allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        # ③ decode 段 20 B 钉 D2；①-1 D2 [0,40)=80 B（③ 后 D2 余 80 全部
        # 吸收）；①-2 P0 [40,70)=60 B（= 260 − 200 暂存义务的裕量）；①-3
        # d1 [70,100)=60 B。token 数 = 算例字节（2 B/token 折算的一半）。
        self.assertEqual(
            [
                (
                    piece.instance_index,
                    piece.token_start,
                    piece.token_end,
                    piece.tier,
                    piece.distance_to_decode,
                    piece.path,
                )
                for piece in placement.pieces
            ],
            [
                (2, 100, 110, "decode_local", 0, (2,)),
                (2, 0, 40, "scatter_remote", 0, (2,)),
                (0, 40, 70, "prefill_stay", 2, (0, 1, 2)),
                (1, 70, 100, "scatter_remote", 1, (1, 2)),
            ],
        )
        # ② 账本拆分：own 60 resident + scratch 140，合计 200 覆盖整段。
        self.assertEqual(placement.staging_shard_bytes, (140,))
        self.assertEqual(placement.piece_tokens(0), 30)
        # 逐 NPU 账本终态：D2 满（resident 100）、P0 = 60 resident + 140
        # scratch（余 60 裕量）、d1 resident 60（余 20）。
        self.assertEqual(
            ledger_view(allocator),
            ((0, 60, 140, 60), (1, 60, 0, 20), (2, 100, 0, 0), (3, 0, 0, 100), (4, 0, 0, 100)),
        )

    def test_drain_releases_scratch_only_then_release_restores_ledger(self) -> None:
        allocator = example_b_allocator()
        before = ledger_view(allocator)
        placement = allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        allocator.release_staging(placement, now_ns=2)
        # drain 只释放 scratch 140；own 60 是准入时已记的 resident，不在此释放。
        self.assertEqual(
            ledger_view(allocator),
            ((0, 60, 0, 200), (1, 60, 0, 20), (2, 100, 0, 0), (3, 0, 0, 100), (4, 0, 0, 100)),
        )
        with self.assertRaisesRegex(RuntimeError, "already released"):
            allocator.release_staging(placement, now_ns=3)
        allocator.release(placement, now_ns=4)
        # release 精确还原：账本回到放置前。
        self.assertEqual(ledger_view(allocator), before)
        allocator.assert_final_state()
        with self.assertRaisesRegex(RuntimeError, "unknown or stale placement"):
            allocator.release(placement, now_ns=5)

    def test_backpressure_value_error_prefill_exceeds_empty_domain(self) -> None:
        # 空域档（②）：prefill 300 B > P0 空域 260 B——该请求在此拓扑下
        # 永不可准入，配置非法（总文档 §7 背压两档之 ValueError）。
        allocator = example_b_allocator()
        with self.assertRaisesRegex(
            ValueError, r"prefill segment exceeds the empty-domain staging"
        ):
            allocator.try_place(
                example_b_request("too_large", "s1", prefill_tokens=150, decode_tokens=10),
                self.route,
                now_ns=1,
            )
        self.assertEqual(allocator.active_placement_count, 0)

    def test_backpressure_none_under_current_pressure(self) -> None:
        # 当前档（②）：final 260 B（prefill 240 + decode 20），三条件空域
        # 检查全过（240≤260、20≤100、260≤440），但当前 P0 余量仅 200 B
        # < 暂存需求 240 B → 回滚 → None → FCFS 队头阻塞（总文档 §7）。
        allocator = example_b_allocator()
        # 先放一个段 60 B 的请求：P 账本总占用恒等于整段 → P0 余量 200。
        prior = allocator.try_place(
            example_b_request("prior", "s1", prefill_tokens=30, decode_tokens=5),
            self.route,
            now_ns=1,
        )
        self.assertIsNotNone(prior)
        self.assertEqual(
            allocator.rank_ledger_snapshots(0)[0].kv_free_bytes, 200
        )
        blocked = allocator.try_place(
            example_b_request("target", "s2", prefill_tokens=120, decode_tokens=10),
            self.route,
            now_ns=2,
        )
        self.assertIsNone(blocked)
        self.assertEqual(allocator.active_placement_count, 1)
        # 释放事件后 frontier 重查：同一请求可重准入（B3 背压复用路径）。
        allocator.release(prior, now_ns=3)
        retried = allocator.try_place(
            example_b_request("target", "s2", prefill_tokens=120, decode_tokens=10),
            self.route,
            now_ns=4,
        )
        self.assertIsNotNone(retried)

    def test_decode_condition_empty_and_current_tiers(self) -> None:
        # ③ 空域档：decode 120 B > D2 空域 100 B → ValueError。
        allocator = example_b_allocator()
        with self.assertRaisesRegex(
            ValueError, r"decode segment exceeds the empty-domain"
        ):
            allocator.try_place(
                example_b_request("big_decode", "s1", prefill_tokens=10, decode_tokens=60),
                self.route,
                now_ns=1,
            )
        # ③ 当前档：先放一个把 D2 占满的请求，再放 decode > 0 的新请求 → None。
        filler = allocator.try_place(
            example_b_request("filler", "s2", prefill_tokens=20, decode_tokens=30),
            self.route,
            now_ns=1,
        )
        self.assertIsNotNone(filler)
        self.assertEqual(allocator.rank_ledger_snapshots(2)[0].kv_free_bytes, 0)
        blocked = allocator.try_place(
            example_b_request("next", "s3", prefill_tokens=5, decode_tokens=5),
            self.route,
            now_ns=2,
        )
        self.assertIsNone(blocked)

    def test_condition_one_never_triggers_independently(self) -> None:
        """①被②③结构性蕴含（总文档 §3.1 注记）：空域档只可能由 ②/③ 报出。

        对算例 B 拓扑的两条路由做 (prefill, decode) 扫描：所有 ValueError
        必属 ②/③ 档；凡返回 placement 必满足守恒（Σ piece 字节 = kv(final)）。
        同时校验代数蕴含：②③通过 ⇒ ①空域通过。
        """

        for prefill_index in (0, 3):
            route = self.mapping.route_for_prefill(prefill_index)
            for prefill_tokens in range(0, 140, 3):
                for decode_tokens in range(0, 60, 3):
                    allocator = example_b_allocator()
                    request = example_b_request(
                        "sweep", "s", prefill_tokens=prefill_tokens, decode_tokens=decode_tokens
                    )
                    final = request.final_context_tokens
                    try:
                        placement = allocator.try_place(request, route, now_ns=1)
                    except ValueError as error:
                        message = str(error)
                        self.assertNotIn(
                            "final context exceeds", message,
                            "condition (1) empty-domain fired independently",
                        )
                        self.assertRegex(
                            message,
                            r"decode segment exceeds the empty-domain"
                            r"|prefill segment exceeds the empty-domain",
                        )
                        continue
                    if placement is None:
                        continue
                    piece_total = sum(
                        sum(kv_cache_shard_bytes_for_tokens(
                            allocator.model,
                            piece.token_end - piece.token_start,
                            allocator.tp_degree,
                        ))
                        for piece in placement.pieces
                    )
                    self.assertEqual(
                        piece_total,
                        kv_cache_bytes_for_tokens(allocator.model, final),
                    )

    def test_p_fallback_when_margin_is_zero(self) -> None:
        """P 裕量为 0 时的兜底：其他实例吸不完的 token 由 P own piece 承接。

        P0=200 == 整段 200 B（②恰通过）：D2 吸 80、d1 吸 80 后剩 40 B 由
        P0 兜底 own——验证"②通过即贪心总能完成"（①当前档结构性蕴含）。
        """

        _, topology = example_b_topology()
        allocator = WscDistributedKvAllocator(
            topology,
            example_b_model(),
            rank_kv_free_bytes={0: 200, 1: 80, 2: 100, 3: 100, 4: 100},
        )
        placement = allocator.try_place(
            example_b_request("tight", "s1", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        self.assertIsNotNone(placement)
        self.assertEqual(
            [
                (piece.instance_index, piece.token_start, piece.token_end, piece.tier)
                for piece in placement.pieces
            ],
            [
                (2, 100, 110, "decode_local"),
                (2, 0, 40, "scatter_remote"),
                (1, 40, 80, "scatter_remote"),
                (0, 80, 100, "prefill_stay"),
            ],
        )
        # 守恒：80 + 80 + 40 = 200 B；P0 总占用 = 整段（scratch 160 + own 40）。
        self.assertEqual(placement.staging_shard_bytes, (160,))
        self.assertEqual(allocator.rank_ledger_snapshots(0)[0].kv_free_bytes, 0)


class PerRankLedgerTests(unittest.TestCase):
    """逐 NPU 记账（裁决 #1）：实例级账本会误判的场景必须失败。"""

    @staticmethod
    def _tp2_fixture():
        # 2×2 mesh、P={0,1}、D={2,3}、整头分片 (2,1)：每 token 逐 rank
        # 字节 (8, 4)，实例级 12 B/token。
        hardware = WscLlmHardware(2, 2, 1000, 1.0, 2.0, 1.0, 0, 0)
        topology = build_instances(
            hardware,
            (
                WscLlmInstanceSpec("p", "1", (0, 1), PREFILL_ROLE),
                WscLlmInstanceSpec("d", "2", (2, 3), DECODE_ROLE),
            ),
        )
        model = WscLlmModel(1, 6, 4, 3, 8, 1)
        mapping = build_static_pd_mapping(topology)
        return topology, model, mapping

    def test_unequal_shards_single_rank_shortfall_fails(self) -> None:
        topology, model, mapping = self._tp2_fixture()
        # P 实例总量 260 ≥ 120（实例级账本会放行），但 rank0 只有 60 < 80。
        allocator = WscDistributedKvAllocator(
            topology,
            model,
            rank_kv_free_bytes={0: 60, 1: 200, 2: 300, 3: 300},
        )
        with self.assertRaisesRegex(
            ValueError, r"relative rank 0"
        ) as raised:
            allocator.try_place(
                example_b_request("r", "s", prefill_tokens=10, decode_tokens=1),
                mapping.route_for_prefill(0),
                now_ns=1,
            )
        self.assertIn("prefill segment exceeds the empty-domain staging", str(raised.exception))

    def test_unequal_shards_admissible_case_deducts_per_rank(self) -> None:
        topology, model, mapping = self._tp2_fixture()
        allocator = WscDistributedKvAllocator(
            topology,
            model,
            rank_kv_free_bytes={0: 100, 1: 200, 2: 300, 3: 300},
        )
        placement = allocator.try_place(
            example_b_request("r", "s", prefill_tokens=10, decode_tokens=1),
            mapping.route_for_prefill(0),
            now_ns=1,
        )
        self.assertIsNotNone(placement)
        # decode 1 tok = (8,4) 钉 D；D 剩余吸收 prefill（fit = min(300//8,
        # 300//4)=37 tok ≥ 10）→ 全部散布到 D；P 只记 scratch (80,40)。
        self.assertEqual(placement.staging_shard_bytes, (80, 40))
        self.assertEqual(
            [(s.resident_kv_bytes, s.staging_scratch_bytes) for s in allocator.rank_ledger_snapshots(0)],
            [(0, 80), (0, 40)],
        )
        self.assertEqual(
            [(s.resident_kv_bytes, s.staging_scratch_bytes) for s in allocator.rank_ledger_snapshots(1)],
            [(88, 0), (44, 0)],
        )


class ConservationAndOverlapTests(unittest.TestCase):
    """守恒、精确还原、跨重叠 D′ 全局账本。"""

    def setUp(self) -> None:
        _, self.topology = example_b_topology()
        self.mapping = build_static_pd_mapping(self.topology)
        self.route = self.mapping.route_for_prefill(0)
        self.allocator = example_b_allocator()
        self.model = self.allocator.model

    def test_scratch_and_resident_exactly_cover_prefill_segment(self) -> None:
        placement = self.allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        own = sum(
            kv_cache_bytes_for_tokens(
                self.model, piece.token_end - piece.token_start
            )
            for piece in placement.pieces
            if piece.tier == "prefill_stay"
        )
        scattered = sum(
            kv_cache_bytes_for_tokens(
                self.model, piece.token_end - piece.token_start
            )
            for piece in placement.pieces
            if piece.tier == "scatter_remote"
        )
        self.assertEqual(own, 60)
        self.assertEqual(scattered, 140)
        self.assertEqual(sum(placement.staging_shard_bytes), scattered)
        # P 账本总扣减 = own resident + scratch = 整段 200 B（无重复计账）。
        p0 = self.allocator.rank_ledger_snapshots(0)[0]
        self.assertEqual(
            260 - p0.kv_free_bytes,
            own + sum(placement.staging_shard_bytes),
        )
        self.assertEqual(own + scattered, 200)

    def test_interval_and_byte_conservation(self) -> None:
        placement = self.allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        # token 半开区间精确划分 [0, final)。
        intervals = sorted(
            (piece.token_start, piece.token_end) for piece in placement.pieces
        )
        self.assertEqual(intervals[0][0], 0)
        for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:]):
            self.assertEqual(previous_end, next_start)
        self.assertEqual(intervals[-1][1], placement.total_tokens)
        # pieces ⊆ 静态路径实例集合（D′ 域，裁决 #32）。
        for piece in placement.pieces:
            self.assertIn(piece.instance_index, placement.static_route)
        # 逐 piece 逐 rank 字节 = 公式现算，Σ = kv(final)。
        total = 0
        for piece in placement.pieces:
            shards = kv_cache_shard_bytes_for_tokens(
                self.model, piece.token_end - piece.token_start, 1
            )
            total += sum(shards)
        self.assertEqual(
            total, kv_cache_bytes_for_tokens(self.model, placement.total_tokens)
        )

    def test_overlapping_domains_share_global_ledger(self) -> None:
        # r1（路由 0-1-2）把 D2 占满；r2（路由 3-2）的 decode 段也钉 D2 →
        # 全局账本正确扣减 → None；释放 r1 后 r2 可准入（无重复预订）。
        first = self.allocator.try_place(
            example_b_request("r1", "s1", prefill_tokens=20, decode_tokens=30),
            self.route,
            now_ns=1,
        )
        self.assertIsNotNone(first)
        self.assertEqual(self.allocator.rank_ledger_snapshots(2)[0].kv_free_bytes, 0)
        other_route = self.mapping.route_for_prefill(3)
        self.assertEqual(other_route.path, (3, 2))
        second = self.allocator.try_place(
            example_b_request("r2", "s2", prefill_tokens=5, decode_tokens=5),
            other_route,
            now_ns=2,
        )
        self.assertIsNone(second)
        self.allocator.release(first, now_ns=3)
        retried = self.allocator.try_place(
            example_b_request("r2", "s2", prefill_tokens=5, decode_tokens=5),
            other_route,
            now_ns=4,
        )
        self.assertIsNotNone(retried)
        # r2 的 decode 段与散布 piece 都在 D′ {3,2} 内（D 剩余 tier 0 优先
        # 吸收了全部 prefill 段，P3 无需承接）。
        self.assertEqual(
            {piece.instance_index for piece in retried.pieces}, {2}
        )

    def test_available_views(self) -> None:
        self.assertEqual(
            self.allocator.empty_domain_shard_bytes(self.route), (440,)
        )
        self.assertEqual(self.allocator.available_tokens(self.route), 220)
        placement = self.allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        self.assertIsNotNone(placement)
        # P 侧 scratch 与 owner 侧 resident 分别占用各实例账本：D′ 可用 =
        # 440 − (D2 100 + d1 60 + P0 own 60 + P0 scratch 140) = 80。
        self.assertEqual(self.allocator.available_shard_bytes(self.route), (80,))
        self.assertEqual(self.allocator.available_tokens(self.route), 40)


class HistoryPullTests(unittest.TestCase):
    """多源历史拉回规划（总文档 §3.4）。"""

    def test_multi_source_pull_plan_and_conservation(self) -> None:
        allocator = example_b_allocator()
        _, topology = example_b_topology()
        mapping = build_static_pd_mapping(topology)
        route = mapping.route_for_prefill(0)
        placement = allocator.try_place(
            example_b_request("t0r0", "s0", prefill_tokens=100, decode_tokens=10, turn_index=0),
            route,
            now_ns=1,
        )
        allocator.release(placement, now_ns=2)
        # turn>0：历史 = 旧 final（110 tok / 220 B），新 P 仍选 P0。
        sources = allocator.plan_history_pull(
            placement, history_tokens=110, target_instance_index=0, now_ns=3
        )
        self.assertEqual(
            [(source.source_instance_index, source.local_hit) for source in sources],
            [(0, True), (1, False), (2, False)],
        )
        # D2 聚合两段（prefill 散布 [0,40) + decode [100,110)）= 80 + 20 B。
        d2 = sources[2]
        self.assertEqual((d2.token_start, d2.token_end), (0, 110))
        self.assertEqual(sum(d2.shard_bytes), 100)
        # 总字节 = kv(history_tokens_before)。
        self.assertEqual(
            sum(sum(source.shard_bytes) for source in sources), 220
        )
        # 拉回规划零账本变更。
        self.assertEqual(allocator.active_placement_count, 0)
        allocator.assert_final_state()

    def test_history_pull_validates_tokens(self) -> None:
        allocator = example_b_allocator()
        with self.assertRaisesRegex(ValueError, "history_tokens"):
            allocator.plan_history_pull(
                KVPlacement(
                    request_id="ghost",
                    session_id="s",
                    turn_index=0,
                    prefill_instance_index=0,
                    decode_instance_index=2,
                    static_route=(0, 1, 2),
                    total_tokens=10,
                    pieces=(),
                    staging_shard_bytes=(0,),
                ),
                history_tokens=11,
                target_instance_index=0,
            )


class JournalAndEventTests(unittest.TestCase):
    """journal 行兼容性（四字段 schema 零改动）与 KV 事件流三类。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.hardware, self.topology = example_b_topology()
        self.mapping = build_static_pd_mapping(self.topology)
        self.route = self.mapping.route_for_prefill(0)
        self.model = example_b_model()

    def _journal_allocator(self) -> tuple[WscDistributedKvAllocator, MemoryActionRecorder]:
        recorder = MemoryActionRecorder(
            MemoryMetricsObserver(5),
            journal_path=Path(self._tmp.name) / "kv_delta_journal.jsonl",
        )
        allocator = WscDistributedKvAllocator(
            self.topology,
            self.model,
            rank_kv_free_bytes={
                rank: free for rank, free in enumerate(EXAMPLE_B_FREE_BYTES)
            },
            recorder=recorder,
        )
        return allocator, recorder

    @staticmethod
    def _rows(recorder: MemoryActionRecorder) -> list[dict]:
        with recorder.journal_path.open("r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def test_journal_rows_structure_causes_and_conservation(self) -> None:
        allocator, recorder = self._journal_allocator()
        placement = allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=10,
        )
        allocator.release_staging(placement, now_ns=20)
        allocator.release(placement, now_ns=30)
        allocator.assert_final_state()

        rows = self._rows(recorder)
        # 5 行 placement 事务（4 行 pieces→resident + 1 行 scratch→reserved）
        # + 1 行 drain（reserved 负）+ 4 行 retire（resident 负）。
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["sequence"] for row in rows], list(range(len(rows)))
        )
        self.assertEqual(
            [row["transaction_id"] for row in rows],
            [1] * 5 + [2] + [3] * 4,
        )
        self.assertTrue(
            all(row["schema_version"] == MemoryActionRecorder.JOURNAL_SCHEMA_VERSION for row in rows)
        )
        self.assertEqual(
            [row["cause"] for row in rows],
            ["relevant_placement"] * 4
            + ["staging_scratch"]
            + ["staging_scratch"]
            + ["relevant_release"] * 4,
        )
        # 行内四字段 delta 与守恒快照齐全（schema 零改动）。
        for row in rows:
            self.assertIn("weight_delta_bytes", row)
            self.assertIn("resident_kv_delta_bytes", row)
            self.assertIn("reserved_kv_delta_bytes", row)
            self.assertIn("before_bytes", row)
            self.assertIn("after_bytes", row)
            self.assertIn("capacity_bytes", row)
            self.assertEqual(row["weight_delta_bytes"], 0)
        placement_rows = rows[:4]
        self.assertEqual(
            sorted(
                (row["rank"], row["resident_kv_delta_bytes"])
                for row in placement_rows
            ),
            [(0, 60), (1, 60), (2, 20), (2, 80)],
        )
        self.assertEqual(
            (rows[4]["rank"], rows[4]["reserved_kv_delta_bytes"],
             rows[4]["resident_kv_delta_bytes"]),
            (0, 140, 0),
        )
        self.assertEqual(rows[5]["reserved_kv_delta_bytes"], -140)
        self.assertEqual(
            sorted(
                (row["rank"], row["resident_kv_delta_bytes"])
                for row in rows[6:]
            ),
            [(0, -60), (1, -60), (2, -80), (2, -20)],
        )
        self.assertEqual(
            [row["planner_time_ns"] for row in rows], [10] * 5 + [20] + [30] * 4
        )
        # 守恒链：重放全部不变量 + journal 侧终态归零。
        recorder.replay_journal()
        for rank in range(5):
            self.assertEqual(recorder.rank_totals(rank), (0, 0, 0))

    def test_without_recorder_behavior_is_unchanged(self) -> None:
        request = example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10)
        bare = example_b_allocator()
        with_recorder, recorder = self._journal_allocator()
        placement_bare = bare.try_place(request, self.route, now_ns=1)
        placement_journal = with_recorder.try_place(request, self.route, now_ns=1)
        self.assertEqual(placement_bare, placement_journal)
        self.assertEqual(ledger_view(bare), ledger_view(with_recorder))
        bare.release_staging(placement_bare, now_ns=2)
        with_recorder.release_staging(placement_journal, now_ns=2)
        bare.release(placement_bare, now_ns=3)
        with_recorder.release(placement_journal, now_ns=3)
        self.assertEqual(ledger_view(bare), ledger_view(with_recorder))
        self.assertEqual(bare.events, with_recorder.events)

    def test_event_stream_three_classes(self) -> None:
        allocator = example_b_allocator()
        placement = allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=100, decode_tokens=10),
            self.route,
            now_ns=1,
        )
        allocator.release_staging(placement, now_ns=2)
        allocator.release(placement, now_ns=3)
        events = allocator.events
        self.assertEqual(
            [(event.event_type, event.reason) for event in events],
            [
                ("placement", "admission_three_conditions"),
                ("release", "staging_scratch_drain"),
                ("release", "placement_retire"),
            ],
        )
        # placement 事件前后快照 = P0 逐 rank 余量（260 → 60）。
        self.assertEqual(
            events[0].instance_remaining_before_bytes, (260,)
        )
        self.assertEqual(events[0].instance_remaining_after_bytes, (60,))
        self.assertEqual(events[0].trigger_request_id, "r0")
        self.assertEqual(events[0].total_bytes, 220)
        self.assertEqual(events[0].shard_bytes, (220,))

    def test_default_capacity_path_preloads_weight_rows(self) -> None:
        # 缺省路径（不覆盖逐 rank 容量）：账本 capacity − weight 分片，权重
        # 预载行 cause=model_weight_preload、run 末 physical=weight。
        topology = PerRankLedgerTests._tp2_fixture()[0]
        model = WscLlmModel(1, 6, 4, 3, 8, 1)
        weight_shards = model_weight_shard_bytes_by_tp_rank(model, 2)
        recorder = MemoryActionRecorder(
            MemoryMetricsObserver(4),
            journal_path=Path(self._tmp.name) / "weight_journal.jsonl",
        )
        allocator = WscDistributedKvAllocator(
            topology, model, recorder=recorder
        )
        snapshots = allocator.rank_ledger_snapshots()
        self.assertEqual(
            [(s.capacity_bytes, s.model_weight_bytes) for s in snapshots],
            [
                (1000, weight_shards[relative])
                for instance_ranks in (topology.instance(0).ranks, topology.instance(1).ranks)
                for relative, _ in enumerate(instance_ranks)
            ],
        )
        rows = self._rows(recorder)
        self.assertEqual(
            [(row["rank"], row["weight_delta_bytes"], row["cause"]) for row in rows],
            [
                (0, weight_shards[0], "model_weight_preload"),
                (1, weight_shards[1], "model_weight_preload"),
                (2, weight_shards[0], "model_weight_preload"),
                (3, weight_shards[1], "model_weight_preload"),
            ],
        )
        self.assertTrue(all(row["transaction_id"] == 0 for row in rows))
        # 放一个能容下的请求，终态回 physical=weight。
        request = example_b_request("r0", "s0", prefill_tokens=5, decode_tokens=1)
        placement = allocator.try_place(
            request, build_static_pd_mapping(topology).route_for_prefill(0), now_ns=1
        )
        self.assertIsNotNone(placement)
        allocator.release_staging(placement, now_ns=2)
        allocator.release(placement, now_ns=3)
        allocator.assert_final_state()
        recorder.replay_journal()
        for relative, rank in enumerate(topology.instance(0).ranks):
            self.assertEqual(
                recorder.rank_totals(rank), (weight_shards[relative], 0, 0)
            )


class ContractAndGuardTests(unittest.TestCase):
    """接口契约（执行文档 §4.1 duck-typing）与 fail-closed 守卫。"""

    def test_dataclass_contract_field_names(self) -> None:
        # 字段名逐字按契约（B2/B3 duck-typing 的钉死接口）。
        self.assertEqual(
            [field.name for field in dataclasses.fields(KVPiece)],
            [
                "instance_index",
                "token_start",
                "token_end",
                "tier",
                "distance_to_decode",
                "path",
            ],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(KVPlacement)],
            [
                "request_id",
                "session_id",
                "turn_index",
                "prefill_instance_index",
                "decode_instance_index",
                "static_route",
                "total_tokens",
                "pieces",
                "staging_shard_bytes",
            ],
        )
        self.assertTrue(dataclasses.is_dataclass(HistoryPullSource))
        # frozen 契约：准入冻结，生命周期内不可变。
        piece = KVPiece(2, 0, 40, "scatter_remote", 0, (2,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            piece.token_end = 50  # type: ignore[misc]

    def test_placement_piece_outside_route_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the static-route"):
            KVPlacement(
                request_id="r",
                session_id="s",
                turn_index=0,
                prefill_instance_index=0,
                decode_instance_index=2,
                static_route=(0, 1, 2),
                total_tokens=10,
                pieces=(KVPiece(3, 0, 10, "scatter_remote", 1, (3, 2)),),
                staging_shard_bytes=(0,),
            )

    def test_duplicate_place_and_unknown_release_fail_closed(self) -> None:
        allocator = example_b_allocator()
        _, topology = example_b_topology()
        route = build_static_pd_mapping(topology).route_for_prefill(0)
        request = example_b_request("dup", "s0", prefill_tokens=10, decode_tokens=2)
        placement = allocator.try_place(request, route, now_ns=1)
        self.assertIsNotNone(placement)
        with self.assertRaisesRegex(RuntimeError, "already has a live placement"):
            allocator.try_place(request, route, now_ns=2)

    def test_invalid_route_rejected(self) -> None:
        allocator = example_b_allocator()
        bogus = StaticPdRoute(
            prefill_instance_index=0,
            decode_instance_index=2,
            path=(0, 2),  # 0-2 不相邻
            hop_count=1,
            shared_edges=(),
        )
        with self.assertRaisesRegex(ValueError, "invalid static route edge"):
            allocator.try_place(
                example_b_request("r", "s", prefill_tokens=5, decode_tokens=1),
                bogus,
                now_ns=1,
            )

    def test_strict_invariants_audit_catches_registry_drift(self) -> None:
        # SH_STRICT_KV_INVARIANTS 同款语义：strict 开时每次变更后做全量
        # 守恒审计——登记表与账本漂移即 fail-closed。
        allocator = example_b_allocator(strict_invariants=True)
        _, topology = example_b_topology()
        route = build_static_pd_mapping(topology).route_for_prefill(0)
        placement = allocator.try_place(
            example_b_request("r0", "s0", prefill_tokens=30, decode_tokens=5),
            route,
            now_ns=1,
        )
        self.assertIsNotNone(placement)
        # 人为漂移：登记一份账本之外的 placement（多记 60 B resident）。
        allocator._placements["ghost"] = (
            KVPlacement(
                request_id="ghost",
                session_id="s",
                turn_index=0,
                prefill_instance_index=0,
                decode_instance_index=2,
                static_route=(0, 1, 2),
                total_tokens=30,
                pieces=(KVPiece(0, 0, 30, "prefill_stay", 2, (0, 1, 2)),),
                staging_shard_bytes=(0,),
            ),
            True,
        )
        with self.assertRaisesRegex(RuntimeError, "placement/rank KV accounting"):
            allocator.try_place(
                example_b_request("r1", "s1", prefill_tokens=5, decode_tokens=1),
                route,
                now_ns=2,
            )

    def test_incremental_check_catches_ledger_drift(self) -> None:
        # 非严格模式下增量不变量同样守门：账本与第二账本不一致即报错。
        allocator = example_b_allocator()
        rank = allocator.topology.instance(0).ranks[0]
        allocator._rank_ledgers[rank].resident_kv_bytes += 1
        with self.assertRaisesRegex(RuntimeError, "placement/rank KV accounting"):
            allocator._check_invariants_after_mutation({rank})


if __name__ == "__main__":
    unittest.main()
