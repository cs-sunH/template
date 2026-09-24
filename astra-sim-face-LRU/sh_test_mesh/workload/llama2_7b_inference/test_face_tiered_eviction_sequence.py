#!/usr/bin/env python3
"""test_face_tiered_eviction_sequence.py -- B2(2026-09-06)两段式逐出序
专项单测(策略 §6 B2 验证项:4 会话小容量档逼出 半层×N → 整体×M 的
序列断言)。

断言面(契约 §9):
  1. 逐出顺序严格 LRU:(last_completion_ns, session_id) 升序,阶段 1 与
     阶段 2 各自独立按该序;
  2. 阶段 1(半层逐出 LOCAL -> PARTIAL)全部耗尽后才进阶段 2(整体
     remote_store -> REMOTE,唯一回退);
  3. 每笔逐出后逐 NPU 重查水位,够即停(逐到"恰好够"边界);
  4. D4-I3 过度逐出守卫的口径:撤销最后一笔会使受影响 rank 回到不满足;
  5. KVTransfer 载体:remote_store、层域 [2,4)/[0,2)、逐 shard 边缘端口
     与 XY 路径在场;cause 串逐字(evict_{reason}_suffix_half:layersS-L /
     evict_{reason}_full_fallback:layers0-R);
  6. 触发请求自身会话保护(protected_session_id);
  7. 两阶段耗尽仍不足 = face 失败语义(admission_blocked + deep_gap),
     不 raise。

算术口径(FaceModel(4,4,4,2,4,1,"gelu") + tp2 + 容量 1000B/rank):
每 token 全层 KV 16B/rank、半层 8B/rank;权重 240B/rank -> 可用 760B/rank;
会话各 10 token -> 全层 160B/rank、半层段 80B/rank。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from face_scheduler import FaceHardware, FaceInstanceSpec, FaceModel, build_instances  # noqa: E402
from session_kv_manager import (  # noqa: E402
    LOCAL_HBM,
    PARTIAL_HBM_REMOTE,
    REMOTE_MEMORY,
    SessionKVCacheManager,
)


def _manager(capacity_bytes: int = 1000) -> SessionKVCacheManager:
    model = FaceModel(4, 4, 4, 2, 4, 1, "gelu")  # L=4 -> prefix 2 / suffix 2
    hardware = FaceHardware(1, 2, capacity_bytes, 1.0, 1.0, 1.0, 0, 0)
    topology = build_instances(
        hardware,
        (FaceInstanceSpec("tp2", "1", (0, 1)),),
    )
    return SessionKVCacheManager(topology, model)


def _admit_and_complete(manager, session_id, request_id, now_ns, tokens=10):
    decision = manager.prepare_history(
        session_id, 0, 0, now_ns, request_id, required_context_tokens=tokens
    )
    assert not decision.admission_blocked, "fixture 预放置必须可行"
    growth = manager.grow_prefill(session_id, tokens, now_ns, request_id)
    assert growth.admitted, "fixture 预放置 grow_prefill 必须成功"
    manager.mark_complete(session_id, now_ns, request_id)


def _place_four(manager):
    """a/b/c/d 各 10 token 完成于 10/20/30/40 ns -> 占 640B/rank,余 120。"""
    for session_id, now_ns in (("a", 10), ("b", 20), ("c", 30), ("d", 40)):
        _admit_and_complete(manager, session_id, f"{session_id}0", now_ns)


class TieredEvictionSequenceTests(unittest.TestCase):
    def test_stage1_all_halves_then_stage2_fulls_in_lru_order(self):
        manager = _manager()
        _place_four(manager)

        # 需 560B/rank > 余 120:半层 a,b,c,d(+80/笔 -> 440)仍不足 ->
        # 整体 a,b(+80/笔 -> 520,600 >= 560 停)。
        fit = manager.ensure_physical_fit(0, (560, 560), 50, "r560")

        self.assertTrue(fit.admitted)
        self.assertEqual(
            [(record.victim_session_id, record.transfer.kind)
             for record in fit.evictions],
            [
                ("a", "remote_store"), ("b", "remote_store"),
                ("c", "remote_store"), ("d", "remote_store"),
                ("a", "remote_store"), ("b", "remote_store"),
            ],
        )
        # 阶段边界逐字:前 4 笔 suffix_half(layers2-4),后 2 笔
        # full_fallback(layers0-2)。
        self.assertEqual(
            [record.transfer.reason for record in fit.evictions],
            [
                "request_physical_fit_suffix_half",
            ] * 4 + [
                "request_physical_fit_full_fallback",
            ] * 2,
        )
        self.assertEqual(
            [(record.transfer.layer_start, record.transfer.layer_end)
             for record in fit.evictions],
            [(2, 4)] * 4 + [(0, 2)] * 2,
        )

        # 每笔后水位重查:逐笔 +80,停在第 6 笔后(600 >= 560;第 5 笔
        # 520 < 560 不停)。事件流 before/after 逐点核对。
        watermark = [120]
        for record in fit.evictions:
            events = [
                event for event in manager.events
                if event.event_type in ("evict_suffix", "evict_full")
                and event.session_id == record.victim_session_id
                and event.reason == (
                    "evict_request_physical_fit_suffix_half:layers2-4"
                    if record.transfer.reason.endswith("suffix_half")
                    else "evict_request_physical_fit_full_fallback:layers0-2"
                )
            ]
            self.assertEqual(len(events), 1, events)
            event = events[0]
            self.assertEqual(
                tuple(event.instance_remaining_before_bytes), (watermark[-1],) * 2)
            watermark.append(event.instance_remaining_after_bytes[0])
        self.assertEqual(watermark, [120, 200, 280, 360, 440, 520, 600])

        # D4-I3 口径:撤销最后一笔(full b,-80) -> 520 < 560 缺口重开。
        self.assertLess(watermark[-2], 560)
        # 终态:a/b REMOTE(无实例无驻留层),c/d PARTIAL(前缀 2 层)。
        for session_id in ("a", "b"):
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.location, REMOTE_MEMORY)
            self.assertIsNone(snapshot.instance_index)
            self.assertEqual(snapshot.resident_prefix_layers, 0)
            self.assertEqual(snapshot.local_bytes, 0)
            self.assertEqual(snapshot.remote_bytes, 320)
        for session_id in ("c", "d"):
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.location, PARTIAL_HBM_REMOTE)
            self.assertEqual(snapshot.instance_index, 0)
            self.assertEqual(snapshot.resident_prefix_layers, 2)
            self.assertEqual(snapshot.local_bytes, 160)
            self.assertEqual(snapshot.remote_bytes, 160)
        # 账本守恒:c/d 前缀 160B/rank -> 余 760-160=600。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (600, 600),
        )
        manager.assert_final_state()

    def test_watermark_recheck_stops_mid_stage1(self):
        manager = _manager()
        _place_four(manager)

        # 需 300B/rank:半层 a(200),b(280)仍不足,c(360 >= 300)停——
        # 阶段 1 中途停机,d 不动,阶段 2 不进入。
        fit = manager.ensure_physical_fit(0, (300, 300), 50, "r300")
        self.assertTrue(fit.admitted)
        self.assertEqual(
            [record.victim_session_id for record in fit.evictions],
            ["a", "b", "c"],
        )
        self.assertTrue(
            all(record.transfer.reason == "request_physical_fit_suffix_half"
                for record in fit.evictions)
        )
        self.assertEqual(manager.session_snapshot("d").location, LOCAL_HBM)
        self.assertEqual(manager.session_snapshot("a").location, PARTIAL_HBM_REMOTE)
        # 撤销最后一笔(半层 c,-80)-> 280 < 300 缺口重开(D4-I3)。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (360, 360),
        )
        self.assertLess(360 - 80, 300)
        manager.assert_final_state()

    def test_tie_break_by_session_id(self):
        manager = _manager()
        # 两会话同一完成时刻 -> 平局按 session_id 升序。
        for session_id in ("y", "x"):
            _admit_and_complete(manager, session_id, f"{session_id}0", 10)
        # 余 760-320=440 -> 需 500:半层 x(520>=500)即停,平局取 x。
        fit = manager.ensure_physical_fit(0, (500, 500), 20, "r500")
        self.assertTrue(fit.admitted)
        self.assertEqual(
            [record.victim_session_id for record in fit.evictions], ["x"]
        )

    def test_trigger_session_is_protected(self):
        manager = _manager()
        _admit_and_complete(manager, "a", "a0", 10)  # 老会话,LRU 头
        _admit_and_complete(manager, "b", "b0", 20)
        # 会话 a 下一 turn 仍在实例 0:准入需 496B/rank(41 token 全量
        # 656 - 已驻 160),余 440 不足 -> 逐 b(半层 +80 -> 520 >= 496)。
        # a 是更老的候选但受 protected_sessions 保护,绝不被自身逐出。
        decision = manager.prepare_history(
            "a", 0, 10, 30, "a1", required_context_tokens=41
        )
        self.assertFalse(decision.admission_blocked)
        self.assertEqual(
            [record.victim_session_id for record in decision.evictions], ["b"]
        )
        self.assertEqual(decision.action, "LOCAL_HIT")
        self.assertTrue(manager.grow_prefill("a", 41, 30, "a1").admitted)
        snapshot_a = manager.session_snapshot("a")
        self.assertEqual(snapshot_a.location, LOCAL_HBM)
        self.assertTrue(snapshot_a.active)
        self.assertEqual(manager.session_snapshot("b").location, PARTIAL_HBM_REMOTE)

    def test_two_stage_exhaustion_keeps_face_blocked_semantics(self):
        manager = _manager()
        # z 保持 ACTIVE(5 token,80B/rank,不可逐);a/b/c 完成(480B/rank)
        # -> 余 200。需 700:半层 a,b,c(->440)后整体 a,b,c(->680)仍
        # 不足 -> 两阶段耗尽:admission_blocked + deep_gap,不 raise。
        decision = manager.prepare_history(
            "z", 0, 0, 5, "z0", required_context_tokens=5
        )
        self.assertFalse(decision.admission_blocked)
        self.assertTrue(manager.grow_prefill("z", 5, 5, "z0").admitted)
        for session_id, now_ns in (("a", 10), ("b", 20), ("c", 30)):
            _admit_and_complete(manager, session_id, f"{session_id}0", now_ns)

        fit = manager.ensure_physical_fit(0, (700, 700), 50, "r700")
        self.assertFalse(fit.admitted)
        self.assertEqual(fit.insufficient_ranks, (0, 1))
        self.assertEqual(
            [(record.victim_session_id, record.transfer.reason)
             for record in fit.evictions],
            [("a", "request_physical_fit_suffix_half"),
             ("b", "request_physical_fit_suffix_half"),
             ("c", "request_physical_fit_suffix_half"),
             ("a", "request_physical_fit_full_fallback"),
             ("b", "request_physical_fit_full_fallback"),
             ("c", "request_physical_fit_full_fallback")],
        )
        # 深缺口台账(face 形态:int 计数 + KVCacheEvent)。
        self.assertEqual(manager.deep_gap_events, 1)
        deep_gaps = [
            event for event in manager.events
            if event.event_type == "deep_gap"
        ]
        self.assertEqual(len(deep_gaps), 1)
        self.assertEqual(deep_gaps[0].reason, "exhausted_completed_candidates")
        # a/b/c 全部 REMOTE,z 原地 ACTIVE;账本只剩 z 前缀(80B/rank)。
        for session_id in ("a", "b", "c"):
            self.assertEqual(
                manager.session_snapshot(session_id).location, REMOTE_MEMORY)
        self.assertEqual(manager.session_snapshot("z").location, LOCAL_HBM)
        self.assertTrue(manager.session_snapshot("z").active)
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (680, 680),
        )

    def test_transfer_shards_carry_edge_port_and_xy_path(self):
        manager = _manager()
        _admit_and_complete(manager, "a", "a0", 10)
        # 余 760-160=600 -> 需 640:半层 a(680>=640)即停,恰一笔。
        fit = manager.ensure_physical_fit(0, (640, 640), 20, "r640")
        self.assertTrue(fit.admitted)
        self.assertEqual(len(fit.evictions), 1)
        record = fit.evictions[0]
        # 1x2 网格两 rank 皆为边界;直连分支路径 = 单点,边缘端口 = 自身。
        self.assertEqual(
            [
                (shard.source_rank, shard.edge_rank, shard.target_rank,
                 list(shard.noc_path), shard.layer_start, shard.layer_end)
                for shard in record.transfer.shards
            ],
            [
                (0, 0, 0, [0], 2, 4),
                (1, 1, 1, [1], 2, 4),
            ],
        )
        self.assertEqual(record.transfer.resident_prefix_layers_before, 4)
        self.assertEqual(record.transfer.resident_prefix_layers_after, 2)
        self.assertEqual(record.transfer.total_bytes, 160)


if __name__ == "__main__":
    unittest.main()
