#!/usr/bin/env python3
"""test_face_tiered_eviction_sequence.py -- session 级 Tiered-LRU 整体逐出序
专项单测(2026-09-25 重写:两阶段半层逐出删除,逐出单位 = 完整 logical
session——全部层、全部 TP shard、全部本地驻留字节一次性整体外迁)。

断言面(规格 session-level Tiered-LRU 测试要求 1-5 + 事件/指标要求):
  1. 完整 session 逐出:多会话 + 容量压力 → 按严格 LRU 序逐 victim 恰一
     笔全量 remote_store [0, model_layers);逐笔后该会话本地零残留;
  2. 禁止部分逐出:单 session 字节 > 空间缺口 → 释放整个 session、允许
     过量释放、不 raise(D4-I3 恰好够守卫已删除),绝不出现多 session 的
     部分层逐出;
  3. LRU 顺序:按 (last_completion_ns, session_id) 升序;tie 时确定性
     稳定排序(session_id 字典序);
  4. 活跃保护:active/in-flight 会话不可选为 victim;直接调用逐出守卫
     fail-closed raise;触发请求自身会话受 protected_sessions 保护;
  5. 耗尽语义:逐光全部合法 victim 仍不足 → admission_blocked + deep_gap,
     不 raise;
  6. KVTransfer 载体:一 victim = 一 EvictionRecord(不因 TP shard 数
     翻倍)、total_bytes = 全 session 字节、layer range = [0, model_layers)
     (before=model_layers / after=0)、逐 rank 边缘端口 + XY 路径在场。

算术口径(FaceModel(4,4,4,2,4,1,"gelu") + tp2 + 容量 1000B/rank):
每 token 全层 KV 16B/rank;权重 240B/rank -> 可用 760B/rank;
会话各 10 token -> 全层 160B/rank、全 session 320B(双 rank 合计)。
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
    REMOTE_MEMORY,
    SessionKVCacheManager,
)


def _manager(capacity_bytes: int = 1000) -> SessionKVCacheManager:
    model = FaceModel(4, 4, 4, 2, 4, 1, "gelu")  # L=4 -> 整体逐出层域 [0,4)
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


class WholeSessionEvictionSequenceTests(unittest.TestCase):
    def test_whole_session_eviction_in_strict_lru_order(self):
        """规格1:压力下逐 victim 恰一笔全量 remote_store [0,L),逐笔后
        本地零残留/远端全量,水位重查在首个覆盖缺口的 victim 处停。"""
        manager = _manager()
        _place_four(manager)

        # 需 560B/rank > 余 120:整体逐出 a(280)、b(440)仍不足,
        # c(600 >= 560)后停——d 不动。
        fit = manager.ensure_physical_fit(0, (560, 560), 50, "r560")

        self.assertTrue(fit.admitted)
        self.assertEqual(
            [record.victim_session_id for record in fit.evictions],
            ["a", "b", "c"],
        )
        # 每 victim 恰一笔整体 store;reason/事件串逐字(全量域)。
        for record in fit.evictions:
            self.assertEqual(record.transfer.kind, "remote_store")
            self.assertEqual(record.transfer.reason, "request_physical_fit_full")
            self.assertEqual(
                (record.transfer.layer_start, record.transfer.layer_end),
                (0, 4),
            )
        events = [
            event for event in manager.events
            if event.event_type == "evict_full"
        ]
        self.assertEqual(
            [event.session_id for event in events], ["a", "b", "c"])
        for event in events:
            self.assertEqual(
                event.reason, "evict_request_physical_fit_full:layers0-4")
            self.assertEqual(event.total_bytes, 320)  # 全 session 字节

        # 每笔后逐 NPU 重查水位:逐笔 +160,停在 c(600 >= 560;逐 b 后
        # 440 < 560 不停)。事件流 before/after 逐点核对。
        watermark = [120]
        for event in events:
            self.assertEqual(
                tuple(event.instance_remaining_before_bytes), (watermark[-1],) * 2)
            watermark.append(event.instance_remaining_after_bytes[0])
        self.assertEqual(watermark, [120, 280, 440, 600])

        # 终态:被逐会话整体 REMOTE——本地零残留(无部分层驻留)、远端 =
        # 全 session 字节;d 保持完整本地。
        for session_id in ("a", "b", "c"):
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.location, REMOTE_MEMORY)
            self.assertIsNone(snapshot.instance_index)
            self.assertEqual(snapshot.resident_prefix_layers, 0)
            self.assertEqual(snapshot.local_bytes, 0)
            self.assertEqual(snapshot.remote_bytes, 320)
        snapshot_d = manager.session_snapshot("d")
        self.assertEqual(snapshot_d.location, LOCAL_HBM)
        self.assertEqual(snapshot_d.resident_prefix_layers, 4)
        self.assertEqual(snapshot_d.local_bytes, 320)  # 160B/rank × 2 rank
        self.assertEqual(snapshot_d.remote_bytes, 0)
        # 账本守恒:d 160B/rank -> 余 760-160=600。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (600, 600),
        )
        manager.assert_final_state()

    def test_oversized_single_session_over_releases(self):
        """规格2:单 session 字节 > 缺口 → 整 session 释放、过量释放允许、
        不 raise(D4-I3 已删);绝不出现部分层逐出。"""
        manager = _manager()
        _admit_and_complete(manager, "a", "a0", 10)
        # 余 600;需 640 -> 缺口 40 < a 的全量 160:整体逐出 a 即过释放
        # 120B/rank,旧 D4-I3"恰好够"守卫若在必 raise——现在必须放行。
        fit = manager.ensure_physical_fit(0, (640, 640), 20, "r640")
        self.assertTrue(fit.admitted)
        self.assertEqual(
            [record.victim_session_id for record in fit.evictions], ["a"])
        snapshot = manager.session_snapshot("a")
        self.assertEqual(snapshot.location, REMOTE_MEMORY)
        self.assertEqual(snapshot.resident_prefix_layers, 0)
        self.assertEqual(snapshot.local_bytes, 0)
        self.assertEqual(snapshot.remote_bytes, 320)
        # 过量释放后的水位(760)高于需求(640):允许。
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (760, 760),
        )
        # 全程无部分逐出:事件流只有整体 evict_full,状态域两值。
        for event in manager.events:
            self.assertNotEqual(event.event_type, "evict_suffix")
        for state in manager._sessions.values():
            self.assertIn(state.location, (LOCAL_HBM, REMOTE_MEMORY))
        manager.assert_final_state()

    def test_lru_order_and_deterministic_tie(self):
        """规格3:不同完成时刻按真实 LRU 序;tie 时按 session_id 升序的
        确定性稳定排序。"""
        manager = _manager()
        # x/y 同一完成时刻(平局),w 更晚。
        _admit_and_complete(manager, "y", "y0", 10)
        _admit_and_complete(manager, "x", "x0", 10)
        _admit_and_complete(manager, "w", "w0", 20)
        # 占 480B/rank,余 280;需 560:LRU 序 (10,x) < (10,y) < (20,w)
        # -> 逐 x(440)、y(600 >= 560 停),平局取 session_id 字典序最小。
        fit = manager.ensure_physical_fit(0, (560, 560), 30, "r560")
        self.assertTrue(fit.admitted)
        self.assertEqual(
            [record.victim_session_id for record in fit.evictions], ["x", "y"])
        self.assertEqual(manager.session_snapshot("w").location, LOCAL_HBM)
        manager.assert_final_state()

    def test_active_inflight_protected(self):
        """规格4:active 不可选为 victim;直接调用逐出守卫 fail-closed
        raise;触发请求自身会话受保护。"""
        manager = _manager()
        _admit_and_complete(manager, "a", "a0", 10)  # 老会话,LRU 头
        # z 保持 ACTIVE(5 token,80B/rank,在飞不可逐)。
        decision = manager.prepare_history(
            "z", 0, 0, 5, "z0", required_context_tokens=5
        )
        self.assertFalse(decision.admission_blocked)
        self.assertTrue(manager.grow_prefill("z", 5, 5, "z0").admitted)

        # 余 520;需 560:候选池只有完成的 a(active 的 z 即使更老也绝不
        # 入池)-> 逐 a(680 >= 560)后停,z 原地完整本地。
        fit = manager.ensure_physical_fit(0, (560, 560), 50, "r560")
        self.assertTrue(fit.admitted)
        self.assertEqual(
            [record.victim_session_id for record in fit.evictions], ["a"])
        snapshot_z = manager.session_snapshot("z")
        self.assertEqual(snapshot_z.location, LOCAL_HBM)
        self.assertTrue(snapshot_z.active)
        self.assertEqual(snapshot_z.resident_prefix_layers, 4)
        self.assertTrue(manager.grow_prefill("z", 6, 50, "z1").admitted)

        # 直调守卫:active 会话不可被逐(fail-closed)。
        with self.assertRaisesRegex(
                RuntimeError, "only completed inactive sessions"):
            manager._evict_session(
                manager._sessions["z"],
                now_ns=60,
                phase="test",
                reason="probe",
                trigger_request_id="probe",
            )
        # z 完成并核销后终态审计通过(active 会话不允许出现在终态)。
        manager.mark_complete("z", 65, "z1")
        manager.retire_terminal_session("z", 65, "z1")
        manager.assert_final_state()

        # 触发请求自身会话保护:独立场景 a(老)/b(新)均完成后,a 下一
        # turn 增量 496B/rank > 余 440——候选池里 a 受 protected_sessions
        # 保护(触发请求自身会话),victim 只能是 b,绝不被自身准入逐出。
        manager2 = _manager()
        _admit_and_complete(manager2, "a", "a0", 10)
        _admit_and_complete(manager2, "b", "b0", 20)
        decision = manager2.prepare_history(
            "a", 0, 10, 30, "a1", required_context_tokens=41
        )
        self.assertFalse(decision.admission_blocked)
        self.assertEqual(
            [record.victim_session_id for record in decision.evictions], ["b"])
        self.assertEqual(manager2.session_snapshot("a").location, LOCAL_HBM)
        self.assertTrue(manager2.session_snapshot("a").active)
        self.assertTrue(manager2.grow_prefill("a", 41, 30, "a1").admitted)
        manager2.mark_complete("a", 35, "a1")
        manager2.retire_terminal_session("a", 35, "a1")
        manager2.retire_terminal_session("b", 20, "b0")
        manager2.assert_final_state()

    def test_exhaustion_keeps_admission_blocked(self):
        """规格5:无合法 victim 或逐光仍不足 -> blocked + deep_gap,不
        raise;active 会话原样保留。"""
        manager = _manager()
        # z 保持 ACTIVE(5 token,80B/rank,不可逐);a/b/c 完成(480B/rank)
        # -> 余 200。需 700:整体逐 a(360)、b(520)、c(680)仍不足 ->
        # 耗尽:admission_blocked + deep_gap,不 raise。
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
            [record.victim_session_id for record in fit.evictions],
            ["a", "b", "c"],
        )
        # 深缺口台账(face 形态:int 计数 + KVCacheEvent)。
        self.assertEqual(manager.deep_gap_events, 1)
        deep_gaps = [
            event for event in manager.events
            if event.event_type == "deep_gap"
        ]
        self.assertEqual(len(deep_gaps), 1)
        self.assertEqual(deep_gaps[0].reason, "exhausted_completed_candidates")
        # a/b/c 全部 REMOTE,z 原地 ACTIVE;账本只剩 z(80B/rank)。
        for session_id in ("a", "b", "c"):
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.location, REMOTE_MEMORY)
            self.assertEqual(snapshot.local_bytes, 0)
        self.assertEqual(manager.session_snapshot("z").location, LOCAL_HBM)
        self.assertTrue(manager.session_snapshot("z").active)
        self.assertEqual(
            tuple(s.remaining_bytes for s in manager.hbm_snapshots(0)),
            (680, 680),
        )
        # 收尾:清掉被阻塞请求的去重键、z 完成核销后终态审计通过。
        manager._forget_pressure_event_keys("r700")
        manager.mark_complete("z", 60, "z0")
        manager.retire_terminal_session("z", 60, "z0")
        manager.assert_final_state()

    def test_store_transfer_edge_ports_and_layer_domain(self):
        """规格事件要求:一 victim = 一 EvictionRecord(不因 TP shard 数
        翻倍)、total_bytes = 全 session、before=model_layers/after=0、
        每 rank 边缘端口 + XY 路径在场。"""
        manager = _manager()
        _admit_and_complete(manager, "a", "a0", 10)
        # 余 600 -> 需 640:整体逐 a,恰一笔恰一 record。
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
                (0, 0, 0, [0], 0, 4),
                (1, 1, 1, [1], 0, 4),
            ],
        )
        # 全量域元数据:before = 全层、after = 0;total_bytes = 全 session
        # 字节(两条 TP shard 传输合计,不折算成多个 victim)。
        self.assertEqual(record.transfer.resident_prefix_layers_before, 4)
        self.assertEqual(record.transfer.resident_prefix_layers_after, 0)
        self.assertEqual(record.transfer.total_bytes, 320)
        self.assertEqual(sum(record.shard_bytes), 320)
        self.assertEqual(record.transfer.kind, "remote_store")
        self.assertEqual(record.transfer.reason, "request_physical_fit_full")
        # 会话事件与 KVTransfer 同源同量。
        events = [
            event for event in manager.events
            if event.event_type == "evict_full"
            and event.session_id == "a"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].total_bytes, record.transfer.total_bytes)
        manager.assert_final_state()


if __name__ == "__main__":
    unittest.main()
