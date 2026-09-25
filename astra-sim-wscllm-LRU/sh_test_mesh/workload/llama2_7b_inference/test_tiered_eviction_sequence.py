#!/usr/bin/env python3
"""test_tiered_eviction_sequence.py -- session 级 Tiered-LRU 整体逐出专项单测。

用例 1(整体逐出序):4 会话小容量档逼出完整 session 级 LRU 逐出序列,
断言:
  - LRU 严格序:候选池按 (last_completion_ns, session_id) 升序;
  - 逐出单位 = 完整 logical session:每笔 evict_session 覆盖全部 L 层
    (层域 [0, L))、全部 TP shard,事件无 evict_suffix/PARTIAL 残留;
  - 逐笔逐 NPU 重查停机:满足需求即停,允许整体逐出产生的空间过量释放
    (D4-I3 过度逐出守卫已删除);
  - 逐出 = remote_store KVTransfer,受害会话转 REMOTE(无本地实例、零
    驻留层),远端账面按存入 rank 记账;被逐会话再请求 → REMOTE_RESTORE
    全量恢复;retire 静默核销后归零。

用例 2(journal 一致性):全部新 mutation(整体逐出、REMOTE 全量回迁、
retire 远端核销)都在 _journal_transaction 装饰的公开方法内发生——run 末
verify_journal_checksum 通过(重放/对账/守恒三重断言),且全部非权重
行都归属某个事务(transaction_id != 0)。

Run: 在 llama2_7b_inference 目录
    python3 -m pytest test_tiered_eviction_sequence.py -q
"""

from __future__ import annotations

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
    LOCAL_HBM,
    REMOTE_MEMORY,
    SessionKVCacheManager,
    set_metrics_observer,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    build_instances,
)


# 4 层模型(session 级 Tiered-LRU:逐出单位 = 完整 session 的全部 4 层)。
# KV = 32 B/token/rank(tp=1,heads=2);10 token 会话全量 320 B;
# 权重 480 B/rank。
def _tiered_model() -> WscLlmModel:
    return WscLlmModel(4, 4, 4, 2, 4, 1, "gelu")


def _topology(capacity_bytes: int):
    hardware = WscLlmHardware(
        mesh_rows=1,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    return build_instances(
        hardware,
        (
            WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
            WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
        ),
    )


def _seed_session(manager, session_id, completion_ns, tokens=10):
    decision = manager.prepare_history(
        session_id, 0, 0, completion_ns, f"{session_id}0",
        required_context_tokens=tokens)
    assert not decision.admission_blocked
    growth = manager.grow_prefill(
        session_id, tokens, completion_ns, f"{session_id}0")
    assert growth.admitted
    manager.mark_complete(session_id, completion_ns, f"{session_id}0")


class TieredEvictionSequenceTests(unittest.TestCase):
    def test_whole_session_sequence_strict_lru_and_over_release(self):
        """完整 session 级 LRU 逐出:严格 LRU 序、逐笔重查、允许过量释放
        (恰 4 笔 evict_session,零 evict_suffix)。"""
        # 容量 = 权重 480 + 1300:4 会话(各 320)驻留后余 20;新请求
        # 需 1280 → 整体 a(340)仍不足 → 整体 b(660)→ c(980)→
        # d(1300 ≥ 1280)满足即停(过量释放 20 字节,合法)。
        manager = SessionKVCacheManager(
            _topology(480 + 1300), _tiered_model(), strict_invariants=True)
        for session_id, completion_ns in (
            ("a", 10), ("b", 20), ("c", 30), ("d", 40),
        ):
            _seed_session(manager, session_id, completion_ns)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 20)

        decision = manager.prepare_history(
            "e", 0, 0, 50, "e0", required_context_tokens=40)
        self.assertFalse(decision.admission_blocked)

        # 整体逐出完整序列:每笔 = 完整 session(LRU 序),无 evict_suffix。
        sequence = [transfer.session_id for transfer in decision.evictions]
        self.assertEqual(sequence, ["a", "b", "c", "d"])
        self.assertTrue(all(
            transfer.kind == "remote_store"
            for transfer in decision.evictions))
        # 层域 = [0, 4) 全层(完整 session),驻留前缀 4 → 0。
        self.assertEqual(
            [(transfer.layer_start, transfer.layer_end) for transfer in
             decision.evictions],
            [(0, 4)] * 4)
        self.assertEqual(
            [transfer.resident_prefix_layers_before
             for transfer in decision.evictions],
            [4] * 4)
        self.assertEqual(
            [transfer.resident_prefix_layers_after
             for transfer in decision.evictions],
            [0] * 4)
        # 每笔 total/shard = 全向量(10 token × 32 B = 320 B;多 shard 行
        # = 单逻辑 victim,不重复计入)。
        for transfer in decision.evictions:
            self.assertEqual(transfer.total_bytes, 320)
            self.assertEqual(
                sum(shard.bytes for shard in transfer.shards), 320)
            self.assertEqual(len(transfer.shards), 1)
        # reason 串逐字(契约 §5):历史准入路径的整会话 reason。
        self.assertEqual(
            {transfer.reason for transfer in decision.evictions},
            {"history_and_prefill_admission_session"})

        # 事件流:恰 4 笔 evict_session,零 evict_suffix;cause 串覆盖
        # 完整层域 [0, model_layers)。
        event_types = [
            (event.event_type, event.session_id)
            for event in manager.events
            if event.event_type in {"evict_suffix", "evict_session"}
        ]
        self.assertEqual(event_types, [
            ("evict_session", "a"), ("evict_session", "b"),
            ("evict_session", "c"), ("evict_session", "d"),
        ])
        eviction_events = [
            event for event in manager.events
            if event.event_type == "evict_session"]
        for event in eviction_events:
            self.assertEqual(
                event.reason,
                f"evict_history_and_prefill_admission_session:layers0-4")
            self.assertEqual(event.total_bytes, 320)
            self.assertEqual(tuple(event.shard_bytes), (320,))
            self.assertEqual(event.source_instance_index, 0)

        # 逐笔重查停机:恰好 4 笔(过量释放 20 字节合法,无 D4-I3 守卫)。
        self.assertEqual(len(decision.evictions), 4)
        self.assertEqual(manager.deep_gap_events, 0)

        # 二态落点:a..d 全部 REMOTE(无本地实例、零驻留层、无本地残留),
        # e LOCAL;远端账面 = 4×320,全部记在存入 rank(实例 0 的 rank 0)。
        for session_id in "abcd":
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.location, REMOTE_MEMORY)
            self.assertIsNone(snapshot.instance_index)
            self.assertEqual(snapshot.resident_prefix_layers, 0)
            self.assertEqual(snapshot.remote_bytes, 320)
        self.assertEqual(manager.session_snapshot("e").location, LOCAL_HBM)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 1280, 1: 0})
        # 本地不残留被逐会话部分 KV:e 尚未 grow,驻留 = 0(全部 KV 已
        # 随整体逐出进入远端池)。
        self.assertEqual(
            manager.hbm_snapshots(0)[0].resident_kv_bytes, 0)

        # e 长成后余量 = 20(过量释放的字节级证据),驻留 = 全量 1280。
        self.assertTrue(manager.grow_prefill("e", 40, 51, "e0").admitted)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 20)
        self.assertEqual(
            manager.hbm_snapshots(0)[0].resident_kv_bytes, 1280)
        manager.mark_complete("e", 52, "e0")

        # 被逐会话再请求 → REMOTE_RESTORE 全量恢复(唯一远端路径):
        # a 回实例 0;容量缺口由整体逐出 e(已完成的 inactive 会话)补足。
        restore = manager.prepare_history(
            "a", 0, 10, 55, "a1", required_context_tokens=10)
        self.assertEqual(restore.action, "REMOTE_RESTORE")
        self.assertEqual(len(restore.transfers), 1)
        transfer = restore.transfers[0]
        self.assertEqual(transfer.kind, "remote_load")
        self.assertEqual(transfer.reason, "history_remote_restore")
        self.assertEqual((transfer.layer_start, transfer.layer_end), (0, 4))
        self.assertEqual(transfer.total_bytes, 320)
        snapshot = manager.session_snapshot("a")
        self.assertEqual(snapshot.location, LOCAL_HBM)
        self.assertEqual(snapshot.resident_prefix_layers, 4)
        self.assertEqual(snapshot.remote_bytes, 0)
        self.assertEqual(
            manager.hbm_snapshots(0)[0].resident_kv_bytes, 320)
        manager.mark_complete("a", 56, "a1")

        # retire 静默核销:本地层域减账 + 远端账面清零。
        for session_id, completion_ns in (
            ("b", 60), ("c", 61), ("d", 62), ("e", 63), ("a", 64),
        ):
            manager.retire_terminal_session(
                session_id, completion_ns,
                "a1" if session_id == "a" else f"{session_id}0")
        manager.assert_final_state()
        self.assertEqual(manager.session_ids, ())
        self.assertEqual(
            set(manager.remote_bytes_by_rank().values()), {0})


class TieredJournalConsistencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(set_metrics_observer, None)

    def test_all_new_mutations_journal_consistent(self):
        """新 mutation 全走 _journal_transaction:run 末 checksum 通过
        (重放 + 逐 rank 对账 + 守恒),非权重行全部归属事务。"""
        recorder = MemoryActionRecorder(
            MemoryMetricsObserver(4),
            journal_path=Path(self._tmp.name) / "kv_delta_journal.jsonl")
        set_metrics_observer(recorder)
        # 容量 = 480 + 700:a、b 驻留(各 320)后余 60;预约 540 逼出
        # "整体 a → 整体 b"(60→380→700 ≥ 540 满足即停,过量释放合法)。
        manager = SessionKVCacheManager(
            _topology(480 + 700), _tiered_model(), strict_invariants=True)
        _seed_session(manager, "a", 10)
        _seed_session(manager, "b", 20)
        # 预约属于第三方请求(被保护会话为空):整体 a、整体 b,LURU 序。
        reservation = manager.reserve_request_capacity(
            "r0", "new_session", 0, (540,), 30,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation")
        self.assertTrue(reservation.admitted)
        self.assertEqual(
            [(t.session_id, t.reason) for t in reservation.evictions],
            [("a", "static_decode_final_kv_reservation_session"),
             ("b", "static_decode_final_kv_reservation_session")])
        self.assertEqual(
            [t.reason for t in reservation.evictions],
            ["static_decode_final_kv_reservation_session"] * 2)
        self.assertEqual(
            manager.session_snapshot("a").location, REMOTE_MEMORY)
        self.assertEqual(
            manager.session_snapshot("b").location, REMOTE_MEMORY)
        manager.release_request_capacity("r0", 40)

        # REMOTE 全量回迁(唯一远端恢复路径):a 回实例 0。
        decision = manager.prepare_history(
            "a", 0, 10, 60, "a1", required_context_tokens=10)
        self.assertEqual(decision.action, "REMOTE_RESTORE")
        self.assertEqual(
            [(t.kind, t.reason) for t in decision.transfers],
            [("remote_load", "history_remote_restore")])
        manager.grow_prefill("a", 10, 61, "a1")
        manager.mark_complete("a", 62, "a1")

        manager.retire_terminal_session("a", 63, "a1")
        manager.retire_terminal_session("b", 64, "b0")
        manager.assert_final_state()

        rows = [json.loads(line) for line in
                recorder.journal_path.open(encoding="utf-8") if line.strip()]
        causes = [row["cause"] for row in rows]
        # 新 mutation 的 cause 全部落账(契约 §5 逐字):整体逐出 +
        # 全量恢复 + 终态核销。
        for expected in (
            "evict_static_decode_final_kv_reservation_session:layers0-4",
            "history_remote_restore",
            "terminal_session_retire",
        ):
            self.assertIn(expected, causes, msg=f"missing cause {expected}")
        # 旧两段式/PARTIAL cause 不再出现。
        for banned in (
            "suffix_half", "full_fallback", "history_remote_suffix_restore",
            "history_partial_prefix_migrate",
        ):
            self.assertNotIn(banned, "\n".join(causes))
        # 全部非权重行都在某个事务内(transaction_id != 0)——新 mutation
        # 全走 _journal_transaction 的直接证据。
        self.assertTrue(all(
            row["transaction_id"] != 0
            for row in rows if row["cause"] != "model_weight_preload"))
        # 远端列:逐行守恒由重放校验;终态归零。
        remote_deltas = sum(row["remote_delta_bytes"] for row in rows)
        self.assertEqual(remote_deltas, 0)

        summary = manager.verify_journal_checksum()
        self.assertEqual(summary["line_count"], len(rows))
        self.assertTrue(summary["checks"]["manager_state_match"])
        self.assertTrue(summary["checks"]["remote_account_zero"])
        for replayed in summary["ranks"].values():
            self.assertEqual(replayed["resident"], 0)
            self.assertEqual(replayed["reserved"], 0)
            self.assertEqual(replayed["physical"], replayed["weight"])
            self.assertEqual(replayed["remote"], 0)


if __name__ == "__main__":
    unittest.main()
