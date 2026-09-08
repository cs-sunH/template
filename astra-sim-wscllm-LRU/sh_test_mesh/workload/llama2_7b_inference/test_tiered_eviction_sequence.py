#!/usr/bin/env python3
"""test_tiered_eviction_sequence.py -- B2 三态两段式 LRU 逐出专项单测。

用例 1(两段式逐出序):4 会话小容量档逼出"半层×N → 整体×M"完整序列,
断言:
  - LRU 严格序:两阶段各自按 (last_completion_ns, session_id) 升序;
  - 阶段 1 全耗尽才进阶段 2:首笔整体外迁发生时,全部 LOCAL 候选已
    PARTIAL(半层化);
  - 逐笔逐 NPU 重查停机:恰好在满足需求的最后一笔停(无过度逐出,
    D4-I3 守卫在线 = 不 raise 且无多余笔);
  - 逐出 = remote_store KVTransfer,受害会话转 PARTIAL/REMOTE,远端
    账面按存入 rank 记账;retire 静默核销后归零。

用例 2(journal 一致性):全部新 mutation(suffix/full 逐出、PARTIAL
同实例后缀回迁、PARTIAL 跨实例两段链、REMOTE 全量回迁、retire 远端
核销)都在 _journal_transaction 装饰的公开方法内发生——run 末
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
    PARTIAL_HBM_REMOTE,
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


# 4 层模型:partial_resident_prefix_layers = 4 - 2 = 2,半层后缀 = 层 2-4。
# KV = 32 B/token/rank(tp=1,heads=2);10 token 会话全量 320 B,半层 160 B;
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
    def test_two_stage_sequence_strict_lru_and_per_step_recheck(self):
        """半层×N → 整体×M 完整序列:LRU 严格序、阶段 1 全耗尽才进
        阶段 2、逐笔重查停机(恰好 8 笔,无过度逐出)。"""
        # 容量 = 权重 480 + 1300:4 会话(各 320)驻留后余 20;新请求
        # 需 1280 → 半层 a,b,c,d(+160×4=660)仍不足 → 整体 a,b,c(+480
        # → 1140)仍不足 → 整体 d(+160 → 1300 ≥ 1280)恰停。
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

        # 两段式完整序列:先全部半层化(LRU 序),再整体外迁(同一 LRU
        # 序),恰好在满足需求的最后一笔停。
        sequence = [
            (transfer.session_id, "suffix" if transfer.layer_start else "full")
            for transfer in decision.evictions
        ]
        self.assertEqual(sequence, [
            ("a", "suffix"), ("b", "suffix"), ("c", "suffix"),
            ("d", "suffix"),
            ("a", "full"), ("b", "full"), ("c", "full"), ("d", "full"),
        ])
        self.assertTrue(all(
            transfer.kind == "remote_store"
            for transfer in decision.evictions))
        # 阶段 1 的层域 = [2, 4)(半层),阶段 2 的层域 = [0, resident)。
        self.assertEqual(
            [transfer.layer_start for transfer in decision.evictions],
            [2, 2, 2, 2, 0, 0, 0, 0])
        # cause 串逐字(契约 §5):历史准入路径的 reason 占位。
        self.assertEqual(
            decision.evictions[0].reason,
            "history_and_prefill_admission_suffix_half")
        self.assertEqual(
            decision.evictions[4].reason,
            "history_and_prefill_admission_full_fallback")

        # 阶段 1 全耗尽才进阶段 2:首笔整体外迁(a)前,a..d 已全部
        # PARTIAL(事件流序佐证:第 5 笔逐出前恰有 4 笔 evict_suffix)。
        event_types = [
            (event.event_type, event.session_id)
            for event in manager.events
            if event.event_type in {"evict_suffix", "evict_session"}
        ]
        self.assertEqual(event_types, [
            ("evict_suffix", "a"), ("evict_suffix", "b"),
            ("evict_suffix", "c"), ("evict_suffix", "d"),
            ("evict_session", "a"), ("evict_session", "b"),
            ("evict_session", "c"), ("evict_session", "d"),
        ])

        # 逐笔重查停机:恰好 8 笔,无第 9 笔;D4-I3 过度逐出守卫在线
        # (最后一笔 d 整体外迁后 free=1300 ≥ 1280,撤销它则 1140 < 1280,
        # 守卫不触发 = 该笔必要)。
        self.assertEqual(len(decision.evictions), 8)
        self.assertEqual(manager.deep_gap_events, 0)

        # 三态落点:a..d 全部 REMOTE(无本地实例、无驻留层),e LOCAL;
        # 远端账面 = 4×320,全部记在存入 rank(实例 0 的 rank 0)。
        for session_id in "abcd":
            snapshot = manager.session_snapshot(session_id)
            self.assertEqual(snapshot.location, REMOTE_MEMORY)
            self.assertIsNone(snapshot.instance_index)
            self.assertEqual(snapshot.resident_prefix_layers, 0)
            self.assertEqual(snapshot.remote_bytes, 320)
        self.assertEqual(manager.session_snapshot("e").location, LOCAL_HBM)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 1280, 1: 0})

        # e 长成后余量 = 20(1280 恰好占用,重查停机的字节级证据)。
        self.assertTrue(manager.grow_prefill("e", 40, 51, "e0").admitted)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 20)
        manager.mark_complete("e", 52, "e0")

        # retire 静默核销:本地层域减账 + 远端账面清零。
        for session_id, completion_ns in (
            ("a", 60), ("b", 61), ("c", 62), ("d", 63), ("e", 64),
        ):
            manager.retire_terminal_session(
                session_id, completion_ns, f"{session_id}0")
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
        # "半层 a → 半层 b → 整体 a"(60→220→380→540 恰停)。
        manager = SessionKVCacheManager(
            _topology(480 + 700), _tiered_model(), strict_invariants=True)
        _seed_session(manager, "a", 10)
        _seed_session(manager, "b", 20)
        # 预约属于第三方请求(被保护会话为空):60 → 半层 a(220) →
        # 半层 b(380) → 整体 a(540)恰停;a REMOTE、b PARTIAL 混合终态。
        reservation = manager.reserve_request_capacity(
            "r0", "new_session", 0, (540,), 30,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation")
        self.assertTrue(reservation.admitted)
        self.assertEqual(
            [(t.session_id, t.reason.rsplit("_", 1)[-1])
             for t in reservation.evictions],
            [("a", "half"), ("b", "half"), ("a", "fallback")])
        self.assertEqual(
            manager.session_snapshot("a").location, REMOTE_MEMORY)
        self.assertEqual(
            manager.session_snapshot("b").location, PARTIAL_HBM_REMOTE)
        manager.release_request_capacity("r0", 40)

        # PARTIAL 跨实例两段链:b 的前缀 NoC 迁往实例 1 + 后缀远端回迁。
        decision = manager.prepare_history(
            "b", 1, 10, 50, "b1", required_context_tokens=10)
        self.assertEqual(decision.action, "PARTIAL_MIGRATE")
        self.assertEqual(
            [(t.kind, t.reason) for t in decision.transfers],
            [("noc_migrate", "history_partial_prefix_migrate"),
             ("remote_load", "history_remote_suffix_restore")])
        manager.grow_prefill("b", 10, 51, "b1")
        manager.mark_complete("b", 52, "b1")

        # REMOTE 全量回迁:a 回实例 0。
        decision = manager.prepare_history(
            "a", 0, 10, 60, "a1", required_context_tokens=10)
        self.assertEqual(decision.action, "REMOTE_RESTORE")
        self.assertEqual(
            [(t.kind, t.reason) for t in decision.transfers],
            [("remote_load", "history_remote_restore")])
        manager.grow_prefill("a", 10, 61, "a1")
        manager.mark_complete("a", 62, "a1")

        manager.retire_terminal_session("b", 63, "b1")
        manager.retire_terminal_session("a", 64, "a1")
        manager.assert_final_state()

        rows = [json.loads(line) for line in
                recorder.journal_path.open(encoding="utf-8") if line.strip()]
        causes = [row["cause"] for row in rows]
        # 新 mutation 的 cause 全部落账(契约 §5 逐字)。
        for expected in (
            "evict_static_decode_final_kv_reservation_suffix_half:layers2-4",
            "evict_static_decode_final_kv_reservation_full_fallback:layers0-2",
            "history_partial_prefix_migrate_target_add",
            "history_partial_prefix_migrate_source_remove",
            "history_remote_suffix_restore",
            "history_remote_restore",
            "terminal_session_retire",
        ):
            self.assertIn(expected, causes, msg=f"missing cause {expected}")
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
