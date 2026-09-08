#!/usr/bin/env python3
"""test_kv_delta_journal.py -- P1 权威 HBM delta journal 单元测试。

覆盖(doc §6-P1 验收要点):
  - 守恒矩阵:reserve/grow/move/evict/recompute/terminal retire 逐事务对账
    manager 快照(③/④ 路径:每事务 commit 点 _journal_reconcile_
    transaction 不抛 = 对账通过;run 末 verify_journal_checksum 全量重放
    通过且终态 resident=0/reserved=0/physical=weight);
  - ② 独立单调计数器:零 delta 记录被跳过(整 rank shard=0)时 sequence
    仍从 0 连续;journal 模式不驻留 deltas list(流式);
  - ① 时序护栏:release/mutation 传入非单调 now_ns 直接 fail-closed;
  - ④ 对账漂移:manager 状态与 journal 记账不一致时 commit 点 raise;
  - checksum 门:人为破坏 journal 一行(改 delta)或断行后重放报行号;
  - 正常路径 checksum 通过并写出 kv_delta_journal_checksum.json。

运行: 在 llama2_7b_inference 目录
    python3 -m pytest test_kv_delta_journal.py -q
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
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
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


def _build_topology(capacity_bytes: int):
    hardware = WscLlmHardware(
        mesh_rows=2,
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
            WscLlmInstanceSpec("prefill", "1", (0, 1), PREFILL_ROLE),
            WscLlmInstanceSpec("decode", "2", (2, 3), DECODE_ROLE),
        ),
    )


def _journal_rows(recorder: MemoryActionRecorder) -> list[dict]:
    with recorder.journal_path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class KvDeltaJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(set_metrics_observer, None)

    def _manager(
        self, *, capacity: int = 10_000, heads: int = 2
    ) -> tuple[SessionKVCacheManager, MemoryActionRecorder]:
        model = WscLlmModel(1, 4, 4, heads, 4, 1, "gelu")
        recorder = MemoryActionRecorder(
            MemoryMetricsObserver(4),
            journal_path=Path(self._tmp.name) / "kv_delta_journal.jsonl",
        )
        set_metrics_observer(recorder)
        manager = SessionKVCacheManager(
            _build_topology(capacity),
            model,
            strict_invariants=True,
        )
        return manager, recorder

    def _drain_full_lifecycle(
        self, manager: SessionKVCacheManager
    ) -> None:
        """reserve/grow/release/move/grow_decode/complete/terminal retire 全
        mutation 链(无逐出;逐出场景另测)。时刻严格非递减。"""
        tokens = 4
        shards = kv_cache_shard_bytes_for_tokens(
            manager.model, tokens, manager.tp_degree)
        manager.prepare_history(
            "sA", 0, 0, 10, "rA0", required_context_tokens=tokens)
        manager.reserve_request_capacity("rA0", "sA", 1, shards, 11)
        manager.grow_prefill("sA", tokens, 12, "rA0")
        manager.release_request_capacity("rA0", 13)
        manager.move_prefill_to_decode("sA", 1, 14, "rA0", final_context_tokens=8)
        manager.grow_decode("sA", 8, 15, "rA0")
        manager.mark_complete("sA", 16, "rA0")
        manager.retire_terminal_session("sA", 16, "rA0")

    # ------------------------------------------------- 守恒矩阵(③/④) --

    def test_conservation_matrix_full_lifecycle(self) -> None:
        manager, recorder = self._manager()
        self._drain_full_lifecycle(manager)
        manager.assert_final_state()
        summary = manager.verify_journal_checksum()

        rows = _journal_rows(recorder)
        # 行数 = 事务内 delta 总数,与 recorder 计数一致(流式不丢行)。
        self.assertEqual(summary["line_count"], recorder.record_count)
        self.assertEqual(summary["line_count"], len(rows))
        # sequence 从 0 连续(②)。
        self.assertEqual(
            [row["sequence"] for row in rows], list(range(len(rows))))
        # 前 4 行 = 权重预载(transaction_id=0,先于任何事务)。
        for row in rows[:4]:
            self.assertEqual(row["transaction_id"], 0)
            self.assertEqual(row["cause"], "model_weight_preload")
            self.assertGreater(row["weight_delta_bytes"], 0)
        # 其余行的事务 id 单调且各组连续;零行事务(prepare/mark_complete)
        # 不落任何行但消耗 id——max id = 事务计数。
        later = [row["transaction_id"] for row in rows[4:]]
        self.assertEqual(later, sorted(later))
        self.assertGreater(
            summary["max_transaction_id"], len(set(later)))
        self.assertEqual(
            summary["max_transaction_id"], recorder.transaction_count)
        # 每行前后快照自洽:after = before + delta(逐 rank 重放推导,B2:
        # 四列含 remote 远端池账面)。
        running: dict[int, list[int]] = {}
        for row in rows:
            state = running.setdefault(row["rank"], [0, 0, 0, 0])
            self.assertEqual(
                [row["before_bytes"]["weight"],
                 row["before_bytes"]["resident"],
                 row["before_bytes"]["reserved"],
                 row["before_bytes"]["remote"]], state)
            state[0] += row["weight_delta_bytes"]
            state[1] += row["resident_kv_delta_bytes"]
            state[2] += row["reserved_kv_delta_bytes"]
            state[3] += row["remote_delta_bytes"]
            self.assertEqual(
                [row["after_bytes"]["weight"],
                 row["after_bytes"]["resident"],
                 row["after_bytes"]["reserved"],
                 row["after_bytes"]["remote"]], state)
        # 终态守恒(B2 新口径):本地只剩权重,远端账面归零(与 manager
        # 会话账本守恒)。
        for rank_state in manager.node_states:
            replayed = summary["ranks"][rank_state.rank]
            self.assertEqual(replayed["resident"], 0)
            self.assertEqual(replayed["reserved"], 0)
            self.assertEqual(replayed["physical"], replayed["weight"])
            self.assertEqual(replayed["remote"], 0)
            self.assertEqual(
                replayed["weight"], rank_state.model_weight_bytes)
        # checksum 产物落盘且内容与返回一致。
        checksum_path = (
            recorder.journal_path.parent
            / MemoryActionRecorder.JOURNAL_CHECKSUM_NAME)
        self.assertTrue(checksum_path.exists())
        on_disk = json.loads(checksum_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["line_count"], summary["line_count"])
        self.assertEqual(on_disk["sha256"], summary["sha256"])
        # JSON 落盘后 rank 键为字符串;数值内容必须逐项一致。
        self.assertEqual(
            {int(rank): state for rank, state in on_disk["ranks"].items()},
            summary["ranks"])

    def test_evict_restore_and_terminal_retire_transactions(self) -> None:
        # 容量收到每 rank weight+20:一个 4-token 会话(16B)驻留后,第二个
        # 会话增长必然逐出前者(LRU completed inactive,B2 三态:整体外迁
        # remote_store;L=1 无半层后缀,阶段 1 空集直入阶段 2),随后被逐
        # 会话的下一 turn 走 REMOTE 全量回迁(再次逐出现驻留者)。
        weight = 72  # WscLlmModel(1,4,4,2,4,1,gelu) 的每 rank TP 权重分片
        manager, recorder = self._manager(capacity=weight + 20)
        manager.prepare_history(
            "sA", 0, 0, 10, "rA0", required_context_tokens=4)
        manager.grow_prefill("sA", 4, 11, "rA0")
        manager.mark_complete("sA", 12, "rA0")
        manager.prepare_history(
            "sB", 0, 0, 13, "rB0", required_context_tokens=4)
        manager.grow_prefill("sB", 4, 14, "rB0")  # 逐出 sA(remote_store)
        manager.mark_complete("sB", 15, "rB0")
        decision = manager.prepare_history("sA", 0, 4, 16, "rA1")  # 远端回迁
        self.assertEqual(decision.action, "REMOTE_RESTORE")
        manager.grow_prefill("sA", 4, 17, "rA1")  # 零 delta 增长(零行事务)
        manager.mark_complete("sA", 18, "rA1")
        manager.retire_terminal_session("sA", 18, "rA1")
        manager.retire_terminal_session("sB", 18, "rB0")
        manager.assert_final_state()

        rows = _journal_rows(recorder)
        causes = [row["cause"] for row in rows]
        # 契约 §5 cause 串逐字:外迁 full_fallback / 远端回迁 / 终局核销。
        self.assertTrue(any(
            "evict_history_and_prefill_admission_full_fallback" in cause
            for cause in causes))
        self.assertTrue(any(
            "evict_history_target_capacity_full_fallback" in cause
            for cause in causes))
        self.assertTrue(
            any("history_remote_restore" in cause for cause in causes))
        self.assertTrue(
            any("terminal_session_retire" in cause for cause in causes))
        # 远端列(B2 新增):外迁 +16、回迁 -16、核销兜底,重放终态归零。
        remote_rows = [row for row in rows if row["remote_delta_bytes"]]
        self.assertTrue(remote_rows)
        self.assertTrue(all("remote" in row["before_bytes"] for row in rows))
        # 零 delta 增长不落行但消耗事务 id:journal 不出现它,checksum 门
        # 通过且行数与写出计数一致。
        summary = manager.verify_journal_checksum()
        self.assertEqual(summary["line_count"], len(rows))
        for replayed in summary["ranks"].values():
            self.assertEqual(replayed["resident"], 0)
            self.assertEqual(replayed["reserved"], 0)
            self.assertEqual(replayed["physical"], replayed["weight"])
            self.assertEqual(replayed["remote"], 0)

    # ------------------------------------------------------- ② 计数器 --

    def test_sequence_contiguous_with_zero_shard_rank(self) -> None:
        # num_heads=1 / tp=2 -> rank1 恒 0 头,KV 分片恒 0:manager 对该 rank
        # 跳过 record(零 delta 记录),sequence 必须仍然连续。
        manager, recorder = self._manager(heads=1)
        manager.prepare_history(
            "sA", 0, 0, 10, "rA0", required_context_tokens=4)
        manager.grow_prefill("sA", 4, 11, "rA0")
        manager.mark_complete("sA", 12, "rA0")
        manager.retire_terminal_session("sA", 12, "rA0")
        manager.assert_final_state()
        rows = _journal_rows(recorder)
        self.assertEqual(
            [row["sequence"] for row in rows], list(range(len(rows))))
        zero_rank = manager.topology.instance(0).ranks[1]
        zero_rank_rows = [row for row in rows if row["rank"] == zero_rank]
        # 该 rank 仅有构造期权重预载一行,无任何 KV 行(零 delta 跳过)。
        self.assertEqual(len(zero_rank_rows), 1)
        self.assertEqual(zero_rank_rows[0]["cause"], "model_weight_preload")
        self.assertIsNotNone(manager.verify_journal_checksum())

    def test_journal_mode_streams_without_retaining_deltas(self) -> None:
        manager, recorder = self._manager()
        # 孪生内存模式 recorder:同一 mutation 序列下,journal 模式的流式
        # 回读必须与改前内存 deltas 逐字段等价(manifest memory_actions 的
        # 流式重建来源)。
        twin = MemoryActionRecorder(MemoryMetricsObserver(4))
        set_metrics_observer(twin)
        twin_model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        twin_manager = SessionKVCacheManager(
            _build_topology(10_000), twin_model,
            strict_invariants=True)
        set_metrics_observer(recorder)
        self._drain_full_lifecycle(manager)
        set_metrics_observer(twin)
        self._drain_full_lifecycle(twin_manager)
        # journal 模式不驻留 deltas list(流式);回读流可重建等价行。
        self.assertEqual(recorder.deltas, [])
        replayed = list(recorder.iter_journal_deltas())
        self.assertEqual(len(replayed), recorder.record_count)
        self.assertEqual(
            [delta.sequence_index for delta in replayed],
            list(range(len(replayed))))
        self.assertEqual([delta.to_dict() for delta in replayed],
                         [delta.to_dict() for delta in twin.deltas])

    # --------------------------------------------------- ① 时序护栏 --

    def test_non_monotonic_mutation_timestamp_fails_closed(self) -> None:
        manager, recorder = self._manager()
        shards = kv_cache_shard_bytes_for_tokens(
            manager.model, 4, manager.tp_degree)
        manager.prepare_history(
            "sA", 0, 0, 100, "rA0", required_context_tokens=4)
        manager.reserve_request_capacity("rA0", "sA", 1, shards, 100)
        with self.assertRaisesRegex(
            RuntimeError, "planner_time_ns went backwards"
        ):
            # 改前行为:release 复用最近事件时间(静默);P1 后非单调即
            # fail-closed raise,禁止钳位。
            manager.release_request_capacity("rA0", 50)

    # ------------------------------------------------- ④ commit 对账 --

    def test_transaction_reconciliation_detects_manager_drift(self) -> None:
        manager, _ = self._manager()
        manager.prepare_history(
            "sA", 0, 0, 10, "rA0", required_context_tokens=4)
        manager.grow_prefill("sA", 4, 11, "rA0")
        manager.mark_complete("sA", 12, "rA0")
        # 破坏 manager 侧权重记账 1 字节:增量不变量只对账 resident/
        # reserved 对 sessions/reservations,不交叉核对 weight,故不会在
        # _check_invariants_after_mutation 处先抛——由 ④ 逐 rank 对账抓出。
        rank = manager.topology.instance(0).ranks[0]
        manager._rank_states[rank].model_weight_bytes += 1
        shards = kv_cache_shard_bytes_for_tokens(
            manager.model, 4, manager.tp_degree)
        with self.assertRaisesRegex(
            RuntimeError, "reconciliation failed for transaction"
        ):
            manager.reserve_request_capacity("rB0", "sB", 0, shards, 13)

    # --------------------------------------------- checksum 门(破坏) --

    def test_checksum_gate_detects_tampered_delta(self) -> None:
        manager, recorder = self._manager()
        self._drain_full_lifecycle(manager)
        manager.verify_journal_checksum()
        # 人为篡改第 2 行的 resident delta:行内 after != before+delta,
        # 重放必须报该行号(fail-closed,不静默跳过)。
        lines = recorder.journal_path.read_text(
            encoding="utf-8").splitlines()
        row = json.loads(lines[1])
        row["resident_kv_delta_bytes"] += 8
        lines[1] = json.dumps(row, sort_keys=True)
        recorder.journal_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(
            RuntimeError, "kv delta journal line 2"
        ):
            manager.verify_journal_checksum()

    def test_checksum_gate_detects_torn_tail_line(self) -> None:
        manager, recorder = self._manager()
        self._drain_full_lifecycle(manager)
        raw = recorder.journal_path.read_text(encoding="utf-8")
        # 模拟崩溃断行:截掉最后一行的一半。
        recorder.journal_path.write_text(
            raw[: len(raw) - 10], encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "is corrupt"):
            manager.verify_journal_checksum()

    # --------------------------------------------------- 正常路径门 --

    def test_checksum_gate_passes_and_writes_summary(self) -> None:
        manager, recorder = self._manager()
        self._drain_full_lifecycle(manager)
        summary = manager.verify_journal_checksum()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["checks"]["manager_state_match"], True)
        self.assertEqual(summary["checks"]["residual_resident_zero"], True)
        self.assertEqual(summary["checks"]["residual_reserved_zero"], True)
        self.assertEqual(summary["checks"]["physical_equals_weight"], True)
        self.assertEqual(len(summary["ranks"]), len(manager.node_states))

    def test_journal_disabled_recorder_keeps_legacy_behavior(self) -> None:
        # journal 关闭(内存模式 recorder):行为与改前一致——deltas 驻留、
        # sequence 仍连续;verify_journal_checksum 对非 journal recorder
        # 返回 None(门只对 journal-on 的 run 生效)。
        recorder = MemoryActionRecorder(MemoryMetricsObserver(4))
        set_metrics_observer(recorder)
        model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
        manager = SessionKVCacheManager(
            _build_topology(10_000), model,
            strict_invariants=True)
        self._drain_full_lifecycle(manager)
        manager.assert_final_state()
        self.assertGreater(len(recorder.deltas), 0)
        self.assertEqual(
            [delta.sequence_index for delta in recorder.deltas],
            list(range(len(recorder.deltas))))
        self.assertIsNone(manager.verify_journal_checksum())


if __name__ == "__main__":
    unittest.main()
