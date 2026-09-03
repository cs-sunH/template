#!/usr/bin/env python3
"""kv_cache_adapter relevant_distributed 分支契约测试（T3/2026-09-02）。

覆盖（总文档 §4 裁决 #23/#24、§3.4 canonical 映射）：
  * hit_state：full/partial/no_history/not_supported 四态 + 无 miss 枚举
    路径（本策略历史必经 1000 拉回复用）；
  * 仓内 policy 分发：relevant 决策行（history_canonical_hit_state 字段
    在场）走新映射，legacy/session_lru 行为与改动前逐字节一致；
  * cache_events：1000 拉回（history_pull_routes 逐 rank 行）按源实例
    聚合 → history_transfer；kv_scatter 行 routes（3100）按 owner 聚合 →
    prefill_decode_transfer；3300 读边/逐出不产事件；
  * --reconcile：native 重放（路由行聚合）与 canonical CSV 逐 family
    对账一致；canonical 侧被篡改时 fail-closed。

运行：python3 sh_test_mesh/slo_tools/tests/test_kv_adapter_relevant.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

import synthetic  # noqa: E402
from slo_common import SloToolError  # noqa: E402
import kv_cache_adapter  # noqa: E402


def _adapter_args(run_dir: Path, **overrides):
    ns = argparse.Namespace(
        run_dir=run_dir, output=str(run_dir / "cache_events.csv"),
        hit_states=str(run_dir / "kv_hit_states.csv"),
        json=str(run_dir / "kv_cache_adapter.json"), reconcile=False,
        repo_variant=None, request_manifest=None)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _relevant_records() -> list[dict]:
    """两请求 relevant_distributed 决策日志（r0 turn-0 / r1 turn-1）。

    r1 拉回两源：源 3（两 rank 路由行 10+15=25B）与源 2（5B）→ canonical
    两条 history_transfer 事件（键升序：源 2 先）；r0 散布两 owner：D0
    40B 与 d4 7B → 两条 prefill_decode_transfer 事件。3300 读边行不产
    cache 事件。
    """
    return [
        {"kind": "run_header", "request_id": "", "tick": 1,
         "decision": {"kv_cache_policy": "relevant_distributed",
                      "kv_remote_read": "physical",
                      "d2d_to_hbm_bandwidth_ratio": 2.5,
                      "sh_train_max_iter": 8,
                      "sh_first_token_split": "off"}},
        {"kind": "kv_placement", "request_id": "r0", "tick": 10,
         "decision": {"decode_instance_index": 0,
                      "backpressure_duration_ns": None}},
        {"kind": "prefill", "request_id": "r0", "tick": 10,
         "decision": {
             "history_action": "NO_HISTORY",
             "history_canonical_hit_state": None,
             "history_local_hit_tokens": 0,
             "history_remote_transfer_bytes": 0,
             "history_transfer_bytes": 0,
             "history_pull_sources": [],
             "history_pull_routes": []}},
        {"kind": "kv_scatter", "request_id": "r0", "tick": 20,
         "decision": {"prefill_instance_index": 1,
                      "routes": [
                          {"category": 3100, "source_instance_index": 1,
                           "target_instance_index": 0,
                           "relative_shard": 0, "bytes": 30, "noc_hops": 1},
                          {"category": 3100, "source_instance_index": 1,
                           "target_instance_index": 0,
                           "relative_shard": 1, "bytes": 10, "noc_hops": 1},
                          {"category": 3100, "source_instance_index": 1,
                           "target_instance_index": 4,
                           "relative_shard": 0, "bytes": 7, "noc_hops": 3}],
                      "scatter_bytes": 47}},
        {"kind": "kv_remote_reads", "request_id": "train_i0_1", "tick": 30,
         "decision": {"train_id": "train_i0_1", "instance_index": 0,
                      "routes": [
                          {"category": 3300, "train_id": "train_i0_1",
                           "request_id": "r0",
                           "source_instance_index": 1,
                           "decode_instance_index": 0, "bytes": 999,
                           "noc_hops": 1}],
                      "total_bytes": 999}},
        {"kind": "kv_placement", "request_id": "r1", "tick": 100,
         "decision": {"decode_instance_index": 0,
                      "backpressure_duration_ns": 500}},
        {"kind": "prefill", "request_id": "r1", "tick": 100,
         "decision": {
             "history_action": "NO_HISTORY",
             "history_canonical_hit_state": "partial",
             "history_local_hit_tokens": 6,
             "history_remote_transfer_bytes": 30,
             "history_transfer_bytes": 36,
             "history_pull_sources": [
                 {"source_instance_index": 2, "source_ordinal": 0,
                  "tokens": 1, "bytes": 5},
                 {"source_instance_index": 3, "source_ordinal": 1,
                  "tokens": 5, "bytes": 25}],
             "history_pull_routes": [
                 {"category": 1000, "source_instance_index": 3,
                  "target_instance_index": 1, "relative_shard": 0,
                  "bytes": 10, "noc_hops": 2},
                 {"category": 1000, "source_instance_index": 3,
                  "target_instance_index": 1, "relative_shard": 1,
                  "bytes": 15, "noc_hops": 2},
                 {"category": 1000, "source_instance_index": 2,
                  "target_instance_index": 1, "relative_shard": 0,
                  "bytes": 5, "noc_hops": 1}]}},
        {"kind": "decode", "request_id": "r0", "tick": 30,
         "decision": {"prefill_decode_transfer": None}},
        {"kind": "completion", "request_id": "r0", "tick": 40,
         "decision": {"completion_evictions": []}},
    ]


class RelevantHitStateMappingTests(unittest.TestCase):
    """canonical hit_state 映射（总文档 §3.4：全源=P→full；否则 partial；
    永不 miss）。"""

    def test_full_partial_no_history(self):
        f = kv_cache_adapter._hit_state_wscllm_relevant
        self.assertEqual(
            f({"history_canonical_hit_state": "full"}, 1, "r")[0], "full")
        self.assertEqual(
            f({"history_canonical_hit_state": "partial"}, 2, "r")[0],
            "partial")
        self.assertEqual(
            f({"history_canonical_hit_state": None}, 0, "r")[0],
            "no_history")

    def test_never_miss_and_fail_closed(self):
        f = kv_cache_adapter._hit_state_wscllm_relevant
        # 映射表封闭集合：未知值/turn>0 缺字段 → not_supported，不猜 miss。
        self.assertEqual(
            f({"history_canonical_hit_state": "RECOMPUTE"}, 1, "r")[0],
            "not_supported")
        self.assertEqual(
            f({"history_canonical_hit_state": None}, 1, "r")[0],
            "not_supported")
        self.assertEqual(
            f({"history_canonical_hit_state": None}, None, "r")[0],
            "not_supported")

    def test_policy_dispatch_inside_repo_entry(self):
        f = kv_cache_adapter._hit_state_wscllm
        # relevant 行（字段在场）→ 新映射；legacy/session_lru 行 → face
        # 口径不变（history_action 四值）。
        self.assertEqual(
            f({"history_canonical_hit_state": "full",
               "history_action": "NO_HISTORY"}, 1, "r")[0], "full")
        self.assertEqual(
            f({"history_action": "NO_HISTORY"}, 0, "r")[0], "no_history")
        self.assertEqual(
            f({"history_action": "NOC_MIGRATE"}, 1, "r")[0], "full")
        self.assertEqual(
            f({"history_action": "RECOMPUTE"}, 1, "r")[0], "miss")


class RelevantEventExtractionTests(unittest.TestCase):
    def test_routes_aggregate_to_instance_level_events(self):
        events = kv_cache_adapter.extract_events_wscllm_relevant(
            {"kind": "prefill", "request_id": "r1", "tick": 100,
             "decision": {"history_pull_routes": [
                 {"source_instance_index": 3, "target_instance_index": 1,
                  "bytes": 10},
                 {"source_instance_index": 3, "target_instance_index": 1,
                  "bytes": 15},
                 {"source_instance_index": 2, "target_instance_index": 1,
                  "bytes": 5}]}})
        self.assertEqual(
            [(e["source"], e["target"], e["bytes"], e["cause"])
             for e in events],
            [(2, 1, 5, "history_transfer:history_pull"),
             (3, 1, 25, "history_transfer:history_pull")])

    def test_scatter_and_no_event_kinds(self):
        scatter = kv_cache_adapter.extract_events_wscllm_relevant(
            {"kind": "kv_scatter", "request_id": "r0", "tick": 20,
             "decision": {"routes": [
                 {"source_instance_index": 1,
                  "target_instance_index": 0, "bytes": 40},
                 {"source_instance_index": 1,
                  "target_instance_index": 4, "bytes": 7}]}})
        self.assertEqual(
            [(e["source"], e["target"], e["bytes"], e["cause"])
             for e in scatter],
            [(1, 0, 40, "prefill_decode_transfer:scatter"),
             (1, 4, 7, "prefill_decode_transfer:scatter")])
        # 3300 读边/位置表/run 头不产 cache 事件（读边归观测后处理，
        # 总文档裁决 #24）。
        for kind in ("kv_remote_reads", "kv_placement", "run_header"):
            self.assertEqual(
                kv_cache_adapter.extract_events_wscllm_relevant(
                    {"kind": kind, "request_id": "x", "tick": 1,
                     "decision": {"routes": [
                         {"source_instance_index": 1,
                          "target_instance_index": 0, "bytes": 999}]}}),
                [])

    def test_dispatch_wrapper_keeps_legacy_shape(self):
        # legacy prefill 行（无 history_pull_routes）→ face 提取（聚合
        # history_transfer_bytes 一条事件）。
        legacy = kv_cache_adapter.extract_events_wscllm(
            {"kind": "prefill", "request_id": "r0", "tick": 5,
             "decision": {"history_transfer_bytes": 42,
                          "history_source_instance_index": 2,
                          "prefill_instance_index": 1,
                          "history_action": "NOC_MIGRATE"}})
        self.assertEqual(
            [(e["bytes"], e["source"], e["target"], e["cause"])
             for e in legacy],
            [(42, 2, 1, "history_transfer:history_action=NOC_MIGRATE")])


class RelevantAdapterCliTests(unittest.TestCase):
    """端到端：CLI 路径 hit_states/cache_events 产物 + --reconcile 对账。"""

    def _make_run(self, tag: str) -> Path:
        run_dir = synthetic.make_run_dir(tag)
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-wscllm")
        synthetic.write_jsonl(
            run_dir, "online_decision_log.jsonl", _relevant_records())
        synthetic.write_metrics_manifest(
            run_dir,
            [synthetic.manifest_request("r0", "s0", 0, 0),
             synthetic.manifest_request("r1", "s0", 1, 1)],
            repo_variant="astra-sim-wscllm")
        return run_dir

    def test_cli_outputs_and_reconcile(self):
        run_dir = self._make_run("kvrel1")
        self.assertEqual(
            kv_cache_adapter.cmd_adapter(_adapter_args(run_dir)), 0)
        header, rows = synthetic.read_csv(
            run_dir / "kv_hit_states.csv")
        self.assertEqual(header, list(
            kv_cache_adapter.KV_HIT_STATE_COLUMNS))
        states = {row["request_id"]: row["kv_hit_state"] for row in rows}
        # r0 turn-0 → no_history；r1 turn-1 拉回两源 → partial（无 miss）。
        self.assertEqual(states, {"r0": "no_history", "r1": "partial"})
        header, rows = synthetic.read_csv(run_dir / "cache_events.csv")
        self.assertEqual(header, list(kv_cache_adapter.CACHE_EVENT_COLUMNS))
        # 事件序：r0 两条 3100（按 owner 升序）→ r1 两条 1000（按源升序）。
        self.assertEqual(
            [(row["request_id"], row["bytes"], row["source"],
              row["target"], row["cause"]) for row in rows],
            [("r0", "40", "1", "0", "prefill_decode_transfer:scatter"),
             ("r0", "7", "1", "4", "prefill_decode_transfer:scatter"),
             ("r1", "5", "2", "1", "history_transfer:history_pull"),
             ("r1", "25", "3", "1", "history_transfer:history_pull")])
        summary = json.loads(
            (run_dir / "kv_cache_adapter.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["kv_hit_state_counts"],
                         {"full": 0, "partial": 1, "miss": 0,
                          "no_history": 1, "not_supported": 0})
        self.assertEqual(summary["hit_rate_over_all_requests"], 0.5)
        # relevant 行出现 → native partial 语义汇总字段置 True。
        self.assertTrue(summary["partial_semantics_native"])
        # --reconcile：native（路由行聚合重放）== canonical，exit 0。
        self.assertEqual(
            kv_cache_adapter.cmd_adapter(
                _adapter_args(run_dir, reconcile=True)), 0)

    def test_reconcile_fails_on_tampered_canonical(self):
        run_dir = self._make_run("kvrel2")
        self.assertEqual(
            kv_cache_adapter.cmd_adapter(_adapter_args(run_dir)), 0)
        events_path = run_dir / "cache_events.csv"
        with events_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        rows[1][4] = str(int(rows[1][4]) + 1)  # 篡改首条 3100 字节
        with events_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        with self.assertRaises(SloToolError):
            kv_cache_adapter.reconcile(
                run_dir=run_dir, repo_variant="astra-sim-wscllm",
                variant=kv_cache_adapter.REPO_VARIANTS["astra-sim-wscllm"],
                out_path=events_path, summary={})

    def test_legacy_run_unaffected(self):
        """legacy/session_lru 决策日志经同一仓入口：行为与改动前一致。"""
        run_dir = synthetic.make_run_dir("kvleg1")
        synthetic.write_cpp_log(run_dir, [], repo_variant="astra-sim-wscllm")
        synthetic.write_jsonl(run_dir, "online_decision_log.jsonl", [
            {"kind": "prefill", "request_id": "r0", "tick": 5,
             "decision": {"history_action": "NOC_MIGRATE",
                          "history_transfer_bytes": 42,
                          "history_source_instance_index": 2,
                          "prefill_instance_index": 1,
                          "admission_evictions": [],
                          "decode_target_evictions": []}},
        ])
        synthetic.write_metrics_manifest(
            run_dir, [synthetic.manifest_request("r0", "s0", 0, 0)],
            repo_variant="astra-sim-wscllm")
        self.assertEqual(
            kv_cache_adapter.cmd_adapter(_adapter_args(run_dir)), 0)
        summary = json.loads(
            (run_dir / "kv_cache_adapter.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["kv_hit_state_counts"]["full"], 1)
        self.assertFalse(summary["partial_semantics_native"])
        header, rows = synthetic.read_csv(run_dir / "cache_events.csv")
        self.assertEqual(
            [(row["bytes"], row["cause"]) for row in rows],
            [("42", "history_transfer:history_action=NOC_MIGRATE")])


if __name__ == "__main__":
    unittest.main()
