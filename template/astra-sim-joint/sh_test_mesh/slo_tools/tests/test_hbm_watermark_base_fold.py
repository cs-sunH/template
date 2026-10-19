#!/usr/bin/env python3
"""test_hbm_watermark_base_fold.py -- W4（2026-09-27）钉测：joint D 边界
就地并基折叠的 base 槽清账重放（词表补齐）。

素材口径（schema 逐符号锚定发射面，字段名/reason 不发明新口径；
2026-10-14 远读退役：增量型折叠（W+base==f(P) 的增量工作副本形态）
的用例随远读退役整删，保留完整型折叠 + 在册互斥守卫；2026-10-15
recompute 退役：重放侧 increment 折叠支与其计数器随五轮审查整删——
本文件只钉完整型存活面）：
  * 折叠轮 decode 行 prefill_decode_transfer = noc_migrate（source=i_P、
    target=base 驻留实例、total_bytes=迁移腿字节）——落点 == base 槽即
    workload 折叠形态（complete_copy_base_fold 发射面同此披露形）；
  * 折叠轮 completion 行 merge_direction=stay、joint_working_copy=False
    （折叠后 working_kind 清空 → 本地提交）；
  * 折叠清账后下一轮跨实例 prefill 重新 stash base = 合法重建。

手算场景（coef=100 B/token：layers=1/hidden=50/bpe=1，与既有合成口径
一致）：
  * 完整型折叠：base f(12)@0 + 工作副本 f(20)@1 → 迁移腿 2000==f(P)
    落 0 → 重复基础副本释放（−1200@0）→ 再 stash（copy 轮）→ reverse
    合并闭合（copy 执行端持并集、base 释放），终态 f(34)@1。
负例反向构造：真重复 stash（无折叠清账，在册守卫不放松）、折叠清账
缺席的 stay 完成（在册互斥守卫），各必须 fail-closed（exit 2）。

运行：python3 -m pytest slo_tools/tests/test_hbm_watermark_base_fold.py
      -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SLO_TOOLS_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SLO_TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

from test_hbm_watermark import (  # noqa: E402
    make_run_dir, run_tool, token_row, write_fixture)

COEF = 100  # B/token
S9 = "session_9"
R0, R1, R2 = "s9_r0", "s9_r1", "s9_r2"

_SEQ = 0


def _nxt() -> int:
    global _SEQ
    _SEQ += 1
    return _SEQ


def _migrate_transfer(source: int, target: int, total: int) -> dict:
    """decode 行迁移腿（noc_migrate，reason=prefill_decode_instance_
    migrate——move_prefill_to_decode 的既有披露口径）。"""
    return {"kind": "noc_migrate", "phase": "prefill_decode",
            "reason": "prefill_decode_instance_migrate",
            "session_id": S9, "source_instance_index": source,
            "target_instance_index": target,
            "layer_start": 0, "layer_end": 1,
            "resident_prefix_layers_before": 1,
            "resident_prefix_layers_after": 1,
            "total_bytes": total, "shards": []}


def _local_hit(instance: int, total: int = 0,
               reason: str = "prefill_decode_local_reuse") -> dict:
    return {"kind": "local_hit", "phase": "prefill_decode",
            "reason": reason, "session_id": S9,
            "source_instance_index": instance,
            "target_instance_index": instance,
            "layer_start": 0, "layer_end": 1,
            "resident_prefix_layers_before": 1,
            "resident_prefix_layers_after": 1,
            "total_bytes": total, "shards": []}


def _prefill(request_id, tick, instance, action, transfers=()):
    return {"kind": "prefill", "request_id": request_id, "priority": 0,
            "seq": _nxt(), "tick": tick,
            "decision": {"prefill_instance_index": instance,
                         "joint_action": action,
                         "history_tokens_discarded": 0,
                         "history_evictions": [],
                         "history_transfers": list(transfers),
                         "history_transfer_bytes": 0,
                         "history_source_instance_index": None}}


def _decode(request_id, tick, instance, action, transfer):
    return {"kind": "decode", "request_id": request_id, "priority": 0,
            "seq": _nxt(), "tick": tick,
            "decision": {"decode_instance_index": instance,
                         "joint_action": action,
                         "prefill_decode_transfer": transfer,
                         "decode_evictions": []}}


def _completion(request_id, tick, *, direction, working_copy, merges=(),
                origin_home=0, action="copy", transferred=None,
                home_flipped=None):
    decision = {"joint_action": action,
                "merge_direction": direction,
                "joint_working_copy": working_copy,
                "merge_transfers": list(merges),
                "home_flipped_to": home_flipped,
                "origin_home_instance": origin_home,
                "kv_location_after_completion": "local_hbm",
                "completion_evictions": []}
    if transferred is not None:
        decision["merge_transferred_bytes"] = transferred
    return {"kind": "completion", "request_id": request_id, "priority": 0,
            "seq": _nxt(), "tick": tick, "decision": decision}


def run_case(tag: str, records, tokens, mutate=None):
    if mutate is not None:
        records = mutate(copy.deepcopy(records))
    run_dir = make_run_dir(tag)
    write_fixture(run_dir, repo_variant="astra-sim-joint",
                  records=records, token_requests=tokens,
                  layers=1, hidden_size=50, bytes_per_elem=1,
                  npu_bytes=1 << 40)
    proc = run_tool(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) \
        if summary_path.exists() else {}
    return proc, summary


class CompleteFoldThenRestashTests(unittest.TestCase):
    """完整型折叠（迁移腿 W == f(P)）：重复基础副本释放 + 再 stash 合法。

    手算时间线（delta@实例，coef=100）：
      I0: +1200 (r0 stay) | r1 折叠 tick：+2000 (迁移腿) −1200 (基础副本
          释放) +400 (decode 增长) → 2400 | r2 reverse：−2400 (base 释放)
          → 0，峰 3200（折叠 tick 内 base+迁移腿同刻瞬态）
      I1: +1200 (r1 历史腿) +800 (长到 f(P)) → 2000 | −2000 (迁移腿)
          | +2400 (r2 历史腿) +600 (长到 f(P)) | +400 (decode 增长)
          → 3400，峰 3400
    """

    def _records(self):
        return [
            # r0：stay-prefill@0 → f(12)=1200 驻留 home。
            _prefill(R0, 100, 0, "stay"),
            _decode(R0, 110, 0, "stay", _local_hit(0)),
            _completion(R0, 120, direction="stay", working_copy=False,
                        action="stay"),
            # r1：copy 工作副本@1（历史腿 1200 + 长到 f(20)=2000）→ D
            # 边界迁移腿 2000==f(P) 落 base 驻留实例 0 → 完整型折叠
            # （重复基础副本 1200 释放）→ stay 本地提交。
            _prefill(R1, 200, 1, "copy", transfers=[{
                "kind": "noc_migrate", "total_bytes": 12 * COEF,
                "source_instance_index": 0,
                "layer_start": 0, "layer_end": 1}]),
            _decode(R1, 210, 0, "copy", _migrate_transfer(1, 0, 20 * COEF)),
            _completion(R1, 220, direction="stay", working_copy=False),
            # r2：折叠清账后重新 stash base（copy@1 全量工作副本：历史
            # 腿 2400 + 长到 f(30)）→ decode 增长至 f(34) → reverse 合并
            # （copy 执行端持并集、base@0 释放、home 迁 1）。
            _prefill(R2, 300, 1, "copy", transfers=[{
                "kind": "noc_migrate", "total_bytes": 24 * COEF,
                "source_instance_index": 0,
                "layer_start": 0, "layer_end": 1}]),
            _decode(R2, 310, 1, "copy", _local_hit(1)),
            _completion(R2, 320, direction="reverse", working_copy=True,
                        action="copy", home_flipped=1),
        ]

    def _tokens(self):
        return [
            token_row(R0, S9, 0, 12, 12),
            token_row(R1, S9, 12, 20, 24),
            token_row(R2, S9, 24, 30, 34),
        ]

    def test_complete_fold_replay_closes_and_restash_legal(self):
        """主线：stash → 完整型折叠清账 → 再 stash → forward 合并闭合；
        逐实例残差/峰值与动作计数逐字段手算断言。"""
        proc, summary = run_case("w4cf1", self._records(), self._tokens())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({k: v for k, v in summary["anomalies"].items()
                          if v}, {})
        instances = summary["instances"]
        self.assertEqual(instances["0"]["residual_occupancy_bytes"], 0)
        # 峰 = 折叠 tick 内瞬态（base 1200 + 迁移腿 2000 在释放前同刻）。
        self.assertEqual(instances["0"]["peak_occupancy_bytes"], 32 * COEF)
        self.assertEqual(instances["1"]["residual_occupancy_bytes"],
                         34 * COEF)
        self.assertEqual(instances["1"]["peak_occupancy_bytes"], 34 * COEF)
        actions = summary["actions"]
        # 折叠清账：完整型 1 例，释放重复基础副本 1200。
        self.assertEqual(actions["joint_boundary_base_fold"], 1)
        self.assertEqual(actions["joint_boundary_base_fold_release_bytes"],
                         12 * COEF)
        # 迁移腿（r1）与两位置 reverse（r2，零字节翻转）不受折叠分支影响。
        self.assertEqual(actions["joint_decode_move"], 1)
        self.assertEqual(actions["joint_decode_move_bytes"], 20 * COEF)
        self.assertEqual(actions["joint_merge_reverse"], 1)
        self.assertEqual(actions["joint_merge_stay"], 2)
        self.assertEqual(actions["joint_merge_settle"], 1)

    def test_restash_without_fold_still_fails_closed(self):
        """真重复 stash 负例：两轮 stash 之间无折叠清账（r1 decode 原地
        local_hit、r1 completion 不落盘）→ r2 跨实例 prefill 再 stash 撞
        在册守卫 fail-closed（在册守卫纪律不因折叠词表放松）。"""

        def mutate(records):
            out = []
            for row in records:
                if row["request_id"] == R1 and row["kind"] == "decode":
                    # 原地 local_hit@1：无迁移腿 → 无折叠清账。
                    row["decision"]["decode_instance_index"] = 1
                    row["decision"]["prefill_decode_transfer"] = _local_hit(1)
                if row["request_id"] == R1 and row["kind"] == "completion":
                    continue  # 不落盘，让扫描直击 r2 prefill 的 stash 点
                if row["request_id"] == R2 and row["kind"] in (
                        "decode", "completion"):
                    continue
                if row["request_id"] == R2 and row["kind"] == "prefill":
                    # 跨实例（1→0）触发第二次 stash。
                    row["decision"]["prefill_instance_index"] = 0
                    row["decision"]["joint_action"] = "copy"
                out.append(row)
            return out

        proc, _ = run_case("w4cf2", self._records(), self._tokens(), mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("joint base stash on already-stashed session",
                      proc.stderr)

    def test_stay_with_pending_base_fails_closed(self):
        """折叠清账缺席的 stay 完成（r1 decode 原地 local_hit、base 槽
        仍在册）→ 在册互斥守卫 fail-closed（宁可早爆，账本定位更准）。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R1 and row["kind"] == "decode":
                    row["decision"]["decode_instance_index"] = 1
                    row["decision"]["prefill_decode_transfer"] = _local_hit(1)
            return records

        proc, _ = run_case("w4if3", self._records(), self._tokens(), mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("merge_direction=stay 但 base 槽在册", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
