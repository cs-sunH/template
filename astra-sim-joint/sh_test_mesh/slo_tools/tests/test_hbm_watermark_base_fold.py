#!/usr/bin/env python3
"""test_hbm_watermark_base_fold.py -- W4（2026-09-27）钉测：joint D 边界
就地并基折叠的 base 槽清账重放（词表补齐）。

素材口径（schema 逐符号锚定发射面，字段名/reason 不发明新口径）：
  * 折叠轮 decode 行 prefill_decode_transfer = noc_migrate（source=i_P、
    target=base 驻留实例、total_bytes=迁移腿字节）——落点 == base 槽即
    workload 折叠形态（complete_copy_base_fold / remote_read_base_fold /
    increment_copy_home_u_only 三发射面同此披露形）；
  * 折叠轮 completion 行 merge_direction=stay、joint_working_copy=False
    （折叠后 working_kind 清空 → 本地提交）；
  * 折叠清账后下一轮跨实例 prefill 重新 stash base = 合法重建。

手算场景（coef=100 B/token：layers=1/hidden=50/bpe=1，与既有合成口径
一致）：
  * 完整型折叠：base f(12)@0 + 工作副本 f(20)@1 → 迁移腿 2000==f(P)
    落 0 → 重复基础副本释放（−1200@0）→ 再 stash（remote-read 轮）→
    forward 合并闭合，终态 f(34)@0 守恒；
  * 增量型折叠：base f(10)@0 + 增量工作副本 f(U)=1000@1 → 迁移腿
    1000+base 1000==f(P)=2000 → 前缀就地转正（占用不动、bytes 并至
    f(P)），decode 增长切完整上下文族 f(F)。
负例反向构造：真重复 stash（无折叠清账，在册守卫不放松）、折叠字节
恒等式破坏（未演练词表）、折叠清账缺席的 stay 完成（在册互斥守卫），
各必须 fail-closed（exit 2）。

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
          释放) +400 (decode 增长) → 2400 | +1000 (r2 forward 合并) → 3400
          （= f(34) 守恒），峰 3400（折叠 tick 内瞬态 3200 < 3400）
      I1: +1200 (r1 历史腿) +800 (长到 f(P)) → 2000 | −2000 (迁移腿)
          | +600 (r2 增量起步) +400 (decode 增长) → 1000 | −1000 (合并)
          → 0，峰 2000
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
            # r2：折叠清账后重新 stash base（remote-read@1 增量 600）→
            # decode 增长 1000 → forward 合并回 home（腿 1000）。
            _prefill(R2, 300, 1, "remote-read"),
            _decode(R2, 310, 1, "remote-read",
                    _local_hit(1, 10 * COEF, "boundary_remote_continue")),
            _completion(R2, 320, direction="forward", working_copy=True,
                        merges=[{"kind": "noc_migrate",
                                 "phase": "completion",
                                 "reason": "merge_working_copy_to_home",
                                 "session_id": S9,
                                 "source_instance_index": 1,
                                 "layer_start": 0, "layer_end": 1,
                                 "total_bytes": 10 * COEF, "shards": []}],
                        action="remote-read", transferred=10 * COEF),
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
        self.assertEqual(instances["0"]["residual_occupancy_bytes"],
                         34 * COEF)
        self.assertEqual(instances["0"]["peak_occupancy_bytes"], 34 * COEF)
        self.assertEqual(instances["1"]["residual_occupancy_bytes"], 0)
        self.assertEqual(instances["1"]["peak_occupancy_bytes"], 20 * COEF)
        actions = summary["actions"]
        # 折叠清账：完整型 1 例，释放重复基础副本 1200。
        self.assertEqual(actions["joint_boundary_base_fold"], 1)
        self.assertEqual(actions["joint_boundary_base_fold_release_bytes"],
                         12 * COEF)
        self.assertEqual(actions.get(
            "joint_boundary_base_fold_absorb_bytes", 0), 0)
        # 迁移腿（r1）与两位置 forward（r2）不受折叠分支影响。
        self.assertEqual(actions["joint_decode_move"], 1)
        self.assertEqual(actions["joint_decode_move_bytes"], 20 * COEF)
        self.assertEqual(actions["joint_merge_forward"], 1)
        self.assertEqual(actions["joint_merge_forward_bytes"], 10 * COEF)
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


class IncrementFoldAbsorbTests(unittest.TestCase):
    """增量型折叠（迁移腿 W + base == f(P)）：基础前缀就地转正并入、
    占用不动、增长基切完整上下文族。

    手算时间线（coef=100）：
      I0: +1000 (r0 stay) | r1 折叠 tick：+1000 (迁移腿) +400 (decode
          增长，f(F)=f(24) 自 f(P)=2000 起) → 2400，峰 2400
      I1: +1000 (r1 增量 f(U)=f(10)) | −1000 (迁移腿) → 0，峰 1000
    """

    def _records(self):
        return [
            _prefill(R0, 100, 0, "stay"),
            _decode(R0, 110, 0, "stay", _local_hit(0)),
            _completion(R0, 120, direction="stay", working_copy=False,
                        action="stay"),
            # r1：remote-read 增量工作副本@1（S=0、f(U)=1000）→ D 边界
            # 迁移腿 1000 落 base 驻留实例 0：1000+base 1000==f(P)=2000
            # → 增量型折叠（前缀转正并入，占用不动）→ stay 本地提交。
            _prefill(R1, 200, 1, "remote-read"),
            _decode(R1, 210, 0, "remote-read",
                    _migrate_transfer(1, 0, 10 * COEF)),
            _completion(R1, 220, direction="stay", working_copy=False,
                        action="remote-read"),
        ]

    def _tokens(self):
        return [
            token_row(R0, S9, 0, 10, 10),
            token_row(R1, S9, 10, 20, 24),
        ]

    def test_increment_fold_absorbs_and_grows_full_context(self):
        """主线：前缀转正零占用差；decode 增长基 = f(F)（完整上下文族，
        增量口径 f(F−H)=1400 < f(P)=2000 会负增长 fail）；终态 f(24)。"""
        proc, summary = run_case("w4if1", self._records(), self._tokens())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({k: v for k, v in summary["anomalies"].items()
                          if v}, {})
        instances = summary["instances"]
        self.assertEqual(instances["0"]["residual_occupancy_bytes"],
                         24 * COEF)
        self.assertEqual(instances["0"]["peak_occupancy_bytes"], 24 * COEF)
        self.assertEqual(instances["1"]["residual_occupancy_bytes"], 0)
        self.assertEqual(instances["1"]["peak_occupancy_bytes"], 10 * COEF)
        actions = summary["actions"]
        self.assertEqual(actions["joint_boundary_base_fold"], 1)
        self.assertEqual(actions["joint_boundary_base_fold_absorb_bytes"],
                         10 * COEF)
        self.assertEqual(actions.get(
            "joint_boundary_base_fold_release_bytes", 0), 0)

    def test_fold_identity_break_fails_closed(self):
        """折叠字节恒等式破坏：remote-read 带 remote_load S=300（完整
        LOCAL 基形态下的未演练组合，token 篡改 P=22）→ 迁移腿 1500 既
        != f(P)=2200 也 != f(P)−base=1200 → 拒绝静默抹账，fail-closed。"""
        records = [
            _prefill(R0, 100, 0, "stay"),
            _decode(R0, 110, 0, "stay", _local_hit(0)),
            _completion(R0, 120, direction="stay", working_copy=False,
                        action="stay"),
            _prefill(R1, 200, 1, "remote-read", transfers=[{
                "kind": "remote_load", "total_bytes": 3 * COEF,
                "source_instance_index": 0,
                "layer_start": 0, "layer_end": 1}]),
            # 迁移腿 = S + f(U) = 300 + 1200 = 1500（== 跟踪工作副本，
            # 迁移对账通过，专击折叠恒等式）。
            _decode(R1, 210, 0, "remote-read",
                    _migrate_transfer(1, 0, 15 * COEF)),
            _completion(R1, 220, direction="stay", working_copy=False,
                        action="remote-read"),
        ]
        tokens = [
            token_row(R0, S9, 0, 10, 10),
            token_row(R1, S9, 10, 22, 26),
        ]
        proc, _ = run_case("w4if2", records, tokens)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("恒等式不成立", proc.stderr)

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
