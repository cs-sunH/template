#!/usr/bin/env python3
"""test_hbm_watermark_three_position.py -- W2（2026-09-27）钉测：joint
merge 重放词表的三位置形态（C4/D12 运行时三位置确定性合并）。

素材口径（schema 逐符号锚定发射面，字段名/reason 不发明新口径）：
  * decode 行 prefill_decode_transfer = local_hit（source=target=新执行端
    j、total_bytes=0、reason=boundary_rebuild_elsewhere（recompute 重建）/
    boundary_remote_no_migration（remote-read 多源））——i_P 旧工作副本
    不迁移、留 i_P 转来源侧（U），j 侧新副本零化起步；
  * completion 行 merge_direction=forward/reverse（reverse ⟺ 胜者=执行
    端）、home_flipped_to（home 迁移时的胜者）、merge_transferred_bytes
    =Σ腿、merge_transfers 逐败者一腿 noc_migrate
    reason=merge_three_position_into_winner（腿字节=该位置全量）；重建
    形态零腿（重复副本直接释放）。

手算场景（coef=100 B/token：layers=1/hidden=50/bpe=1，与既有合成口径
一致）：两会话混排——s7 多源三位置（base H@0 + U@1 + growth@2，胜者=
home 0 两腿搬入）+ s8 重建三位置（零载荷，败者 0/1 静默释放，胜者=
exec 2）+ s7 前两轮的两位置 forward/stay（混合序列共存）。逐实例
残差/峰值与动作计数逐字段手算断言；负例反向构造方向真错/腿错/守恒
破坏，各必须 fail-closed（exit 2）。重合豁免臂：重建×base==exec
（recompute@基础驻留端）闭合 + 非重建重合/非零载荷腿仍 fail。

运行：python3 -m pytest slo_tools/tests/test_hbm_watermark_three_position.py
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

TP_COEF = 100  # B/token

# s7（多源三位置）：base H@0、U@1、growth@2；胜者 = home 0（保留量最大）。
# s8（重建三位置）：recompute 重建副本@2 持并集；败者 0/1 零载荷释放。
S7, S8 = "session_7", "session_8"
R7_0, R7_1, R7_2 = "s7_r0", "s7_r1", "s7_r2"
R8_0, R8_1 = "s8_r0", "s8_r1"


def tp_token_requests() -> list[dict]:
    return [
        token_row(R7_0, S7, 0, 10, 12),
        token_row(R7_1, S7, 12, 20, 24),
        token_row(R8_0, S8, 0, 10, 12),
        token_row(R8_1, S8, 12, 20, 24),
        token_row(R7_2, S7, 24, 34, 40),
    ]


def _local_hit_boundary(instance: int, reason: str,
                        session: str) -> dict:
    """decode 行无迁移边界披露（face_scheduler 边界事务零化后产
    local_hit：source=target=j、字节 0；shards 空）。"""
    return {"kind": "local_hit", "phase": "prefill_decode",
            "reason": reason, "session_id": session,
            "source_instance_index": instance,
            "target_instance_index": instance,
            "layer_start": 0, "layer_end": 1,
            "resident_prefix_layers_before": 1,
            "resident_prefix_layers_after": 1,
            "total_bytes": 0, "shards": []}


def _merge_leg(source: int, total_bytes: int, session: str) -> dict:
    """completion 行三位置合并腿（noc_migrate，
    reason=merge_three_position_into_winner）。"""
    return {"kind": "noc_migrate", "phase": "completion",
            "reason": "merge_three_position_into_winner",
            "session_id": session, "source_instance_index": source,
            "layer_start": 0, "layer_end": 1, "total_bytes": total_bytes,
            "shards": []}


def three_position_records() -> list[dict]:
    """混合序列：两位置 stay/forward（s7 前两轮）+ 重建三位置（s8）+
    多源三位置（s7_r2）——tick 单调，逐事件手算见类注记。"""
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    def prefill(request_id, tick, instance, action):
        return {"kind": "prefill", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": {"prefill_instance_index": instance,
                             "joint_action": action,
                             "history_tokens_discarded": 0,
                             "history_evictions": [],
                             "history_transfers": [],
                             "history_transfer_bytes": 0,
                             "history_source_instance_index": None}}

    def decode(request_id, tick, instance, action, *, transfer=None,
               decode_action=None):
        decision = {"decode_instance_index": instance,
                    "joint_action": action,
                    "decode_evictions": []}
        if transfer is not None:
            decision["prefill_decode_transfer"] = transfer
        if decode_action is not None:
            decision["joint_decode_action"] = decode_action
            decision["is_continuation"] = False
        return {"kind": "decode", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": decision}

    def completion(request_id, tick, *, direction, working_copy, merges,
                   home_flipped=None, origin_home=0, transferred=None,
                   action="remote-read"):
        decision = {"joint_action": action,
                    "merge_direction": direction,
                    "joint_working_copy": working_copy,
                    "merge_transfers": merges,
                    "home_flipped_to": home_flipped,
                    "origin_home_instance": origin_home,
                    "kv_location_after_completion": "local_hbm",
                    "completion_evictions": []}
        if transferred is not None:
            decision["merge_transferred_bytes"] = transferred
        return {"kind": "completion", "request_id": request_id,
                "priority": 0, "seq": nxt(), "tick": tick,
                "decision": decision}

    return [
        # -- s7_r0：stay-prefill@0 → f(12)=1200 驻留 home。 --
        prefill(R7_0, 100, 0, "stay"),
        decode(R7_0, 110, 0, "stay"),
        completion(R7_0, 120, direction="stay", working_copy=False,
                   merges=[], action="stay"),
        # -- s7_r1：remote-read 工作副本@1（f(24)−f(12)=1200）→ 两位置
        #    forward 合并回 home（腿 1200）→ home 持 f(24)=2400。 --
        prefill(R7_1, 200, 1, "remote-read"),
        decode(R7_1, 210, 1, "remote-read"),
        completion(R7_1, 220, direction="forward", working_copy=True,
                   merges=[{"kind": "noc_migrate", "phase": "completion",
                            "reason": "merge_working_copy_to_home",
                            "session_id": S7, "source_instance_index": 1,
                            "layer_start": 0, "layer_end": 1,
                            "total_bytes": 1200, "shards": []}],
                   transferred=1200),
        # -- s8_r0：stay-prefill@0 → f(12)=1200（与 s7 同宿 home 0）。 --
        prefill(R8_0, 300, 0, "stay"),
        decode(R8_0, 310, 0, "stay"),
        completion(R8_0, 320, direction="stay", working_copy=False,
                   merges=[], action="stay"),
        # -- s8_r1：recompute 工作副本@1（f(20)=2000）→ D 边界重建@2
        #    （来源侧 1 转 U，j 侧重建至 f(24)=2400）→ 重建三位置零载荷
        #    合并：败者 0（1200）/1（2000）静默释放，胜者=exec 2，
        #    home 迁移 0→2。 --
        prefill(R8_1, 400, 1, "recompute"),
        decode(R8_1, 410, 2, "recompute",
               transfer=_local_hit_boundary(
                   2, "boundary_rebuild_elsewhere", S8),
               decode_action="recompute"),
        completion(R8_1, 420, direction="reverse", working_copy=True,
                   merges=[], home_flipped=2, transferred=0,
                   action="recompute"),
        # -- s7_r2：remote-read 工作副本@1（f(34)−f(24)=1000）→ D 边界
        #    多源远读@2（U 留 1，j 侧仅 decode 增长 f(40)−f(34)=600）→
        #    多源三位置合并：胜者=home 0（保留量 2400 最大），两腿
        #    1→0（1000）/2→0（600）整份搬入。 --
        prefill(R7_2, 500, 1, "remote-read"),
        decode(R7_2, 510, 2, "remote-read",
               transfer=_local_hit_boundary(
                   2, "boundary_remote_no_migration", S7),
               decode_action="remote-read"),
        completion(R7_2, 520, direction="forward", working_copy=True,
                   merges=[_merge_leg(1, 1000, S7), _merge_leg(2, 600, S7)],
                   transferred=1600),
    ]


def run_three_position(tag: str, mutate=None, mutate_tokens=None):
    """跑三位置混合切片；mutate/mutate_tokens 供负例篡改（深拷贝）。"""
    records = three_position_records()
    tokens = tp_token_requests()
    if mutate is not None:
        records = mutate(copy.deepcopy(records))
    if mutate_tokens is not None:
        tokens = mutate_tokens(copy.deepcopy(tokens))
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


def _landing_at_base(records, *, reason="boundary_rebuild_elsewhere",
                     merges=None, transferred=None):
    """把 s8_r1 的 D 边界重建落点改到 base 驻留实例 0（recompute@基础
    驻留端形态）：decode 行落 0、local_hit source 同步 0；胜者 = exec
    = 0 = home（home 未迁移，无 home_flipped_to 披露）。merges/
    transferred 供负例覆写 completion 的传输披露。"""
    for row in records:
        if row["request_id"] != R8_1:
            continue
        decision = row["decision"]
        if row["kind"] == "decode":
            decision["decode_instance_index"] = 0
            decision["prefill_decode_transfer"] = _local_hit_boundary(
                0, reason, S8)
        elif row["kind"] == "completion":
            decision["home_flipped_to"] = None
            if merges is not None:
                decision["merge_transfers"] = merges
            if transferred is not None:
                decision["merge_transferred_bytes"] = transferred
    return records


class ThreePositionReplayTests(unittest.TestCase):
    """W2 钉测：三位置重放（源扣/目标加/守恒/无负占用）+ 混合序列共存。

    手算时间线（delta@实例，coef=100）：
      I0: +1000 +200 (s7_r0) | +1200 (s7_r1 fwd) | +1000 +200 (s8_r0)
          | −1200 (s8 重建释放) | +1600 (s7_r2 两腿) → 终 4000，峰 4000
      I1: +800 +400 (s7_r1) −1200 (fwd) | +2000 (s8_r1) −2000 (释放)
          | +1000 (s7_r2) −1000 (腿) → 终 0，峰 2000
      I2: +2400 (s8 重建) | +600 (s7_r2 增长) −600 (腿) → 终 2400，峰 3000
    """

    def test_three_position_replay_closes_and_occupancy_exact(self):
        """主线：重放零异常闭合；源扣/目标加逐字节精确（腿字节=位置
        全量）；胜者终态守恒 = f(final)；全程无负占用（fail-closed 未
        触发即无多扣）；两位置 stay/forward 轮与三位置轮共存同一会话。"""
        proc, summary = run_three_position("w2tp1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({k: v for k, v in summary["anomalies"].items()
                          if v}, {})
        instances = summary["instances"]
        self.assertEqual(instances["0"]["residual_occupancy_bytes"],
                         40 * TP_COEF)
        self.assertEqual(instances["0"]["peak_occupancy_bytes"],
                         40 * TP_COEF)
        self.assertEqual(instances["1"]["residual_occupancy_bytes"], 0)
        self.assertEqual(instances["1"]["peak_occupancy_bytes"],
                         20 * TP_COEF)
        self.assertEqual(instances["2"]["residual_occupancy_bytes"],
                         24 * TP_COEF)
        self.assertEqual(instances["2"]["peak_occupancy_bytes"],
                         30 * TP_COEF)
        actions = summary["actions"]
        # 边界来源侧转记录：s8_r1（2000）+ s7_r2（1000）。
        self.assertEqual(actions["joint_boundary_source_stash"], 2)
        self.assertEqual(actions["joint_boundary_source_stash_bytes"],
                         3000)
        # 三位置结算：两例；腿字节 = 多源 1600（重建零载荷）。
        self.assertEqual(actions["joint_merge_three_position"], 2)
        self.assertEqual(actions["joint_merge_three_position_bytes"], 1600)
        self.assertEqual(
            actions["joint_merge_three_position_zero_payload"], 1)
        self.assertEqual(
            actions["joint_merge_three_position_release_bytes"], 3200)
        # 两位置 forward（s7_r1）不受三位置分支影响；home 迁移 = s8
        # 重建（0→2）恰一次。
        self.assertEqual(actions["joint_merge_forward"], 1)
        self.assertEqual(actions["joint_merge_forward_bytes"], 1200)
        self.assertEqual(actions["joint_working_release"], 1)
        self.assertEqual(actions["joint_home_migration"], 1)
        self.assertEqual(actions["joint_merge_settle"], 3)
        self.assertEqual(actions["joint_merge_stay"], 2)
        self.assertEqual(actions["joint_stay_prefill"], 2)

    def test_direction_flip_forward_to_reverse_fails_closed(self):
        """方向真错（反向构造①）：多源行胜者实为 home 0，把
        merge_direction 篡改为 reverse（home_flipped_to 仍 None）→ 胜者
        裁决为执行端 2，exec 增长腿源端 2 == 胜者 → 方向披露与传输矛盾
        fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_direction"] = "reverse"
            return records

        proc, _ = run_three_position("w2tp2", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("方向披露与传输矛盾", proc.stderr)

    def test_direction_flip_reverse_home_flipped_fails_closed(self):
        """方向真错（反向构造②）：reverse 行的 home_flipped_to 与胜者
        （执行端）矛盾 → 方向交叉校验直接命中。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_direction"] = "reverse"
                    row["decision"]["home_flipped_to"] = 0
            return records

        proc, _ = run_three_position("w2tp3", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("home_flipped_to=0 != 胜者（执行端）2", proc.stderr)

    def test_rebuild_direction_forward_fails_closed(self):
        """方向真错（反向构造③）：重建行胜者恒为执行端（reverse），篡改
        为 forward（home_flipped_to=2 保留）→ forward 胜者==执行端
        fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R8_1 and row["kind"] == "completion":
                    row["decision"]["merge_direction"] = "forward"
            return records

        proc, _ = run_three_position("w2tp4", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("forward ⟺ 胜者≠执行端", proc.stderr)

    def test_leg_bytes_mismatch_fails_closed(self):
        """腿字节 != 位置跟踪全量（少搬/多搬）→ fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_transfers"][0][
                        "total_bytes"] += TP_COEF
            return records

        proc, _ = run_three_position("w2tp5", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("整份搬移", proc.stderr)

    def test_leg_from_untracked_position_fails_closed(self):
        """腿源端不在跟踪位置集合（幻影源）→ fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_transfers"][0][
                        "source_instance_index"] = 5
            return records

        proc, _ = run_three_position("w2tp6", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("不在跟踪位置", proc.stderr)

    def test_missing_leg_fails_closed(self):
        """多源形态败者无腿（漏腿=漏搬移字节）→ fail-closed，拒绝静默
        漏记。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_transfers"] = \
                        row["decision"]["merge_transfers"][:1]
                    row["decision"]["merge_transferred_bytes"] = 1000
            return records

        proc, _ = run_three_position("w2tp7", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("无 noc 腿", proc.stderr)

    def test_rebuild_with_noc_leg_fails_closed(self):
        """重建形态携带 noc 传输（与零载荷语义矛盾）→ fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R8_1 and row["kind"] == "completion":
                    row["decision"]["merge_transfers"] = [
                        _merge_leg(1, 2000, S8)]
                    row["decision"]["merge_transferred_bytes"] = 2000
            return records

        proc, _ = run_three_position("w2tp8", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("重建合并不得携带 noc 传输", proc.stderr)

    def test_stay_with_pending_source_fails_closed(self):
        """来源槽在册而完成行走 stay 本地提交 → 形态矛盾 fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_direction"] = "stay"
            return records

        proc, _ = run_three_position("w2tp9", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("merge_direction=stay 但来源侧在册", proc.stderr)

    def test_legacy_completion_with_pending_source_fails_closed(self):
        """来源槽在册而完成行无 merge_direction（legacy 抹账面）→
        显式拒收。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"].pop("merge_direction")
            return records

        proc, _ = run_three_position("w2tp10", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("缺 merge_direction 但来源侧在册", proc.stderr)

    def test_conservation_break_fails_closed(self):
        """守恒破坏（manager 结算守恒断言的镜像）：构造逐腿对账全过、但
        三位置字节合计 ≠ f(final) 的非法账本（P 侧改为 copy 语义——工作
        副本持 f(prefill_context) 全量而非 U 增量，与来源侧转记录形态组
        合后重复计入 base 域；真实 manager 侧自身的守恒断言也会拦）→
        胜者终态守恒断言命中。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2:
                    if row["kind"] == "prefill":
                        # copy 语义：工作副本长到 f(34)=3400（非 U=1000）。
                        row["decision"]["joint_action"] = "copy"
                    elif row["kind"] == "completion":
                        # 腿按篡改后的位置全量对齐（逐腿检查全过）。
                        row["decision"]["merge_transfers"] = [
                            _merge_leg(1, 34 * TP_COEF, S7),
                            _merge_leg(2, 6 * TP_COEF, S7)]
                        row["decision"]["merge_transferred_bytes"] = \
                            40 * TP_COEF
            return records

        proc, _ = run_three_position("w2tp11", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("结算守恒失败", proc.stderr)
        self.assertIn("!= f(final)", proc.stderr)

    def test_disclosed_transferred_bytes_mismatch_fails_closed(self):
        """披露 merge_transferred_bytes != 腿合计 → 两侧账本不一致
        fail-closed。"""

        def mutate(records):
            for row in records:
                if row["request_id"] == R7_2 and row["kind"] == "completion":
                    row["decision"]["merge_transferred_bytes"] = 1500
            return records

        proc, _ = run_three_position("w2tp12", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("transferred_bytes 1500 != 腿合计 1600", proc.stderr)

    def test_recompute_at_base_resident_end_closes(self):
        """重合豁免（镜像 manager 的 coincident-holders 豁免）：重建×
        base 与 exec 同实例（recompute@基础驻留端）——base 记录是重复
        副本，不占位置、零载荷就地释放；胜者 = exec = 0，终态守恒
        = f(final)=f(24)。

        手算时间线（delta@实例，coef=100；与主线仅 s8_r1 落点不同）：
          I0: …3600（同主线至 s8_r0）+2400（重建增长@0）−1200（重复
              base 释放）+1600（s7_r2 两腿）→ 终 6400，峰 6400
          I1: +2000（s8_r1 P 侧）−2000（来源侧释放）→ 终 0，峰 2000
          I2: +600 −600（s7_r2）→ 终 0，峰 600
        计数披露：zero_payload=1；release_bytes = 2000（来源）+
        1200（重复 base）= 3200；home 迁移 0 次（胜者即 home）。"""
        proc, summary = run_three_position(
            "w2tp13", lambda records: _landing_at_base(records))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual({k: v for k, v in summary["anomalies"].items()
                          if v}, {})
        instances = summary["instances"]
        self.assertEqual(instances["0"]["residual_occupancy_bytes"],
                         64 * TP_COEF)
        self.assertEqual(instances["0"]["peak_occupancy_bytes"],
                         64 * TP_COEF)
        self.assertEqual(instances["1"]["residual_occupancy_bytes"], 0)
        self.assertEqual(instances["1"]["peak_occupancy_bytes"],
                         20 * TP_COEF)
        self.assertEqual(instances["2"]["residual_occupancy_bytes"], 0)
        self.assertEqual(instances["2"]["peak_occupancy_bytes"],
                         6 * TP_COEF)
        actions = summary["actions"]
        self.assertEqual(actions["joint_merge_three_position"], 2)
        # 腿字节 = 多源 1600（重建豁免形态零载荷、零腿）。
        self.assertEqual(actions["joint_merge_three_position_bytes"], 1600)
        self.assertEqual(
            actions["joint_merge_three_position_zero_payload"], 1)
        # 重复副本释放披露：来源侧 2000 + 同实例重复 base 1200。
        self.assertEqual(
            actions["joint_merge_three_position_release_bytes"], 3200)
        # 胜者落 home（0）——无 home 迁移（零计数不落键，get 口径）；
        # 其余混合序列计数不变。
        self.assertEqual(actions.get("joint_home_migration", 0), 0)
        self.assertEqual(actions["joint_boundary_source_stash"], 2)
        self.assertEqual(actions["joint_boundary_source_stash_bytes"],
                         3000)
        self.assertEqual(actions["joint_merge_forward"], 1)
        self.assertEqual(actions["joint_working_release"], 1)
        self.assertEqual(actions["joint_merge_settle"], 3)
        self.assertEqual(actions["joint_merge_stay"], 2)

    def test_coincident_holder_non_rebuild_fails_closed(self):
        """非法重合①（非 rebuild）：base==exec 的重合出现在多源远读
        边界（growth_only）——重建豁免不适用，位置重合守卫 fail-closed
        （多源远读族的任何重合仍是边界事务合同违规）。"""

        def mutate(records):
            return _landing_at_base(
                records, reason="boundary_remote_no_migration")

        proc, _ = run_three_position("w2tp14", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("three-position merge 位置重合（instance=0", proc.stderr)

    def test_coincident_holder_rebuild_noc_leg_fails_closed(self):
        """非法重合②（字节不符/非零载荷）：豁免形态下给重复 base 补一
        根 noc 腿（1200 B，试图把重复副本按搬移结算）——重建合并零传
        输（重复副本只能零载荷就地释放），fail-closed。"""

        def mutate(records):
            return _landing_at_base(
                records, merges=[_merge_leg(0, 12 * TP_COEF, S8)],
                transferred=12 * TP_COEF)

        proc, _ = run_three_position("w2tp15", mutate)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("重建合并不得携带 noc 传输", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
