#!/usr/bin/env python3
"""test_w8_force_target_hook.py -- W8 钉测：instance×action 精确强制钩子
（方案 §11.3"不能只按 action 汇总最便宜 instance，强制钩子也按该动作
重新选实例"的修复）+ 决策行状态版本披露（状态一致硬证书的事实源）。

合同（与 run_scripts/joint_same_state_probe.py 的 TARGET env 契约互锁）：

* env ``JOINT_FORCE_BOUNDARY_TARGET`` / ``JOINT_FORCE_ENTRY_TARGET`` 值形如
  ``"<instance_index>:<action>"``，与共享 REQUEST env（JOINT_FORCE_BOUNDARY_
  REQUEST）成对武装；语法/值域错、缺 REQUEST、与同族 per-action 变量同设、
  与对族钩子同设、与 p-continuation 同设 ⇒ 启动失败（fail-closed）；
* TARGET 武装时强制**精确** (request, instance, action)：指定实例不在候选
  面 / 该组合不适用 ⇒ 显式 RuntimeError（含同动作各实例），**不静默重选
  其他实例**——含"指定实例非该动作最低成本实例"的反证用例；
* ``forced_by_probe`` 携完整强制身份（stage/hook/action/instance_index/
  is_continuation；hook = "boundary-target"/"entry-target"）；
* boundary_decision 新增 additive ``state_versions`` 披露（W8 证书的目录/
  预约版本事实源），off 路径（无任何钩子 env）行为零变化。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_w8_force_target_hook.py -q
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import ALL_ACTIONS  # noqa: E402
from joint.joint_cost_model import ActionCandidate  # noqa: E402
from joint.joint_scheduler import SelectionRecord  # noqa: E402
from joint.predictor_cost_accounting import (  # noqa: E402
    DECISION_STAGE_D,
    DECISION_STAGE_P,
)
from online.sh30_online_scheduler import (  # noqa: E402
    FORCE_BOUNDARY_ACTION_ENV,
    FORCE_BOUNDARY_REQUEST_ENV,
    FORCE_BOUNDARY_TARGET_ENV,
    FORCE_ENTRY_ACTION_ENV,
    FORCE_ENTRY_TARGET_ENV,
    Sh30OnlineScheduler,
    _check_boundary_policy_instrument_conflict,
    _parse_force_target_env,
)


# --------------------------------------------------------------- env 解析 --

class ForceTargetEnvParseTest(unittest.TestCase):
    """TARGET env 语法/互斥合同（fail-closed：二义仪器比没有更糟）。"""

    def test_off_by_default(self):
        self.assertIsNone(_parse_force_target_env(
            {}, FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV))
        self.assertIsNone(_parse_force_target_env(
            {}, FORCE_ENTRY_TARGET_ENV, FORCE_ENTRY_ACTION_ENV))

    def test_valid_target_parses(self):
        self.assertEqual(
            _parse_force_target_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "session_1_request_2",
                 FORCE_BOUNDARY_TARGET_ENV: "3:copy"},
                FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV),
            ("session_1_request_2", 3, "copy"))

    def test_syntax_errors_fail_closed(self):
        for bad in ("copy", "1:", ":copy", "x:copy", "1:copy:extra",
                    "-1:copy", "1:teleport"):
            with self.assertRaises(RuntimeError, msg=bad):
                _parse_force_target_env(
                    {FORCE_BOUNDARY_REQUEST_ENV: "r",
                     FORCE_BOUNDARY_TARGET_ENV: bad},
                    FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV)

    def test_missing_or_empty_request_fails_closed(self):
        with self.assertRaises(RuntimeError):
            _parse_force_target_env(
                {FORCE_BOUNDARY_TARGET_ENV: "0:stay"},
                FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV)
        with self.assertRaises(RuntimeError):
            _parse_force_target_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "",
                 FORCE_BOUNDARY_TARGET_ENV: "0:stay"},
                FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV)

    def test_sibling_action_var_co_arm_fails_closed(self):
        # 同族 per-action（弃用方向）与 TARGET 同设 = 二义，fail-closed。
        with self.assertRaises(RuntimeError):
            _parse_force_target_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_BOUNDARY_ACTION_ENV: "copy",
                 FORCE_BOUNDARY_TARGET_ENV: "0:copy"},
                FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV)
        with self.assertRaises(RuntimeError):
            _parse_force_target_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_ENTRY_ACTION_ENV: "copy",
                 FORCE_ENTRY_TARGET_ENV: "0:copy"},
                FORCE_ENTRY_TARGET_ENV, FORCE_ENTRY_ACTION_ENV)

    def test_other_family_hook_co_arm_fails_closed(self):
        # 每 run 恰一个动作钩子（entry vs boundary 两族互斥先例沿用）。
        with self.assertRaises(RuntimeError):
            _parse_force_target_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_ENTRY_ACTION_ENV: "copy",
                 FORCE_BOUNDARY_TARGET_ENV: "0:copy"},
                FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV)
        with self.assertRaises(RuntimeError):
            _parse_force_target_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_BOUNDARY_TARGET_ENV: "1:stay",
                 FORCE_ENTRY_TARGET_ENV: "0:copy"},
                FORCE_ENTRY_TARGET_ENV, FORCE_ENTRY_ACTION_ENV)

    def test_boundary_policy_conflict_extends_to_targets(self):
        # p-continuation × 任意强制钩子（含 TARGET）= 仪器二义 fail-closed。
        with self.assertRaises(RuntimeError):
            _check_boundary_policy_instrument_conflict(
                "p-continuation", None, None, ("r", 0, "copy"), None)
        with self.assertRaises(RuntimeError):
            _check_boundary_policy_instrument_conflict(
                "p-continuation", None, None, None, ("r", 0, "stay"))
        # off（缺省 boundary_policy）与 TARGET 武装照常合法。
        self.assertIsNone(_check_boundary_policy_instrument_conflict(
            "off", None, None, ("r", 0, "copy"), None))

    def test_env_names_match_probe_contract(self):
        self.assertEqual(FORCE_BOUNDARY_TARGET_ENV,
                         "JOINT_FORCE_BOUNDARY_TARGET")
        self.assertEqual(FORCE_ENTRY_TARGET_ENV, "JOINT_FORCE_ENTRY_TARGET")


# --------------------------------------------------------- 裁决点（单元）--

def _candidate(instance_index, action, *, applicable=True, cost_ns=1000,
               reason=None, is_continuation=False):
    return ActionCandidate(
        instance_index=instance_index, action=action,
        applicable=applicable,
        inapplicable_reason=(None if applicable else (
            reason or "physically infeasible")),
        cost_ns=(cost_ns if applicable else None),
        is_continuation=is_continuation)


def _record(candidates, chosen):
    return SelectionRecord(
        mode="joint", chosen=chosen, candidates=tuple(candidates),
        instance_rule_note="test")


def _two_instance_surface():
    """stay@0（继续，cost 500）/ copy@0(400) / copy@1(300)——copy 最低成本
    实例是 1：TARGET 强制 copy@0 即"非最低成本实例"反证面；stay@1 物理
    不适用（history not resident）供组合不适用负例；实例 2 仅有 copy
    候选供组合缺位负例（原 recompute 不适用候选已随 2026-10-15
    recompute 退役删除——二动作下以 stay 物理不适用形态承载组合负例）。"""
    stay0 = _candidate(0, "stay", cost_ns=500, is_continuation=True)
    copy0 = _candidate(0, "copy", cost_ns=400)
    copy1 = _candidate(1, "copy", cost_ns=300)
    stay1 = _candidate(1, "stay", applicable=False,
                       reason="history not resident at target")
    copy2 = _candidate(2, "copy", cost_ns=700)
    return _record([stay0, copy0, copy1, stay1, copy2], chosen=copy1)


class ForceBoundaryTargetUnitTest(unittest.TestCase):
    """D 边界 TARGET 精确强制（__new__ 壳单元）。"""

    def _scheduler(self, request_id="r0", instance=0, action="copy"):
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._force_boundary_request = None
        scheduler._force_boundary_action = None
        scheduler._force_boundary_target = (request_id, instance, action)
        return scheduler

    def test_off_returns_record_untouched(self):
        record = _two_instance_surface()
        unarmed = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        self.assertIs(
            unarmed._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)[0], record)
        mismatch = self._scheduler(request_id="other")
        self.assertIs(
            mismatch._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)[0], record)

    def test_exact_instance_action_forced_not_min_cost(self):
        # copy 最低成本实例是 1（300）；TARGET=0:copy 必须强制实例 0，
        # 不得静默重选最低成本实例——instance×action 反事实真值前提。
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=0, action="copy")
        forced, note = scheduler._apply_forced_boundary_choice(
            record, "r0", DECISION_STAGE_D)
        self.assertEqual(forced.chosen.action, "copy")
        self.assertEqual(forced.chosen.instance_index, 0)
        self.assertIs(forced.candidates, record.candidates)  # 计价输入不动
        self.assertEqual(
            note,
            {"stage": DECISION_STAGE_D, "hook": "boundary-target",
             "action": "copy", "instance_index": 0,
             "is_continuation": False})

    def test_continuation_identity_disclosed(self):
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=0, action="stay")
        _forced, note = scheduler._apply_forced_boundary_choice(
            record, "r0", DECISION_STAGE_D)
        self.assertTrue(note["is_continuation"])

    def test_instance_outside_surface_fails_closed(self):
        # 实例 3 无任何候选（越界面）：显式失败，错误信息携带同动作实例
        # 清单（可定位的失败，非静默重选）。
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=3, action="copy")
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)
        message = str(ctx.exception)
        self.assertIn(FORCE_BOUNDARY_TARGET_ENV, message)
        self.assertIn("3:copy", message)
        self.assertIn("exist at instances [0, 1, 2]", message)
        self.assertIn("silently re-select", message)

    def test_inapplicable_pair_fails_closed_with_reason(self):
        # (1, stay) 在候选面但不适用：显式失败并携带理由。
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=1, action="stay")
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)
        message = str(ctx.exception)
        self.assertIn("history not resident at target", message)
        self.assertIn(FORCE_BOUNDARY_TARGET_ENV, message)

    def test_action_absent_at_instance_fails_closed(self):
        # 实例 2 有 copy 候选但无 stay 动作：组合缺位 = 显式失败。
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=2, action="stay")
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)
        self.assertIn(
            "instance/action pair not in candidate surface",
            str(ctx.exception))


class ForceEntryTargetUnitTest(unittest.TestCase):
    """P 入口 TARGET 精确强制（__new__ 壳单元）。"""

    def _scheduler(self, request_id="r0", instance=1, action="copy"):
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._force_entry_request = None
        scheduler._force_entry_action = None
        scheduler._force_entry_target = (request_id, instance, action)
        return scheduler

    def test_exact_instance_action_forced(self):
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=0, action="copy")
        forced, note = scheduler._apply_forced_entry_choice(record, "r0")
        self.assertEqual(forced.chosen.instance_index, 0)
        self.assertEqual(forced.chosen.action, "copy")
        self.assertEqual(
            note,
            {"stage": DECISION_STAGE_P, "hook": "entry-target",
             "action": "copy", "instance_index": 0,
             "is_continuation": False})

    def test_outside_surface_fails_closed(self):
        record = _two_instance_surface()
        scheduler = self._scheduler(instance=5, action="copy")
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._apply_forced_entry_choice(record, "r0")
        self.assertIn(FORCE_ENTRY_TARGET_ENV, str(ctx.exception))

    def test_off_returns_record_untouched(self):
        record = _two_instance_surface()
        unarmed = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        self.assertIs(
            unarmed._apply_forced_entry_choice(record, "r0")[0], record)


class BoundaryPolicyOneShotEnvCompatTest(unittest.TestCase):
    """旧 per-action 解析器对 TARGET 武装形态的 REQUEST 容忍（不误报半配
    置）+ 全部动作名合法枚举钉死（与 joint_config.ALL_ACTIONS 同源）。"""

    def test_boundary_action_parser_tolerates_target_armed(self):
        from online.sh30_online_scheduler import _parse_force_boundary_env
        # REQUEST 由 TARGET 武装：per-action 边界钩子 off（不误报半配置）。
        self.assertEqual(
            _parse_force_boundary_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_BOUNDARY_TARGET_ENV: "0:copy"}),
            (None, None))
        self.assertEqual(
            _parse_force_boundary_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_ENTRY_TARGET_ENV: "0:copy"}),
            (None, None))

    def test_action_universe_stable(self):
        self.assertEqual(
            ALL_ACTIONS,
            ("stay", "copy"))


# ----------------------------------------------- D 入口全链（真决策行）--

class DPathStateVersionsDisclosureTest(unittest.TestCase):
    """真 drain 链路：boundary_decision 落 state_versions（W8 证书事实源）
    + TARGET 强制全链生效（经扩展的 _drive_drain(force_target=...)）。"""

    def test_state_versions_key_present(self):
        from test_c7a_force_boundary_probe_hook import _decode_row, _drive_drain
        scheduler, _runtime = _drive_drain()
        boundary = _decode_row(scheduler)["decision"]["boundary_decision"]
        self.assertIn("state_versions", boundary)
        versions = boundary["state_versions"]
        # 形状钉死（降级维允许 None——嵌入壳无纪元账本；键恒在）。
        self.assertEqual(
            sorted(versions),
            ["directory_version", "kv_ledger_epoch", "stage_epochs"])

    def test_state_versions_disclosure_shape_conversion(self):
        # 单元：_joint_boundary_versions 元组 → JSON 形状（含降级 None）。
        self.assertEqual(
            Sh30OnlineScheduler._joint_state_versions_disclosure(None),
            {"kv_ledger_epoch": None, "directory_version": None,
             "stage_epochs": None})
        self.assertEqual(
            Sh30OnlineScheduler._joint_state_versions_disclosure(
                (7, {"directory_version": 3,
                     "stage_epochs": (("s1", 2), ("s2", 5))})),
            {"kv_ledger_epoch": 7, "directory_version": 3,
             "stage_epochs": [["s1", 2], ["s2", 5]]})
        self.assertEqual(
            Sh30OnlineScheduler._joint_state_versions_disclosure(
                (4, None)),
            {"kv_ledger_epoch": 4, "directory_version": None,
             "stage_epochs": None})

    def test_target_forces_exact_instance_action_full_chain(self):
        # 全链：TARGET=0:stay 强制 D 边界选择 stay@0（式 1c 原地接续），
        # 计价输入（candidates）与 off run 逐位一致，披露完整身份。
        from test_c7a_force_boundary_probe_hook import _decode_row, _drive_drain
        off_scheduler, _ = _drive_drain()
        off_boundary = _decode_row(off_scheduler)["decision"][
            "boundary_decision"]
        on_scheduler, _ = _drive_drain(force_target=("r0", 0, "stay"))
        boundary = _decode_row(on_scheduler)["decision"]["boundary_decision"]
        self.assertEqual(boundary["chosen_action"], "stay")
        self.assertEqual(boundary["chosen_instance_index"], 0)
        note = boundary["forced_by_probe"]
        self.assertEqual(
            note,
            {"stage": DECISION_STAGE_D, "hook": "boundary-target",
             "action": "stay", "instance_index": 0,
             "is_continuation": True})
        # 同状态（计价输入）：候选面逐位一致（适用性/代价/理由/实例；
        # 非计价遥测 wall_ns 经 _price_face_rows 剥除，§5.6 交付面披露
        # 的宿主墙时逐次运行必不同）。
        from test_c7a_force_boundary_probe_hook import _price_face_rows
        self.assertEqual(
            _price_face_rows(boundary["candidates"]),
            _price_face_rows(off_boundary["candidates"]))

    def test_target_outside_surface_fails_closed_full_chain(self):
        from test_c7a_force_boundary_probe_hook import _drive_drain
        # 实例 1 无 stay 候选（该场景 stay 仅在实例 0）：组合缺位 fail。
        with self.assertRaises(RuntimeError) as ctx:
            _drive_drain(force_target=("r0", 1, "stay"))
        message = str(ctx.exception)
        self.assertIn(FORCE_BOUNDARY_TARGET_ENV, message)
        self.assertIn("1:stay", message)


if __name__ == "__main__":
    unittest.main()
