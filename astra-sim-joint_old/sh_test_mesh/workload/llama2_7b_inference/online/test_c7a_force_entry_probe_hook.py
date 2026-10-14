#!/usr/bin/env python3
"""test_c7a_force_entry_probe_hook.py -- P 入口单点强制钩子
JOINT_FORCE_ENTRY_ACTION（修复轮二任务 7，2026-09-27）钉测。

仪器合同（仿 JOINT_FORCE_BOUNDARY_ACTION，sanctioned 测量仪器、默认
off）：

* env 对 = ``JOINT_FORCE_BOUNDARY_REQUEST``（既有，请求命中面复用）+
  ``JOINT_FORCE_ENTRY_ACTION``（新增，值域四动作，只作用于 P 准入
  选择点）；
* 成对武装；半配置/空 request/非法动作 ⇒ 启动失败；与
  ``JOINT_FORCE_BOUNDARY_ACTION`` 同时武装 ⇒ 二义启动失败；
* on：命中请求的 P 准入最终选择强制为指定动作（绕过 argmin），
  candidates 原样（计价/披露照常），决策行落 ``forced_by_probe``
  （stage="P"，hook="entry"）；动作出适用面 ⇒ fail-closed；
* off：行为逐字节不变（原 record 对象原样返回、无披露键）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_c7a_force_entry_probe_hook.py
      （或 pytest 同路径）
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
from online.sh30_online_scheduler import (  # noqa: E402
    DECISION_STAGE_D,
    DECISION_STAGE_P,
    FORCE_BOUNDARY_ACTION_ENV,
    FORCE_BOUNDARY_REQUEST_ENV,
    FORCE_ENTRY_ACTION_ENV,
    Sh30OnlineScheduler,
    _parse_force_boundary_env,
    _parse_force_entry_env,
)


# --------------------------------------------------------------- env 解析 --

class ForceEntryEnvParseTest(unittest.TestCase):
    """P 入口钩子 env 合同（fail-closed：半配置比没有更糟）。"""

    def test_off_by_default(self):
        self.assertEqual(_parse_force_entry_env({}), (None, None))

    def test_entry_action_without_request_fails_closed(self):
        with self.assertRaises(RuntimeError):
            _parse_force_entry_env({FORCE_ENTRY_ACTION_ENV: "copy"})

    def test_empty_request_id_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_force_entry_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "",
                 FORCE_ENTRY_ACTION_ENV: "copy"})

    def test_unknown_action_name_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_force_entry_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_ENTRY_ACTION_ENV: "teleport"})

    def test_both_action_hooks_armed_rejected(self):
        # 两仪器同时武装 = 二义（测量仪器不得有歧义姿态）。
        with self.assertRaises(RuntimeError):
            _parse_force_entry_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_BOUNDARY_ACTION_ENV: "copy",
                 FORCE_ENTRY_ACTION_ENV: "stay"})

    def test_valid_pair_arms_the_entry_hook(self):
        self.assertEqual(
            _parse_force_entry_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "session_9_request_9",
                 FORCE_ENTRY_ACTION_ENV: "recompute"}),
            ("session_9_request_9", "recompute"))

    def test_request_only_boundary_config_keeps_entry_off(self):
        # REQUEST 单独在场属边界钩子的配置面：入口钩子保持 off。
        self.assertEqual(
            _parse_force_entry_env({FORCE_BOUNDARY_REQUEST_ENV: "r"}),
            (None, None))

    def test_boundary_parser_tolerates_entry_armed_shape(self):
        # REQUEST + 仅 ENTRY ACTION：边界钩子 off（不误报半配置）。
        self.assertEqual(
            _parse_force_boundary_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_ENTRY_ACTION_ENV: "copy"}),
            (None, None))

    def test_env_names_match_probe_contract(self):
        self.assertEqual(FORCE_BOUNDARY_REQUEST_ENV,
                         "JOINT_FORCE_BOUNDARY_REQUEST")
        self.assertEqual(FORCE_BOUNDARY_ACTION_ENV,
                         "JOINT_FORCE_BOUNDARY_ACTION")
        self.assertEqual(FORCE_ENTRY_ACTION_ENV, "JOINT_FORCE_ENTRY_ACTION")
        self.assertEqual(
            sorted(ALL_ACTIONS),
            ["copy", "recompute", "remote-read", "stay"])


# --------------------------------------------------------- 裁决点（单元）--

def _candidate(instance_index, action, *, applicable=True, cost_ns=1000,
               reason=None):
    return ActionCandidate(
        instance_index=instance_index, action=action,
        applicable=applicable,
        inapplicable_reason=(None if applicable else (
            reason or "physically infeasible")),
        cost_ns=(cost_ns if applicable else None))


def _record(candidates, chosen):
    return SelectionRecord(
        mode="joint", chosen=chosen, candidates=tuple(candidates),
        instance_rule_note="test", remote_enabled=True)


class ForceEntryChoiceUnitTest(unittest.TestCase):
    """_apply_forced_entry_choice 裁决点语义（__new__ 壳）。"""

    def _scheduler(self, request_id="r0", action="copy"):
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._force_entry_request = request_id
        scheduler._force_entry_action = action
        return scheduler

    def _surface(self):
        stay0 = _candidate(0, "stay", cost_ns=500)
        copy0 = _candidate(0, "copy", cost_ns=400)
        copy1 = _candidate(1, "copy", cost_ns=900)
        remote1 = _candidate(1, "remote-read", applicable=False,
                             reason="no remote source")
        recompute1 = _candidate(1, "recompute", applicable=False,
                                reason="mask excluded")
        return _record(
            [stay0, copy0, copy1, remote1, recompute1], chosen=stay0)

    def test_off_returns_record_untouched(self):
        # 未武装（类级缺省）与请求不匹配：原 record 对象原样返回、无
        # 披露——off 行为零变化（逐字节不变的裁决点证据）。
        record = self._surface()
        unarmed = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        self.assertIs(
            unarmed._apply_forced_entry_choice(record, "r0")[0], record)
        self.assertIsNone(
            unarmed._apply_forced_entry_choice(record, "r0")[1])
        mismatch = self._scheduler(request_id="other")
        self.assertIs(
            mismatch._apply_forced_entry_choice(record, "r0")[0], record)

    def test_forced_entry_action_overrides_argmin(self):
        # on：argmin 本选 stay(500)，强制 copy → 取 copy 面内 min cost
        # 实例 0；candidates 元组同一对象（计价输入不变）；披露键
        # forced_by_probe 携 hook="entry"/stage="P"。
        record = self._surface()
        scheduler = self._scheduler(action="copy")
        forced, note = scheduler._apply_forced_entry_choice(record, "r0")
        self.assertIsNot(forced, record)
        self.assertIs(forced.candidates, record.candidates)
        self.assertEqual(forced.chosen.action, "copy")
        self.assertEqual(forced.chosen.instance_index, 0)
        self.assertEqual(
            note,
            {"stage": DECISION_STAGE_P, "action": "copy",
             "instance_index": 0, "hook": "entry",
             "is_continuation": False})

    def test_outside_surface_fails_closed_with_reasons(self):
        record = self._surface()
        scheduler = self._scheduler(action="remote-read")
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._apply_forced_entry_choice(record, "r0")
        message = str(ctx.exception)
        self.assertIn(FORCE_ENTRY_ACTION_ENV, message)
        self.assertIn("applicable surface", message)
        self.assertIn("no remote source", message)
        # 候选面完全没有的动作（无逐候选理由）同拒。
        scheduler2 = self._scheduler(action="stay")
        record2 = _record([_candidate(0, "copy", cost_ns=10)], chosen=None)
        with self.assertRaises(RuntimeError) as ctx2:
            scheduler2._apply_forced_entry_choice(record2, "r0")
        self.assertIn("not in candidate surface", str(ctx2.exception))

    def test_entry_hook_does_not_touch_boundary_adjudication(self):
        # 入口钩子武装不影响边界裁决点（D 边界照常走
        # _apply_forced_boundary_choice 的 off 分支——类级缺省 off；
        # 边界钩子只接线 D 边界，单元调用一律用接线 stage = D）。
        record = self._surface()
        scheduler = self._scheduler(action="copy")
        # _apply_forced_boundary_choice 读的是 boundary 对（None/None）。
        scheduler._force_boundary_request = None
        scheduler._force_boundary_action = None
        self.assertIs(
            scheduler._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)[0], record)


# ------------------------------------------------------- P 入口全链（源级）--

class PEntryWiringTest(unittest.TestCase):
    """P 准入选择点接线：只接入口钩子（修复轮三 R8：BOUNDARY 钩子的
    P 侧接线已删——boundary 臂只强制 D 边界，不触 P 面）、决策行披露
    之前。"""

    def test_admission_wires_only_entry_hook(self):
        import inspect
        source = inspect.getsource(Sh30OnlineScheduler._try_admit_request)
        # 修复轮三 R8：P 准入不再接线 BOUNDARY 钩子（轮三前的 P 侧接线
        # 会在到达 D 边界前先被 P 面适用性核验 fail-closed）——P 面强制
        # 只走入口钩子，披露键照常落 forced_by_probe。
        self.assertNotIn("_apply_forced_boundary_choice", source)
        self.assertIn("_apply_forced_entry_choice(", source)
        self.assertIn("forced_by_probe", source)

    def test_init_parses_entry_hook(self):
        import inspect
        source = inspect.getsource(Sh30OnlineScheduler.__init__)
        self.assertIn("_parse_force_entry_env", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
