#!/usr/bin/env python3
"""test_c1_boundary_mask_continuation.py -- C1 钉测：M2 掩码 × D 边界
continuation 红线的掩码感知裁定 + 在线边界块异常安全。

背景（joint仓库改造深入审查报告 2026-09-27 §一.1）：完整型边界把原地
继续映射为 stay@i_P，``JOINT_ALLOWED_ACTIONS`` 掩码剔除 stay 后允许
面整体塌缩或仅剩迁移候选，选择器误报 "boundary traversal lost the
in-place continuation record"；异常发生在事务 try 块之外，阶段引用已
获取未释放，请求悬挂在 D_DECISION_PENDING。

钉测条款（joint修改文档 M2：掩码是候选集上限不是强制）：
  1. 延续动作属允许面（无掩码/掩码放行）时，x0 记录缺席仍 =
     结构性缺陷 fail-closed（红线不变）；
  2. 延续动作被掩码剔除时，在允许面内正常选择（平局键退化为
     (cost, instance_index, action_order)，无继续优先），决策行
     diagnostics 披露 continuation_masked_out；
  3. 掩码把允许面整体剔空 → 明确的 no-applicable-under-mask
     fail-closed 错误（不许空序列 min()，不误报 continuation 丢失）；
  4. {copy,recompute}+load-first 臂：P 选择 → 边界 → copy@j →
     D_COMMITTED（decode 就绪入队）全链路不抛异常，无阶段引用/预约
     残留（decode 列车发射与完成结算由冒烟矩阵覆盖，本文件不重建）；
  5. {recompute} 臂：选择层不再误报 continuation 错——允许面空 →
     明确掩码 fail-closed 错误，且异常路径阶段引用幂等释放、无泄漏
     （该臂全链路通由 C4 验收）；
  6. 边界块自 acquire_stage_reference 起整体异常安全：任何异常先幂等
     释放阶段引用再重抛（故障注入钉测，不依赖具体异常类型）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_c1_boundary_mask_continuation.py
      （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_STAY,
    MASK_INAPPLICABLE_REASON,
    ActionCandidate,
)
from joint.joint_scheduler import (  # noqa: E402
    JointSchedulerError,
    select_decode_target_and_action,
    select_instance_and_action,
)
from joint.test_d18_boundary_decision import (  # noqa: E402
    _CountingModel,
    _complete_boundary,
    _complete_costs,
    _drain_ready_scheduler,
    _increment_boundary,
    _loads as d_loads,
    _model as d_model,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _cost_model,
    _load,
    _request,
    _session,
)


def _config(allowed):
    """审查指定的臂预设（真实 parse_joint_config 解析）。"""
    return parse_joint_config({
        "JOINT_ALLOWED_ACTIONS": allowed,
        "JOINT_SCHEDULER_MODE": "load-first",
    })


class _MaskedCountingModel(_CountingModel):
    """掩码感知存根：``allowed_actions`` 之外的动作不可用（与 JCM
    ``applicable_D_remaining_actions`` 尾部过滤同语义——掩码命中即剔
    除），其余代价表行为同 ``_CountingModel``。"""

    allowed_actions = None

    def estimate_D_remaining(self, *, boundary, instance_index, action,
                             remote_enabled, **_d17_disclosure):
        if (self.allowed_actions is not None
                and action not in self.allowed_actions):
            return ActionCandidate(
                instance_index=instance_index, action=action,
                applicable=False,
                inapplicable_reason=MASK_INAPPLICABLE_REASON, cost_ns=None)
        return super().estimate_D_remaining(
            boundary=boundary, instance_index=instance_index,
            action=action, remote_enabled=remote_enabled,
            **_d17_disclosure)


class C1MaskedBoundarySelectionTest(unittest.TestCase):
    """D 边界选择器：continuation 红线的掩码感知裁定。"""

    def test_copy_recompute_mask_selects_within_allowed_face(self):
        """{copy,recompute} 掩码：stay@i_P 被掩码剔除 = 掩码上限的合
        法结果，选择在允许面内正常进行（recompute@0 / copy@j——C4 完
        整动作面后两者同面竞争，取成本更低者），不再误报 continuation
        丢失；决策行披露 continuation 被 mask 排除。"""
        config = _config("copy,recompute")
        p_model = _cost_model(
            {0: _load(0, 100), 1: _load(1, 5000)},
            allowed_actions=config.allowed_action_set)
        p = select_instance_and_action(
            mode=config.scheduler_mode, cost_model=p_model,
            session=_session(home=0, resident=1), request=_request(),
            remote_enabled=True)
        self.assertEqual(
            (p.chosen.instance_index, p.chosen.action), (0, "recompute"))
        d_model_inst = d_model(d_loads())
        d_model_inst.allowed_actions = config.allowed_action_set
        record = select_decode_target_and_action(
            mode=config.scheduler_mode, cost_model=d_model_inst,
            boundary=_complete_boundary(
                p_instance_index=p.chosen.instance_index,
                p_action=p.chosen.action),
            remote_enabled=True)
        # 允许面内选择：不出面（stay/remote-read 被掩码剔除）。
        self.assertIn(record.chosen.action, ("copy", "recompute"))
        self.assertFalse(record.chosen.is_continuation)
        self.assertTrue(record.diagnostics["continuation_masked_out"])
        self.assertEqual(record.diagnostics["continuation_action"],
                         ACTION_STAY)
        self.assertIsNone(record.diagnostics["continuation_cost_ns"])
        self.assertIn("mask", record.instance_rule_note)
        # 被掩码剔除的 stay 候选在候选面留痕（拒绝理由 = 掩码统一串）。
        stay_at_p = [
            candidate for candidate in record.candidates
            if candidate.instance_index == 0
            and candidate.action == ACTION_STAY][0]
        self.assertFalse(stay_at_p.applicable)
        self.assertEqual(stay_at_p.inapplicable_reason,
                         MASK_INAPPLICABLE_REASON)

    def test_recompute_only_mask_selects_recompute_face(self):
        """{recompute} 掩码（C4 后允许面非空）：D 边界 recompute 在可
        选面内——选择层正常出解（静默重建候选、无 continuation 误报；
        掩码把面收窄到 recompute@j 仍可全链路执行，见 C4 专项测试）。"""
        config = _config("recompute")
        d_model_inst = d_model(d_loads())
        d_model_inst.allowed_actions = config.allowed_action_set
        record = select_decode_target_and_action(
            mode=config.scheduler_mode, cost_model=d_model_inst,
            boundary=_complete_boundary(p_instance_index=0,
                                        p_action="recompute"),
            remote_enabled=True)
        self.assertEqual(record.chosen.action, "recompute")
        self.assertTrue(record.chosen.applicable)
        self.assertIsNotNone(record.chosen.cost_ns)
        self.assertTrue(record.diagnostics["continuation_masked_out"])
        # 允许面外动作（stay/copy/remote-read）逐个掩码剔除留痕；物理
        # 上本就不可用的组合保留物理理由（M2 模板：掩码只覆盖可用者
        # ——stay@i_P 物理可用 ⇒ 掩码串；copy@i_P/remote@i_P 物理不可
        # 用 ⇒ 物理串）。
        mask_expected = {
            (0, "stay"), (1, "copy"), (1, "remote-read"),
        }
        for candidate in record.candidates:
            if candidate.action != "recompute":
                self.assertFalse(candidate.applicable)
                if (candidate.instance_index,
                        candidate.action) in mask_expected:
                    self.assertEqual(candidate.inapplicable_reason,
                                     MASK_INAPPLICABLE_REASON)
                else:
                    self.assertNotEqual(
                        candidate.inapplicable_reason,
                        MASK_INAPPLICABLE_REASON)

    def test_empty_allowed_face_still_fails_closed_with_mask_attribution(
            self):
        """掩码把允许面整体剔空 → 明确的 no-applicable-under-mask
        fail-closed 错误（不许空序列 min()；不误报 continuation 丢失）。
        C4 后 recompute 恒适用，空面形态 = 剔除 recompute 且增量型边界
        上 stay 物理不适用（如 {stay} 掩码 × 增量型布局）。"""
        config = _config("stay")
        d_model_inst = d_model(d_loads())
        d_model_inst.allowed_actions = config.allowed_action_set
        with self.assertRaises(JointSchedulerError) as caught:
            select_decode_target_and_action(
                mode=config.scheduler_mode, cost_model=d_model_inst,
                boundary=_increment_boundary(),
                remote_enabled=True)
        message = str(caught.exception)
        self.assertIn("no applicable candidate", message)
        self.assertIn("JOINT_ALLOWED_ACTIONS mask", message)
        self.assertNotIn("lost the in-place continuation", message)

    def test_continuation_in_allowed_face_keeps_structural_fail_closed(self):
        """延续动作属允许面（掩码放行 stay）时，x0 记录缺席仍是结构性
        缺陷——红线 fail-closed 不变。"""
        costs = _complete_costs(100, 90)
        costs[(0, "stay")] = None  # x0 缺席 = 候选面结构性缺陷
        model = _MaskedCountingModel(costs)
        model.allowed_actions = frozenset({ACTION_STAY, "copy"})
        with self.assertRaises(JointSchedulerError) as caught:
            select_decode_target_and_action(
                mode="joint", cost_model=model,
                boundary=_complete_boundary(), remote_enabled=True)
        self.assertIn("lost the in-place continuation record",
                      str(caught.exception))

    def test_masked_tie_key_degrades_without_continuation_preference(self):
        """延续动作被掩码剔除时平局键退化为
        (cost, instance_index, action_order)——无继续候选可优先（式 1c
        的继续优先只在延续动作属允许面时成立）。"""
        # 掩码剔除 stay 后存根允许面 = copy@0/copy@1 同成本并列。
        costs = {
            (0, "stay"): (100, True),
            (0, "recompute"): None,
            (0, "copy"): (100, False),
            (0, "remote-read"): None,
            (1, "stay"): None,
            (1, "recompute"): None,
            (1, "copy"): (100, False),
            (1, "remote-read"): None,
        }
        model = _MaskedCountingModel(costs)
        model.allowed_actions = frozenset({"copy"})
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary(), remote_enabled=True)
        self.assertEqual(record.chosen.instance_index, 0)
        self.assertEqual(record.chosen.action, "copy")
        self.assertFalse(record.chosen.is_continuation)
        self.assertTrue(record.diagnostics["continuation_masked_out"])
        self.assertIn("(cost, instance, action order)",
                      record.tie_break_note)


# ------------------------------------------------------------ 状态机 --


class C1OnlineBoundarySafetyTest(unittest.TestCase):
    """在线 drain 边界：掩码臂链路 + 边界块整体异常安全。"""

    def _masked_shell(self, allowed, decode_length=8):
        """d18 真 drain 壳 + 审查臂预设（P 工作副本已在实例 0 物化，
        按臂语义标注 P 动作后进入真实 drain 边界）。"""
        scheduler, kv, runtime = _drain_ready_scheduler(
            decode_length=decode_length)
        config = _config(allowed)
        scheduler.joint_config = config
        scheduler._joint_mode = config.scheduler_mode
        runtime.joint_action = "recompute"
        return scheduler, kv, runtime

    def test_copy_recompute_mask_arm_drains_to_copy_join_ready(self):
        """{copy,recompute}+load-first 臂：P 选择（真实选择器同参）→
        边界（真实 drain 回调）→ copy@j → D_COMMITTED（decode 就绪入
        队）不抛异常；无阶段引用/预约残留、无回退。"""
        # P 腿：与审查复现同一真实选择器链（load-first → recompute@i_P）。
        config = _config("copy,recompute")
        p_model = _cost_model(
            {0: _load(0, 100), 1: _load(1, 5000)},
            allowed_actions=config.allowed_action_set)
        p = select_instance_and_action(
            mode=config.scheduler_mode, cost_model=p_model,
            session=_session(home=0, resident=1), request=_request(),
            remote_enabled=True)
        self.assertEqual(
            (p.chosen.instance_index, p.chosen.action), (0, "recompute"))
        # 边界腿：真 drain。
        scheduler, kv, runtime = self._masked_shell("copy,recompute")
        scheduler._on_prefill_drain(runtime.request_id, 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertEqual(runtime.joint_decode_action, "copy")
        self.assertFalse(runtime.joint_decode_is_continuation)
        self.assertFalse(runtime.boundary_decision["fallback"])
        self.assertEqual(
            runtime.boundary_decision["chosen_instance_index"], 1)
        self.assertEqual(runtime.boundary_decision["chosen_action"], "copy")
        # decode 就绪：成员进入目标实例的待加入队列（列车发射与完成结
        # 算由冒烟矩阵覆盖）。
        self.assertIn(runtime,
                      scheduler.instances[1].pending_decode_ready)
        # 边界窗口正常闭合：无阶段引用、无请求预约残留。
        self.assertIsNone(runtime.joint_stage_reference_id)
        self.assertEqual(
            kv.session_stage_references(runtime.session_id), ())
        self.assertNotIn("r0", kv._reservations)

    def test_recompute_only_mask_arm_drains_to_recompute_commit(self):
        """{recompute} 臂（C4 接通）：真实 drain 边界在允许面内选中
        recompute@j——D_COMMITTED、静默重建元数据落账（重建基数/抑制标
        志/首 token 待消费恰一次）、无阶段引用/预约残留；decode 就绪
        入队（列车发射与静默重建 span 由 C4 专项测试覆盖，此处钉边界
        提交面）。"""
        scheduler, kv, runtime = self._masked_shell("recompute")
        scheduler._on_prefill_drain(runtime.request_id, 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.joint_decode_action, "recompute")
        self.assertFalse(runtime.joint_decode_is_continuation)
        self.assertFalse(runtime.boundary_decision["fallback"])
        # 静默重建元数据：重建基数 = H_B、抑制采样/输出、首 token 已
        # 发待首个 D 前向消费（恰一次——二次消费 fail-closed）。
        self.assertEqual(
            runtime.joint_decode_rebuild_tokens,
            runtime.history_tokens_before + runtime.joint_input_tokens)
        self.assertTrue(runtime.joint_decode_rebuild_pending)
        self.assertTrue(runtime.joint_suppress_sampling)
        self.assertTrue(runtime.joint_suppress_output)
        self.assertEqual(runtime.joint_prefill_stage, "context_rebuild")
        first_token = runtime.joint_boundary_first_token
        self.assertTrue(first_token["issued"])
        self.assertFalse(first_token["consumed"])
        self.assertEqual(first_token["rng_advances"], 1)
        self.assertTrue(first_token["rebuild_sampling_suppressed"])
        # 阶段工作披露 = recompute 模板（H_B 静默重建 + n̂_B 剩余前向）。
        stage_work = runtime.boundary_decision["stage_work"]
        self.assertEqual(
            stage_work["rebuild_tokens"],
            runtime.joint_decode_rebuild_tokens)
        self.assertTrue(stage_work["rebuild_silent"])
        self.assertIn(runtime, scheduler.instances[
            runtime.decode_instance_index].pending_decode_ready)
        # 边界窗口闭合：无阶段引用、无请求预约残留。
        self.assertIsNone(runtime.joint_stage_reference_id)
        self.assertEqual(
            kv.session_stage_references(runtime.session_id), ())
        self.assertNotIn("r0", kv._reservations)

    def test_any_exception_after_acquire_releases_reference_and_reraises(
            self):
        """边界块整体异常安全（故障注入，不依赖具体异常类型）：自
        acquire 起的任何异常先幂等释放阶段引用再重抛——fail-closed
        语义不变、无引用泄漏、请求不悬挂在持引用的 D_DECISION_PENDING。"""
        scheduler, kv, runtime = self._masked_shell("all")
        import online.sh30_online_scheduler as sh30
        original = sh30.select_decode_target_and_action

        def _boom(**_kwargs):
            raise RuntimeError("injected boundary failure")

        sh30.select_decode_target_and_action = _boom
        try:
            with self.assertRaises(RuntimeError):
                scheduler._on_prefill_drain(runtime.request_id, 1000)
        finally:
            sh30.select_decode_target_and_action = original
        self.assertEqual(runtime.joint_stage, "D_DECISION_PENDING")
        self.assertIsNone(runtime.joint_stage_reference_id)
        self.assertEqual(
            kv.session_stage_references(runtime.session_id), ())


if __name__ == "__main__":
    unittest.main()
