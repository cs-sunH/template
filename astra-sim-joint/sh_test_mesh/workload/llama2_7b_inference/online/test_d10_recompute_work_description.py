#!/usr/bin/env python3
"""test_d10_recompute_work_description.py -- D10 recompute 语义重定义的
执行侧（在线工作描述）零后端单测。

源：joint修改文档.md D10 条目（v2.0 §6.2，2026-09-26）。钉住面：

  1. ``_apply_joint_recompute_work_description``（准入点改写，取代
     R13 缺失层折算块）：P 初始 = 完整 H+U 重放——重建基数 = 整份
     已保存上下文 H（LOCAL/PARTIAL/REMOTE × 驻留/异地四组同口径）、
     span 基恒 0、joint_prefill_work = H + 本轮 input（旧 U 属 H 的
     重建基数，不把旧 U 再算成新 prefill——token 序列各恰一次）、
     prefill_tokens_to_process / remaining_chunks 同步改写。
  2. 阶段字段与 token/RNG 元数据：joint_prefill_stage =
     "prefill_replay"（P 完整重放；D 边界静默重建 context_rebuild +
     suppress_sampling/output=True 为 D18/D19 落地点，两语义不混用）、
     重建段不重复输出历史 token ⇒ suppress 标志 False（P 重放末
     chunk 以正常 prefill 采样生成本轮首 token 恰一次、RNG 推进一次）。
  3. 工作描述三方同源：runtime、plan_dict（构图器消费）、
     _joint_working_context（drain 物化覆盖 H+U）。
  4. 非重算动作恒缺省值（0/None/False/False），键恒在。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_d10_recompute_work_description.py
      （或 pytest 同路径）
"""
import math
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import SessionKVView  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineRequestRuntime,
    Sh30OnlineScheduler,
)


def _scheduler(p_chunk: int = 64) -> Sh30OnlineScheduler:
    """绕过 __init__ 装配被测方法所需属性面（p_chunk 为唯一依赖）。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.p_chunk = p_chunk
    return scheduler


def _runtime(*, history_tokens: int = 100, input_tokens: int = 50,
             decode_length: int = 8) -> _OnlineRequestRuntime:
    return _OnlineRequestRuntime({
        "request_id": "r1",
        "session_id": "s1",
        "turn_index": 1,
        "queue_index": 0,
        "prefill_length": input_tokens,
        "decode_length": decode_length,
        "history_tokens_before": history_tokens,
        "prefill_context_tokens": history_tokens + input_tokens,
        "final_context_tokens": history_tokens + input_tokens
        + decode_length,
    })


def _session(*, home: int = 0, resident: int = 0, location: str =
             "local_hbm", prefix: int = 4, history_tokens: int = 100,
             missing: tuple = (0, 0)) -> SessionKVView:
    return SessionKVView(
        session_id="s1", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=(1000, 1000),
        missing_bytes_by_tp_rank=missing)


class D10RecomputeWorkDescriptionTest(unittest.TestCase):
    """准入点工作描述改写：完整 H+U 重放、span 基 0、阶段字段。"""

    def setUp(self):
        self.scheduler = _scheduler(p_chunk=64)

    def test_four_groups_identical_work_description(self):
        # 验收四组：LOCAL/PARTIAL/REMOTE × 驻留/异地——工作描述逐字段
        # 同值（重放不依赖旧 KV；层驻留状态只影响空间，不影响工作
        # 量与 span 基）。
        groups = {
            "local_resident": _session(location="local_hbm", prefix=4),
            "partial_resident": _session(
                location="partial_hbm_remote", prefix=2,
                missing=(500, 500)),
            "local_away": _session(home=1, resident=1,
                                   location="local_hbm", prefix=4),
            "partial_away": _session(
                home=1, resident=1, location="partial_hbm_remote",
                prefix=2, missing=(500, 500)),
            "remote_base": _session(location="remote_memory", prefix=4),
        }
        seen = {}
        for name, session_view in groups.items():
            runtime = _runtime()
            self.scheduler._apply_joint_recompute_work_description(
                runtime, session_view)
            seen[name] = (
                runtime.joint_prefill_work,
                runtime.joint_span_base_context,
                runtime.joint_rebuild_context_tokens,
                runtime.joint_prefill_stage,
                runtime.joint_suppress_sampling,
                runtime.joint_suppress_output,
                runtime.prefill_tokens_to_process,
                runtime.remaining_chunks,
            )
        self.assertEqual(len(set(seen.values())), 1, msg=seen)
        (work, base, rebuild, stage, suppress_sampling, suppress_output,
         to_process, remaining_chunks) = next(iter(seen.values()))
        # token 序列守恒：重建 H=100 + 本轮输入 U=50 各恰一次（旧 U 属
        # H 的重建基数，不再算成新 prefill）。
        self.assertEqual(work, 150)
        self.assertEqual(rebuild, 100)
        self.assertEqual(base, 0)
        self.assertEqual(to_process, 150)
        self.assertEqual(remaining_chunks, math.ceil(150 / 64))
        # 阶段字段：P 完整重放（重建段不输出/不采样，末 chunk 正常
        # 采样本轮首 token 一次 ⇒ 抑制标志 False）。
        self.assertEqual(stage, "prefill_replay")
        self.assertFalse(suppress_sampling)
        self.assertFalse(suppress_output)

    def test_span_base_zero_drives_working_context(self):
        # drain 物化覆盖面：_joint_working_context = joint_prefill_work
        # + 已完成 decode ⇒ prefill 完成点 = H+U（重放写满全上下文）。
        runtime = _runtime()
        runtime.joint_action = "recompute"
        self.scheduler._apply_joint_recompute_work_description(
            runtime, _session())
        self.assertEqual(
            self.scheduler._joint_working_context(runtime), 150)
        runtime.decode_tokens_consumed = 3
        self.assertEqual(
            self.scheduler._joint_working_context(runtime), 153)

    def test_plan_dict_carries_stage_fields(self):
        # 构图器同步：plan_dict 键恒在、值与 runtime 同源。
        runtime = _runtime()
        defaults = runtime.plan_dict()
        # 非重算动作（缺省 stay）恒缺省值。
        self.assertEqual(defaults["joint_rebuild_context_tokens"], 0)
        self.assertIsNone(defaults["joint_prefill_stage"])
        self.assertIs(defaults["joint_suppress_sampling"], False)
        self.assertIs(defaults["joint_suppress_output"], False)
        runtime.joint_action = "recompute"
        self.scheduler._apply_joint_recompute_work_description(
            runtime, _session())
        plan = runtime.plan_dict()
        self.assertEqual(plan["joint_rebuild_context_tokens"], 100)
        self.assertEqual(plan["joint_prefill_stage"], "prefill_replay")
        self.assertIs(plan["joint_suppress_sampling"], False)
        self.assertIs(plan["joint_suppress_output"], False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
