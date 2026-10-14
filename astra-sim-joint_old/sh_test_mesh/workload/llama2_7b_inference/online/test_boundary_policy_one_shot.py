#!/usr/bin/env python3
"""test_boundary_policy_one_shot.py -- JOINT_BOUNDARY_POLICY 一次映射
对照臂钉测（验收纪律第 2 类"一次映射 vs 两阶段闭环对照"仪器面，
2026-09-28）。

仪器合同（定义源 joint/joint_config.py；消费 online/sh30_online_scheduler
.py；测量仪器模式，仿 JOINT_FORCE_* 钩子先例）：

* env ``JOINT_BOUNDARY_POLICY``：缺省（未设）= off = 两阶段闭环零漂移
  （决策行无新键）；唯一合法非缺省值 ``p-continuation`` = 一次映射对照
  臂——D 边界不做二次全候选重估，直接按 P 准入
  ``estimate_P_with_continuation`` 披露的默认延续形态提交（decode_
  instance = prefill_instance、完整型默认延续 = stay@i_P / 增量型
  （P=remote-read）= remote-read@i_P）；非法值/空值 fail-closed；
* 生效时 D 决策行以专用披露键 ``boundary_policy`` 披露（off 时键缺席
  = 行与既有钉测逐位一致）；candidates 空（无候选遍历）、cost_ns 真值
  NA（不虚构候选代价）；
* 一次映射下 D 边界实例恒等于 P 实例、无迁移腿；完成归属/构图器/
  merge 链路与两阶段延续分支同路（同一 commit 路径）；
* 已知终止（_joint_boundary_terminal）仍直接进结算，不做 D 选点；
* 掩码一致性（joint_config __post_init__ fail-closed）：p-continuation
  要求掩码放行 stay；与 probe 强制钩子互斥武装（同设 = 启动失败）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_boundary_policy_one_shot.py -q
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    KVCacheManager,
    build_instances,
)

from joint.joint_config import (  # noqa: E402
    BOUNDARY_POLICIES,
    BOUNDARY_POLICY_OFF,
    BOUNDARY_POLICY_P_CONTINUATION,
    JointConfigError,
    parse_joint_config,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from online.sh30_online_scheduler import (  # noqa: E402
    BOUNDARY_POLICY_ENV,
    FORCE_BOUNDARY_ACTION_ENV,
    FORCE_ENTRY_ACTION_ENV,
    Sh30OnlineScheduler,
    _check_boundary_policy_instrument_conflict,
)
from online.test_decision_route_serialization import (  # noqa: E402
    _GraphStub,
    _decision_shell,
    _make_runtime,
)


# ---------------------------------------------------------------- env 解析 --

class BoundaryPolicyConfigTest(unittest.TestCase):
    """JOINT_BOUNDARY_POLICY 解析合同（joint_config 单一裁决点）。"""

    def test_value_domain(self):
        self.assertEqual(
            BOUNDARY_POLICIES,
            (BOUNDARY_POLICY_OFF, BOUNDARY_POLICY_P_CONTINUATION))
        self.assertEqual(BOUNDARY_POLICY_OFF, "off")
        self.assertEqual(BOUNDARY_POLICY_P_CONTINUATION, "p-continuation")
        self.assertEqual(BOUNDARY_POLICY_ENV, "JOINT_BOUNDARY_POLICY")

    def test_default_off_when_unset(self):
        self.assertEqual(parse_joint_config({}).boundary_policy, "off")

    def test_explicit_off_accepted(self):
        self.assertEqual(
            parse_joint_config({BOUNDARY_POLICY_ENV: "off"}).boundary_policy,
            "off")

    def test_p_continuation_accepted(self):
        config = parse_joint_config(
            {BOUNDARY_POLICY_ENV: "p-continuation"})
        self.assertEqual(config.boundary_policy, "p-continuation")
        self.assertEqual(
            config.manifest_dict()["boundary_policy"],
            "p-continuation")
        self.assertEqual(
            config.manifest_dict()["source_env"][BOUNDARY_POLICY_ENV],
            "p-continuation")

    def test_manifest_discloses_off_by_default(self):
        self.assertEqual(
            parse_joint_config({}).manifest_dict()["boundary_policy"],
            "off")

    def test_illegal_value_fails_closed(self):
        for bad in ("continuation", "P-CONTINUATION", "two-phase", "on"):
            with self.assertRaises(JointConfigError, msg=bad):
                parse_joint_config({BOUNDARY_POLICY_ENV: bad})

    def test_empty_and_whitespace_fail_closed(self):
        for bad in ("", " ", "p-continuation "):
            with self.assertRaises(JointConfigError, msg=repr(bad)):
                parse_joint_config({BOUNDARY_POLICY_ENV: bad})

    def test_mask_excluding_stay_rejected(self):
        # 一次映射 × 掩码一致性（fail-closed）：完整型 P 动作的默认延续
        # 恒为 stay@i_P——掩码剔除 stay 时该组合启动即拒。
        with self.assertRaises(JointConfigError) as ctx:
            parse_joint_config({
                BOUNDARY_POLICY_ENV: "p-continuation",
                "JOINT_ALLOWED_ACTIONS": "copy,recompute",
            })
        self.assertIn("stay", str(ctx.exception))

    def test_mask_admitting_stay_accepted(self):
        config = parse_joint_config({
            BOUNDARY_POLICY_ENV: "p-continuation",
            "JOINT_ALLOWED_ACTIONS": "stay,recompute",
        })
        self.assertEqual(config.boundary_policy, "p-continuation")

    def test_quota_combination_accepted(self):
        # 与配额开关正交（boundary 提交路径与配额准入门无耦合）。
        config = parse_joint_config({
            BOUNDARY_POLICY_ENV: "p-continuation",
            "JOINT_QUOTA_MODE": "static",
        })
        self.assertEqual(config.boundary_policy, "p-continuation")
        self.assertEqual(config.quota_mode, "static")


# ------------------------------------------------------------ 仪器互斥裁决 --

class InstrumentConflictTest(unittest.TestCase):
    """一次映射对照臂 × probe 强制钩子互斥武装（fail-closed）。"""

    def test_off_coexists_with_hooks(self):
        _check_boundary_policy_instrument_conflict("off", "copy", None)
        _check_boundary_policy_instrument_conflict("off", None, "stay")

    def test_p_continuation_alone_is_legal(self):
        _check_boundary_policy_instrument_conflict(
            BOUNDARY_POLICY_P_CONTINUATION, None, None)

    def test_p_continuation_with_boundary_hook_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            _check_boundary_policy_instrument_conflict(
                BOUNDARY_POLICY_P_CONTINUATION, "copy", None)
        self.assertIn(FORCE_BOUNDARY_ACTION_ENV, str(ctx.exception))
        self.assertIn(BOUNDARY_POLICY_ENV, str(ctx.exception))

    def test_p_continuation_with_entry_hook_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            _check_boundary_policy_instrument_conflict(
                BOUNDARY_POLICY_P_CONTINUATION, None, "stay")
        self.assertIn(FORCE_ENTRY_ACTION_ENV, str(ctx.exception))

    def test_init_wires_the_conflict_check(self):
        import inspect
        source = inspect.getsource(Sh30OnlineScheduler.__init__)
        self.assertIn("_check_boundary_policy_instrument_conflict", source)


# ------------------------------------------------------- 全链 drain 夹具 --

class _TrainGraphStub(_GraphStub):
    """发射面 stub：列车 + 完成批（同 test_c4 全链路夹具）。"""

    last_train_plan = None
    pending_store_tails = {}

    def emit_iteration_train(self, train_plan):
        return {
            "drain_members": {},
            "exit_members": {
                member["request_id"]: {
                    0: "exit_{}".format(member["request_id"]),
                    1: "exit_{}".format(member["request_id"])}
                for member in train_plan["exit_members"]
            },
            "sentinel_members": {},
            "block_ends": {0: "end_0", 1: "end_1"},
            "eviction_watches": [],
        }

    def emit_completion_batch(self, plan, *, block_end_stage="seg2"):
        return {"merge_done_members": []}


def _drain_shell(config_env=None, decode_length=8):
    """一次映射对照臂全链路基底：真 KVCacheManager + 真 drain 链，
    P = stay@实例 0（完整型）。config_env 注入 parse_joint_config。"""
    hardware = _tiny_hardware()
    model = _tiny_model()
    topology = _two_instance_topology(hardware)
    kv = KVCacheManager(topology, model)
    scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                topology=topology)
    scheduler.graph = _TrainGraphStub()
    scheduler.next_request = {}
    scheduler._runtime_index = {}
    scheduler.runtimes = [None]
    scheduler.completed_requests = 0
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(next_trigger_type="human",
                        inter_request_interval_ns=1000)])
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    scheduler._batch["watches"] = []
    scheduler._batch["tick"] = 2000
    scheduler._train_max_iter = 0
    scheduler._train_instance_index = {}
    scheduler._pending_eviction_watches = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler._stalled_by_instance = {}
    scheduler._pending_merge_alarms = {}
    runtime = _make_runtime("r0", decode_length=decode_length)
    scheduler.runtime_by_request_id["r0"] = runtime
    scheduler.next_request["r0"] = None
    scheduler._runtime_index["r0"] = 0
    scheduler.runtimes.append(runtime)
    kv.reserve_request_capacity(
        request_id="r0", session_id=runtime.session_id, instance_index=0,
        final_context_tokens=runtime.prefill_context_tokens)
    kv.prepare_prefill(
        session_id=runtime.session_id, target_instance_index=0,
        history_tokens=0, trigger_request_id="r0")
    kv.expand_prefill(
        session_id=runtime.session_id, instance_index=0,
        context_tokens=runtime.prefill_context_tokens,
        trigger_request_id="r0")
    runtime.prefill_instance_index = 0
    runtime.joint_stage = "P_RUNNING"
    scheduler.instances[0].qp.append(runtime)
    runtime.queued_chunk_load_ns = 0
    config = parse_joint_config(config_env or {})
    scheduler.joint_config = config
    scheduler._joint_mode = config.scheduler_mode
    runtime.joint_action = "stay"
    return scheduler, kv, runtime


def _decode_rows(scheduler):
    return [row for row in scheduler.online_log_rows
            if row.get("kind") == "decode"]


# ------------------------------------------------------------- off 零漂移 --

class OffZeroDriftTest(unittest.TestCase):
    """不设 env：D 边界行为与决策行形态逐位不变（无新键）。"""

    def test_off_boundary_row_has_no_policy_key(self):
        scheduler, _kv, runtime = _drain_shell()
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        boundary = runtime.boundary_decision
        self.assertNotIn("boundary_policy", boundary)
        # 两阶段形态：全候选遍历披露（candidates 非空、cost 数值、
        # is_continuation 来自选择器）。
        self.assertTrue(boundary["candidates"])
        self.assertIsInstance(boundary["cost_ns"], int)
        self.assertFalse(boundary["fallback"])
        rows = _decode_rows(scheduler)
        self.assertEqual(len(rows), 1)
        serialized = rows[0]["decision"]["boundary_decision"]
        self.assertNotIn("boundary_policy", serialized)
        self.assertTrue(serialized["candidates"])

    def test_off_config_reports_off(self):
        scheduler, _kv, _runtime = _drain_shell()
        self.assertEqual(
            scheduler.joint_config.boundary_policy, BOUNDARY_POLICY_OFF)


# ------------------------------------------------- p-continuation 全链路 --

class PContinuationOneShotTest(unittest.TestCase):
    """一次映射对照臂全链路（真 drain）：D 实例恒等于 P 实例、无迁移腿、
    决策行专用披露键。"""

    def test_one_shot_commits_default_continuation_at_p_instance(self):
        scheduler, kv, runtime = _drain_shell(
            config_env={BOUNDARY_POLICY_ENV: "p-continuation"})
        scheduler._on_prefill_drain("r0", 1000)
        # D 边界实例恒等于 P 实例（一次映射语义）；动作 = 完整型默认
        # 延续 stay@i_P。
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index,
                         runtime.prefill_instance_index)
        self.assertEqual(runtime.completion_instance_index,
                         runtime.prefill_instance_index)
        self.assertEqual(runtime.joint_decode_action, "stay")
        self.assertTrue(runtime.joint_decode_is_continuation)
        # 无迁移腿：P→D 交接 = 同实例 local_hit 复用（零 shard——无
        # 物理 NoC 传输腿）或 None；decode 相无逐出。
        transfer = runtime.prefill_decode_transfer
        if transfer is not None:
            self.assertEqual(transfer.kind, "local_hit")
            self.assertEqual(transfer.shards, ())
        self.assertEqual(runtime.decode_evictions, ())
        # 会话照常折叠在 P 实例（与两阶段延续分支同路）。
        session = kv._sessions[runtime.session_id]
        self.assertEqual(session.instance_index, 0)
        # 决策记录：专用披露键在位、无候选遍历、代价真值 NA、无回退。
        boundary = runtime.boundary_decision
        self.assertEqual(boundary["boundary_policy"], "p-continuation")
        self.assertEqual(boundary["chosen_instance_index"], 0)
        self.assertEqual(boundary["chosen_action"], "stay")
        self.assertTrue(boundary["is_continuation"])
        self.assertIsNone(boundary["cost_ns"])
        self.assertEqual(boundary["candidates"], [])
        self.assertFalse(boundary["fallback"])

    def test_one_shot_decision_row_discloses_policy(self):
        scheduler, _kv, _runtime = _drain_shell(
            config_env={BOUNDARY_POLICY_ENV: "p-continuation"})
        scheduler._on_prefill_drain("r0", 1000)
        rows = _decode_rows(scheduler)
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertEqual(
            decision["boundary_decision"]["boundary_policy"],
            "p-continuation")
        self.assertEqual(decision["boundary_decision"]["candidates"], [])
        self.assertEqual(decision["decode_instance_index"],
                         decision["boundary_decision"]
                         ["chosen_instance_index"])

    def test_one_shot_no_rebuild_suppression(self):
        # 完整型默认延续 = stay（非 recompute）：无静默重建面。
        scheduler, _kv, runtime = _drain_shell(
            config_env={BOUNDARY_POLICY_ENV: "p-continuation"})
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_decode_rebuild_tokens, 0)
        self.assertFalse(runtime.joint_decode_rebuild_pending)
        self.assertFalse(runtime.joint_suppress_sampling)
        self.assertFalse(runtime.joint_suppress_output)

    def test_one_shot_joins_decode_train(self):
        # 构图器链路与两阶段延续分支同路：成员进 pending_decode_ready、
        # 列车发射后 D_RUNNING。
        scheduler, _kv, runtime = _drain_shell(
            config_env={BOUNDARY_POLICY_ENV: "p-continuation"})
        scheduler._on_prefill_drain("r0", 1000)
        self.assertIn(runtime,
                      scheduler.instances[0].pending_decode_ready)
        scheduler._plan_and_emit_trains(2000)
        self.assertEqual(runtime.joint_stage, "D_RUNNING")


# --------------------------------------------- 增量型（P=remote-read）臂 --

HISTORY_TOKENS = 96        # 增量型基础区间 H（home 实例）
INPUT_TOKENS = 16          # 增量型 U（i_P 工作副本）
HOME_INSTANCE = 2          # 三实例几何的权威基础驻留实例
P_INSTANCE = 0             # i_P（P = remote-read 执行实例）


def _three_instance_topology(hardware):
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _three_hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _increment_drain_shell(config_env=None, decode_length=8):
    """增量型一次映射全链路基底（真 KVCacheManager + 真 drain 链）：
    基础历史 H 在 home=实例 2（turn-0），P = remote-read@实例 0（U 增量
    工作副本）——三实例几何同 test_c4 边界事务夹具。"""
    hardware = _three_hardware()
    model = _tiny_model()
    topology = _three_instance_topology(hardware)
    kv = KVCacheManager(topology, model)
    kv.prepare_prefill(
        session_id="r0_session", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    kv.expand_prefill(
        session_id="r0_session", instance_index=HOME_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    kv.mark_complete("r0_session", 100)
    scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                topology=topology)
    scheduler.graph = _TrainGraphStub()
    scheduler.next_request = {}
    scheduler._runtime_index = {}
    scheduler.runtimes = [None]
    scheduler.completed_requests = 0
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(next_trigger_type="human",
                        inter_request_interval_ns=1000)])
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    scheduler._batch["watches"] = []
    scheduler._batch["tick"] = 2000
    scheduler._train_max_iter = 0
    scheduler._train_instance_index = {}
    scheduler._pending_eviction_watches = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler._stalled_by_instance = {}
    scheduler._pending_merge_alarms = {}
    runtime = _make_runtime(
        "r0", prefill_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
        decode_length=decode_length, history_tokens_before=HISTORY_TOKENS)
    scheduler.runtime_by_request_id["r0"] = runtime
    scheduler.next_request["r0"] = None
    scheduler._runtime_index["r0"] = 0
    scheduler.runtimes.append(runtime)
    kv.reserve_request_capacity(
        request_id="r0", session_id=runtime.session_id,
        instance_index=P_INSTANCE,
        final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS)
    kv.prepare_prefill(
        session_id=runtime.session_id, target_instance_index=P_INSTANCE,
        history_tokens=HISTORY_TOKENS, trigger_request_id="r0",
        action="remote-read")
    kv.expand_prefill(
        session_id=runtime.session_id, instance_index=P_INSTANCE,
        context_tokens=INPUT_TOKENS, trigger_request_id="r0")
    runtime.prefill_instance_index = P_INSTANCE
    runtime.joint_stage = "P_RUNNING"
    scheduler.instances[P_INSTANCE].qp.append(runtime)
    runtime.queued_chunk_load_ns = 0
    config = parse_joint_config(config_env or {})
    scheduler.joint_config = config
    scheduler._joint_mode = config.scheduler_mode
    runtime.joint_action = "remote-read"
    return scheduler, kv, runtime


class PContinuationIncrementRemoteTest(unittest.TestCase):
    """增量型（P=remote-read）一次映射：延续动作 = remote-read@i_P，
    沿用 P 入口已建的 credit plan 延续分支（无 U 迁移腿）。"""

    def test_increment_one_shot_continues_remote_at_p_instance(self):
        scheduler, kv, runtime = _increment_drain_shell(
            config_env={BOUNDARY_POLICY_ENV: "p-continuation"})
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        # D 实例恒等于 P 实例；延续动作 = remote-read（增量型默认延续）。
        self.assertEqual(runtime.decode_instance_index, P_INSTANCE)
        self.assertEqual(runtime.completion_instance_index, P_INSTANCE)
        self.assertEqual(runtime.joint_decode_action, "remote-read")
        self.assertTrue(runtime.joint_decode_is_continuation)
        # P 入口已建的读计划延续分支照常建立（同一冻结点）。
        self.assertIsNotNone(runtime.remote_read_credit_plan)
        plan = runtime.remote_read_credit_plan
        self.assertEqual(plan["exec_instance"], P_INSTANCE)
        self.assertEqual(plan["home_instance"], HOME_INSTANCE)
        # 决策记录：专用披露键 + 无候选遍历 + 无回退。
        boundary = runtime.boundary_decision
        self.assertEqual(boundary["boundary_policy"], "p-continuation")
        self.assertEqual(boundary["chosen_instance_index"], P_INSTANCE)
        self.assertIsNone(boundary["cost_ns"])
        self.assertEqual(boundary["candidates"], [])
        self.assertFalse(boundary["fallback"])
        # 会话：无 U 迁移（工作副本仍为 i_P 的增量形态、base 记录在
        # home 不变——纯远读不搬运）。
        session = kv._sessions[runtime.session_id]
        self.assertEqual(session.instance_index, P_INSTANCE)
        self.assertIsNotNone(session.working_kind)
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)


# ------------------------------------------------------- 已知终止不动 --

class TerminalSettlementUntouchedTest(unittest.TestCase):
    """已知终止仍直接结算：不做 D 选点、决策行无 policy 键。"""

    def test_terminal_settles_regardless_of_policy(self):
        scheduler, _kv, runtime = _drain_shell(
            config_env={BOUNDARY_POLICY_ENV: "p-continuation"},
            decode_length=0)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "SETTLE_DONE")
        self.assertEqual(runtime.decode_instance_index,
                         runtime.prefill_instance_index)
        boundary = runtime.boundary_decision
        self.assertTrue(boundary.get("terminal"))
        self.assertNotIn("boundary_policy", boundary)
        kinds = [row["kind"] for row in scheduler.online_log_rows]
        self.assertIn("boundary_terminal", kinds)
        self.assertNotIn("decode", kinds)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
