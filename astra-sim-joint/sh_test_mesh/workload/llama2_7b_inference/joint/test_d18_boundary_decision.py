#!/usr/bin/env python3
"""test_d18_boundary_decision.py -- D18 两阶段 P→D 边界决策单元测试。

验收红线逐条落测（joint修改文档 D18 / v2.0 §2/§16/§17）：
  1. 初始候选不做全 D 搜索 —— ``estimate_P_with_continuation`` 单候选
     只估默认延续（本文件以选择器存根证明边界遍历只走
     ``estimate_D_remaining``、且初始入口与边界入口分离）；
  2. 边界只做一次全遍历 —— 选择器对三动作 × N 实例恰好各调一次；
  3. 无第三次选点 —— 选择器之外无其他选点入口（状态机
     D_COMMITTED 后仅资源执行；见 sh30 drain 流水）；
  4. 已完成 P 不重复收费 —— continuation 候选零准备腿、计算段仅
     n̂_B 步 decode 前向；
  5. D 新迁移不回填为 P 期间重叠 —— 切换候选的迁移腿是 t_B 后新
     工作（history_prep_ns > 0 且串行计入关键路径）；
  6. 原地继续恒为候选且平局优先（式 1c）—— 候选面缺失 x0 即
     fail-closed；成本并列必选继续，严格更低才切换；
  7. 已知终止直接结算不选 D（状态机 shell 端到端）。

2026-10-14 远读退役同步：增量族/credit plan/remote off 用例删除
（增量型布局族随 P=remote-read 退役一步删除，D6）；候选面断言
4N→3N。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d18_boundary_decision.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import KVCacheManager  # noqa: E402
from joint.test_joint_mechanisms import (  # noqa: E402
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_ORDER,
    ACTION_STAY,
    ActionCandidate,
    ActionCostBreakdown,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    ServiceFactors,
)
from joint.joint_scheduler import (  # noqa: E402
    JointSchedulerError,
    select_decode_target_and_action,
)
from joint.prediction_config import JointPredictionConfig  # noqa: E402

# tiny model：256 B/token 全层、128 B/token/rank（K+V，4 层）。
KV_TOKEN_BYTES = 128


def _route2(source, target):
    return ((source, target), 1)


def _model(loads):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads,
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route2)


def _load(instance_index, active_ns=0):
    return InstanceLoadView(
        instance_index=instance_index, queued_task_load_ns=0,
        running_task_load_ns=active_ns, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


def _loads(active=(0, 0)):
    return {
        0: _load(0, active_ns=active[0]),
        1: _load(1, active_ns=active[1]),
    }


def _complete_boundary(**overrides):
    """完整型边界（布局恒完整型单族——增量型已随远读退役删除，D6；
    P=stay，H_B=100 token 全量在 i_P=0，home=0）。"""
    values = dict(
        session_id="s", home_instance=0, p_instance_index=0,
        p_action="stay", history_tokens=100,
        working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2,
        decode_steps=8, kv_token_bytes=KV_TOKEN_BYTES)
    values.update(overrides)
    return PDBoundaryView(**values)


class D18ContinuationPricingTest(unittest.TestCase):
    """式 1b continuation 计价：已完成 P 不重复收费。"""

    def test_continuation_charges_no_completed_p_work(self):
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY)
        self.assertTrue(candidate.applicable)
        self.assertTrue(candidate.is_continuation)
        # 沉没纪律：零准备腿（P 的历史准备/传输已沉没），计算段仅
        # n̂_B 步 decode 前向（8 步 × 2.0 ns × factor 1.0）。
        self.assertEqual(candidate.breakdown.history_prep_ns, 0)
        self.assertEqual(candidate.breakdown.compute_ns, 16)
        self.assertEqual(candidate.cost_ns, 16)
        self.assertIn("boundary_stage=D", candidate.breakdown.notes)
        self.assertIn(
            "is_continuation=True", candidate.breakdown.notes)

    def test_decode_growth_footprint_is_step_exact(self):
        """decode 增长逐 rank = 每 token KV 字节 × n̂_B（精确非比例）：
        宽裕容量零驱逐等待；容量收紧后增长产生真实空间等待。"""
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 0)
        # 容量收紧到增长放不下 → 空间等待非零（深缺口计价细节由 R3'
        # 既有钉测看护，此处只钉"增长足迹真实进容量账"）。
        tight = _loads(active=(0, 0))
        tight[0] = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=10 ** 7, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(512, 512),
            reclaimable_bytes_by_tp_rank=(0, 0))
        tight_model = _model(tight)
        squeezed = tight_model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY)
        self.assertGreater(
            squeezed.breakdown.eviction_wait_ns,
            candidate.breakdown.eviction_wait_ns)


class D18SwitchPricingTest(unittest.TestCase):
    """式 1b 切换计价：迁移腿是 t_B 后的新工作，不回填 P 期重叠。"""

    def test_switch_migration_is_new_boundary_work(self):
        model = _model(_loads())
        switch = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1,
            action="copy")
        self.assertTrue(switch.applicable)
        self.assertFalse(switch.is_continuation)
        # 迁移腿 = 新准备（H_B 工作副本 0→1 的 NoC copy 三腿），真实
        # 计入关键路径（串行于 decode 前向），不是零成本改名。
        self.assertGreater(switch.breakdown.history_prep_ns, 0)
        self.assertIn(
            f"boundary_move_bytes={KV_TOKEN_BYTES * 100 * 2}",
            switch.breakdown.notes)
        self.assertEqual(
            switch.cost_ns,
            switch.breakdown.target_wait_ns
            + max(switch.breakdown.history_prep_ns,
                  switch.breakdown.eviction_wait_ns)
            + switch.breakdown.compute_ns)

    def test_switch_costs_more_than_continuation_at_equal_load(self):
        """同负载下切换（含迁移）严格贵于原地继续——迁移不虚构重叠。"""
        model = _model(_loads())
        continuation = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action=ACTION_STAY)
        switch = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1,
            action="copy")
        self.assertGreater(switch.cost_ns, continuation.cost_ns)


class D18BoundaryApplicabilityTest(unittest.TestCase):
    """边界候选面：按布局物理语义划面（无容量/距离预过滤；完整型
    动作面——recompute 静默重建 / copy 迁移均可选）。"""

    def test_complete_family_face(self):
        model = _model(_loads())
        # i_P 上：stay（= x0）+ recompute（原地重建）；迁移目标上：
        # recompute / copy。
        at_p = model.applicable_D_remaining_actions(
            _complete_boundary(), 0)
        at_other = model.applicable_D_remaining_actions(
            _complete_boundary(), 1)
        self.assertEqual(
            [ok for ok, _ in at_p],
            [True, True, False])
        self.assertEqual(
            [ok for ok, _ in at_other],
            [False, True, True])

    def test_boundary_inapplicable_reasons_are_physical(self):
        """不可用候选的理由串 = 物理语义（无协议级禁选，理由单源
        为布局物理语义，与 P 入口 applicable_actions 同模板）。"""
        model = _model(_loads())
        # 完整型 j==i_P：copy 退化（无缺失区间）。
        complete_at_p = dict(zip(
            ACTION_ORDER, model.applicable_D_remaining_actions(
                _complete_boundary(), 0)))
        self.assertEqual(
            complete_at_p["copy"], (False, "no history to copy"))

    def test_action_mask_applies_at_boundary(self):
        """M2 掩码在边界同为候选集上限（命中即剔除）。"""
        model = _model(_loads())
        model.allowed_actions = frozenset({"copy"})
        at_other = model.applicable_D_remaining_actions(
            _complete_boundary(), 1)
        self.assertEqual(
            [ok for ok, _ in at_other],
            [False, False, True])
        model.allowed_actions = frozenset({"stay"})
        masked = model.applicable_D_remaining_actions(
            _complete_boundary(), 1)
        self.assertEqual([ok for ok, _ in masked],
                         [False, False, False])


class _CountingModel:
    """选择器存根：恰好一次全遍历 + 式 1c 平局语义证明。

    故意**不实现** ``estimate_P_with_continuation``——边界选择若误走
    初始入口立即 AttributeError（初始候选不做全 D 搜索的负证）。

    E0 仲裁面对齐（生产 ``_projection_arbitrate`` 直访）：存根代价表
    本身即精确值（配置态即精确态），``_resolved_prediction_config``
    回缺省配置、``reprice_candidate_exact`` 恒等返回；适用候选携带
    零值 breakdown（空 projection_stats = 无窗口近似触发位）。
    """

    def __init__(self, costs):
        self.loads = {0: object(), 1: object()}
        self.costs = costs
        self.calls = []

    def _resolved_prediction_config(self):
        return JointPredictionConfig()

    def reprice_candidate_exact(self, candidate):
        return candidate

    @staticmethod
    def _stub_breakdown():
        return ActionCostBreakdown(
            target_wait_ns=0, history_prep_ns=0, eviction_wait_ns=0,
            compute_ns=0, merge_ns=0,
            contention_divisor=1, hops=0)

    def estimate_D_remaining(self, *, boundary, instance_index, action,
                             **_disclosure):
        # D17：选择器透传 decision_epoch / inherited_obligation_ids 披
        # 露通道——存根以 **kwargs 接收（纯披露，不影响代价表）。
        self.calls.append((instance_index, action))
        entry = self.costs[(instance_index, action)]
        if entry is None:
            return ActionCandidate(
                instance_index=instance_index, action=action,
                applicable=False, inapplicable_reason="stub", cost_ns=None,
                breakdown=self._stub_breakdown())
        cost, is_continuation = entry
        return ActionCandidate(
            instance_index=instance_index, action=action, applicable=True,
            inapplicable_reason=None, cost_ns=cost,
            breakdown=self._stub_breakdown(),
            is_continuation=is_continuation)


def _complete_costs(continuation_cost, switch_cost):
    """完整型两实例代价表：(0, stay)=x0、(1, copy)=迁移，其余不适用。"""
    return {
        (0, "stay"): (continuation_cost, True),
        (0, "recompute"): None,
        (0, "copy"): None,
        (1, "stay"): None,
        (1, "recompute"): None,
        (1, "copy"): (switch_cost, False),
    }


class D18BoundarySelectionTest(unittest.TestCase):
    """边界选择：恰好一次全遍历 + 原地继续恒为候选且平局优先。"""

    def test_exactly_one_full_traversal_no_initial_search(self):
        model = _CountingModel(_complete_costs(100, 90))
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary())
        # 三动作 × 2 实例 = 恰好 6 次边界估价；无任何初始入口调用。
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(
            sorted(model.calls),
            sorted((i, a) for i in (0, 1) for a in ACTION_ORDER))
        self.assertEqual(record.diagnostics["decision_stage"], "D")

    def test_equal_cost_prefers_continuation_eq_1c(self):
        model = _CountingModel(_complete_costs(100, 100))
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary())
        self.assertTrue(record.chosen.is_continuation)
        self.assertEqual(record.chosen.instance_index, 0)
        self.assertEqual(record.chosen.action, ACTION_STAY)

    def test_strict_improvement_switches_eq_1c(self):
        model = _CountingModel(_complete_costs(100, 90))
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary())
        self.assertFalse(record.chosen.is_continuation)
        self.assertEqual(record.chosen.instance_index, 1)
        self.assertEqual(record.chosen.action, "copy")
        self.assertEqual(record.diagnostics["continuation_cost_ns"], 100)

    def test_missing_continuation_record_fails_closed(self):
        costs = _complete_costs(100, 90)
        costs[(0, "stay")] = None  # x0 缺席 = 候选面结构性缺陷
        model = _CountingModel(costs)
        with self.assertRaises(JointSchedulerError):
            select_decode_target_and_action(
                mode="joint", cost_model=model,
                boundary=_complete_boundary())


# ------------------------------------------------------------ 状态机 --

from online.test_decision_route_serialization import (  # noqa: E402
    _GraphStub,
    _decision_shell,
    _make_runtime,
)


class _CompletionGraphStub(_GraphStub):
    """发射面 stub：补完成批（merge 零传输 → 无 merge watch），并记录
    触发门锚段（D18 终端结算 = seg1 drain 块末）。"""

    last_block_end_stage = None

    def emit_completion_batch(self, plan, *, block_end_stage="seg2"):
        self.last_block_end_stage = block_end_stage
        return {"merge_done_members": []}


def _drain_ready_scheduler(decode_length, *, instances=2):
    """构造 r0 已在实例 0 完整 prefill 就绪态的调度器壳（真 drain）。"""
    hardware = _tiny_hardware()
    model = _tiny_model()
    topology = _two_instance_topology(hardware)
    kv = KVCacheManager(topology, model)
    scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                topology=topology)
    scheduler.graph = _CompletionGraphStub()
    # 完成事务所需账本（_complete_requests；following=None 终轮形态）。
    from types import SimpleNamespace
    scheduler.next_request = {}
    scheduler._runtime_index = {}
    scheduler.runtimes = [None]
    scheduler.completed_requests = 0
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(next_trigger_type="human",
                        inter_request_interval_ns=1000)])
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
    return scheduler, kv, runtime


class D18StateMachineTest(unittest.TestCase):
    """在线状态机（v2.0 §16.4）：drain 边界的真实转移。"""

    def test_non_terminal_enter_d_decision_and_continues_in_place(self):
        """P_COMPUTE_DONE -> D_DECISION_PENDING -> D_COMMITTED；均衡负
        载下平局优先继续（decode 实例 = P 实例、is_continuation）。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 0)
        self.assertTrue(runtime.joint_decode_is_continuation)
        self.assertEqual(runtime.joint_decode_action, "stay")
        self.assertFalse(runtime.boundary_decision["fallback"])
        # W5 消费切换重锚（W6 §2）：n̂_B = 条件剩余估计器现值
        # （estimate_conditional——P/D 正式来源；旧 horizon 冷启动 10 退役
        # 为诊断/先验，不再进计价；式 4a 下界由估计器保证）。
        self.assertEqual(
            runtime.boundary_decision["decode_steps"],
            scheduler._joint_length_history_or_default()
            .estimate_conditional(
                session_id=runtime.session_id,
                exposed_progress_steps=runtime.decode_tokens_consumed,
                category=getattr(
                    runtime, "joint_revealed_category", None),
            ).remaining_forward_steps)
        self.assertGreaterEqual(
            runtime.boundary_decision["decode_steps"], 1)
        # 成员进入 decode 实例的就绪队列（加入列车后 D_RUNNING）。
        self.assertIn(runtime,
                      scheduler.instances[0].pending_decode_ready)
        # 决策行披露边界产物。
        decode_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "decode"]
        self.assertEqual(len(decode_rows), 1)
        decision = decode_rows[0]["decision"]
        # 修复轮三 F7：decision_stage 词表四方对齐为 "P"/"D"（P 行、
        # 候选面 notes、ACC、probe 合成行同此）——状态机细节
        # （D_COMMITTED）由 boundary_transaction/stage_epoch 披露。
        self.assertEqual(decision["decision_stage"], "D")
        self.assertTrue(decision["is_continuation"])
        self.assertEqual(
            decision["boundary_decision"]["chosen_instance_index"], 0)

    def test_terminal_settles_directly_without_d_selection(self):
        """已知终止：P_COMPUTE_DONE -> SETTLE_PENDING 直接结算——不选
        D、不入列车、完成事务当场闭合（DONE）。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=0)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "SETTLE_DONE")
        self.assertTrue(runtime.completed)
        self.assertEqual(runtime.decode_instance_index, 0)
        self.assertTrue(runtime.boundary_decision["terminal"])
        # 无 D 选点：无 decode 决策行、未入就绪队列；显式终止披露行。
        self.assertEqual(
            [row for row in scheduler.online_log_rows
             if row["kind"] == "decode"], [])
        self.assertNotIn(
            runtime, scheduler.instances[0].pending_decode_ready)
        self.assertEqual(
            len([row for row in scheduler.online_log_rows
                 if row["kind"] == "boundary_terminal"]), 1)
        # 预约已注销（结算事务完成）；无退出列车 → completion 门锚
        # = drain 块末 seg1（图批锚段语义精确）。
        self.assertNotIn("r0", kv._reservations)
        self.assertEqual(scheduler.graph.last_block_end_stage, "seg1")


    def test_switch_commit_failure_falls_back_to_continuation(self):
        """§16.4 提交失败回退：切换候选（实例 1 轻载）提交时目标容量
        不足（活跃会话占位 = 非合法 victim 深缺口）→ KVCapacityError
        → 回退继续方案（零字节迁移、is_continuation=True）+ 披露行；
        继续态保持合法（§16.4），无引用/预约双份或丢失。

        C4 完整动作面后该夹具把允许面掩成 {copy, stay}（recompute/
        multi-source remote 的目标预约足迹小、不再触发容量失败——回
        退红线的触发形态以 copy 迁移承载；掩码上限语义见 M2/C1）。"""
        from joint.joint_config import parse_joint_config
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler.joint_config = parse_joint_config({
            "JOINT_ALLOWED_ACTIONS": "copy,stay"})
        scheduler._joint_mode = scheduler.joint_config.scheduler_mode
        # 实例 1 驻留 ACTIVE 会话（未 mark_complete，非合法 victim）吃
        # 空容量：free/rank ≈ 5.2e4 < r0 基础足迹 1.31e5 → 目标预约
        # 深缺口 fail。仅使容量不足；负载视图仍轻载（候选面照常选中）。
        kv.prepare_prefill(
            session_id="hog", target_instance_index=1,
            history_tokens=0, trigger_request_id="hog")
        kv.expand_prefill(
            session_id="hog", instance_index=1, context_tokens=7_812_000,
            trigger_request_id="hog")
        # 实例 0 压**在服**负载（active decode 成员进原地接续等待基）
        # 翻转 argmin 到轻载实例 1。D19 等待接口拆分（2026-09-26）后
        # 排队积压不再重压原地接续（不重排队），翻转机制自 D19 起改用
        # 在服流负载——排队积压不再能翻转 argmin（见
        # test_d19_boundary_layout 的不重排队钉测）。
        for seq in range(3):
            hog_load = _make_runtime(f"active0_{seq}")
            scheduler.runtime_by_request_id[f"active0_{seq}"] = hog_load
            scheduler.instances[0].active_decode.append(hog_load)

        scheduler._on_prefill_drain("r0", 1000)

        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertTrue(runtime.boundary_decision["fallback"])
        self.assertEqual(runtime.decode_instance_index, 0)
        self.assertTrue(runtime.joint_decode_is_continuation)
        self.assertEqual(runtime.joint_decode_action, "stay")
        # 会话仍在 P 实例（零字节移动）；实例 1 占用视图零残留
        # （换址从未生效）；drain 终点预约照常注销。
        self.assertEqual(kv._sessions[runtime.session_id].instance_index, 0)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)), (0, 0))
        self.assertNotIn("r0", kv._reservations)
        # 回退披露行 + 成员照常进入继续实例的就绪队列。
        self.assertEqual(
            len([row for row in scheduler.online_log_rows
                 if row["kind"] == "boundary_commit_fallback"]), 1)
        self.assertIn(runtime, scheduler.instances[0].pending_decode_ready)


class DBIPhysicalBaseBoundaryTest(unittest.TestCase):
    """D-BI 回归（2026-09-26，冒烟矩阵 v20-b2 第 2 轮根因）：P→D 边界
    的基础驻留实例必须是**物理基础所在**（``base_instance_index``），
    不是逻辑 home——P→D 迁移（``move_prefill_to_decode``）+ merge stay
    提交使二者合法解耦后，按逻辑 home 判原地继续适用性会把 x0 记录
    整个剔除（"boundary traversal lost the in-place continuation
    record" fail-closed）。远读退役后该夹具以 P=copy 承载（基础在
    home、工作副本建到 i_P——同一物理基础/逻辑 home 解耦几何）。"""

    def _shell_with_migrated_session(self):
        hardware = _tiny_hardware()
        model = _tiny_model()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                    topology=topology)
        # turn-0：prefill@0 → decode 迁 1 → merge stay 提交
        #（终驻实例 1，逻辑 home 粘滞 0——冒烟同型序列）。
        kv.prepare_prefill(
            session_id="s", target_instance_index=0, history_tokens=0,
            trigger_request_id="r0")
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=4,
            trigger_request_id="r0")
        kv.move_prefill_to_decode(
            session_id="s", target_instance_index=1,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=1, final_context_tokens=5,
            trigger_request_id="r0")
        kv.merge_back(session_id="s", trigger_request_id="r0", new_tokens=1)
        session = kv._sessions["s"]
        self.assertEqual(
            (session.instance_index, session.home_instance), (1, 0))
        # turn-1：copy 到 0（物理基础仍在 1）。
        kv.prepare_prefill(
            session_id="s", target_instance_index=0, history_tokens=5,
            trigger_request_id="r1", action="copy")
        self.assertEqual(session.base_instance_index, 1)
        return scheduler, kv

    def test_boundary_view_carries_physical_base_and_traversal_keeps_x0(self):
        from types import SimpleNamespace
        scheduler, _kv = self._shell_with_migrated_session()
        # D19 轮二（envelope 填充）：_joint_boundary_view 现随视图材料化
        # 首 token/义务表（_joint_boundary_stage_view 同一材料化）与
        # terminated（_joint_boundary_terminal 读 decode_length）——夹具
        # runtime 补齐这几项生产字段。
        runtime = SimpleNamespace(
            session_id="s", history_tokens_before=5, joint_input_tokens=1,
            joint_action="copy", request_id="r1",
            decode_length=8)
        boundary = scheduler._joint_boundary_view(
            runtime, SimpleNamespace(index=0))
        self.assertEqual(boundary.p_instance_index, 0)
        # 完整型单族：copy 的工作副本（基础前缀已随 prepare 物化到
        # i_P）= H_B 全量所在——边界源端 = 物理数据所在（== i_P），
        # 不取逻辑 home（0 恰同值纯属夹具几何；基础快照的物理所在
        # 仍解耦登记在 base_instance_index=1）。
        self.assertEqual(boundary.home_instance, 0)
        # 原地继续 x0 恒为候选（冒烟第 2 轮在此 fail-closed）；x0 记录
        # = 默认延续 stay@i_P（完整型单族），必须在场且 applicable。
        record = select_decode_target_and_action(
            mode="joint", cost_model=_model(_loads()), boundary=boundary)
        x0_records = [
            candidate for candidate in record.candidates
            if candidate.instance_index == 0
            and candidate.action == ACTION_STAY]
        self.assertEqual(len(x0_records), 1)
        self.assertTrue(x0_records[0].applicable)
        self.assertTrue(x0_records[0].is_continuation)


class DBoundaryCapacityAuditTest(unittest.TestCase):
    """D11/D17 修复轮二：D 边界决策行的容量审计字段（decode 行）。

    此前 capacity_summary_view/load_view 等审计块只随 P 决策行
    （joint_admission）落盘——D 边界时刻的容量判断仅凭日志不可重算。
    本轮给 decode 行补 ``capacity_summary_view``（与 P 行同一构造器
    ``_capacity_summary_audit_entry``，桶结构逐位同构）与
    ``capacity_ledger_by_tp_rank``（逐实例 C/W/O/Z 台账 + F 派生）。
    """

    def _decode_decision(self):
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "decode"]
        self.assertEqual(len(rows), 1)
        return scheduler, kv, rows[0]["decision"]

    def test_decode_row_carries_capacity_summary_isomorphic_to_p_row(self):
        from types import SimpleNamespace
        from online.sh30_online_scheduler import (
            _capacity_summary_audit_entry,
        )
        scheduler, kv, decision = self._decode_decision()
        self.assertIn("capacity_summary_view", decision)
        view = decision["capacity_summary_view"]
        self.assertEqual(
            [entry["instance_index"] for entry in view],
            [state.index for state in scheduler.instances])
        # P 行构造器同源重算：同函数 + 同快照（stay 夹具 drain 后无 KV
        # 变更，emit 时点快照 == 事后快照）→ 逐位同构（含 summary_id）。
        snapshot = kv.capacity_summary_snapshot()
        expected = [
            _capacity_summary_audit_entry(SimpleNamespace(
                instance_index=state.index,
                capacity_summary=snapshot.instance(state.index),
                reclaimable_bytes_by_tp_rank=(
                    kv._instance_reclaimable_capacity_by_tp_rank(
                        state.index))))
            for state in scheduler.instances]
        self.assertEqual(view, expected)
        # 冻结桶 schema：顶层键与逐桶键集与 P 行 capacity 块一致
        # （2026-09-30 修复轮扩三键：summary_version/maintenance_event_
        # count/error_sources——§5.5 版本与误差来源随审计块落盘）。
        for entry in view:
            self.assertEqual(
                set(entry),
                {"instance_index", "summary_id", "buckets",
                 "summary_version", "maintenance_event_count",
                 "error_sources"})
            self.assertTrue(entry["summary_id"].startswith("capsum-"))
            # §5.5：版本 = 管理器累计维护事件数（夹具决策过程已多次
            # 维护；与快照时点管理器同源同值）+ 误差来源登记在册。
            self.assertIsInstance(entry["summary_version"], int)
            self.assertGreaterEqual(entry["summary_version"], 0)
            self.assertIn("theta_share_continuous_approximation",
                          entry["error_sources"])
            self.assertIn("incremental_maintenance_lag",
                          entry["error_sources"])
            for bucket in entry["buckets"]:
                self.assertEqual(set(bucket), {
                    "category", "suffix_class", "shard_ratio",
                    "writeback_ports", "reclaimable_by_rank",
                    "clean_by_rank", "dirty_by_rank",
                    "atom_count_by_rank", "max_atom_by_rank"})

    def test_decode_row_carries_cwoz_ledger_with_free_derivation(self):
        scheduler, kv, decision = self._decode_decision()
        self.assertIn("capacity_ledger_by_tp_rank", decision)
        ledger_rows = decision["capacity_ledger_by_tp_rank"]
        self.assertEqual(
            [entry["instance_index"] for entry in ledger_rows],
            [state.index for state in scheduler.instances])
        tp = kv.tp_degree
        for entry in ledger_rows:
            self.assertEqual(
                set(entry),
                {"instance_index", "ledger_by_tp_rank", "free_by_tp_rank"})
            self.assertEqual(len(entry["ledger_by_tp_rank"]), tp)
            self.assertEqual(len(entry["free_by_tp_rank"]), tp)
            for (c, w, o, z), free in zip(
                    entry["ledger_by_tp_rank"], entry["free_by_tp_rank"]):
                # C/W/O/Z 分量分明（式 19）+ F = C−W−O−Z 派生一致。
                self.assertEqual(free, c - w - o - z)
                self.assertGreaterEqual(free, 0)
                self.assertGreaterEqual(o, 0)
                self.assertGreaterEqual(z, 0)


if __name__ == "__main__":
    unittest.main()
