#!/usr/bin/env python3
"""test_d19_boundary_layout.py -- D19 D 候选专用边界视图单元测试。

验收红线逐条落测（joint修改文档 D19 / v2.0 §5.3/§6.5/§13.1/§13.3/
§19.1 P→D 专项）：

  1. **第三目标 copy 补 H_B 全量** —— 完整型单族下 copy 计划 = i_P
     的 H_B 上下文区间（单来源齐全）；
  2. **首 token 只消费一次** —— 已发首 token 恰消费一次（二次
     消费 fail-closed）；
  3. **等待接口拆分** —— 原地接续不重排队（排队积压不进等待基、不再
     翻转 argmin）；新接纳不免费插队（目标全部在账负载照常计价）；
  4. **区间账本纪律** —— 片上位置上限、字节真值 = 账本向量、已承诺
     接收端排除（已在向 j 传输的内容不重复发送）、权威唯一性按
     token×layer 初等单元格裁决。

2026-10-14 远读退役同步：增量型布局族（P=remote-read）随远读一步
删除（D6）——InPlaceRemote/回 home copy 仅补 U/第三目标两源/增量矩形
用例删除，区间代数（层域互补/交叠 fail-closed）保留并换用
CONTEXT/BASE/GROWTH 三角色。

2026-10-15 recompute 退役同步：D recompute 静默重建族（rebuild 字段/
抑制语义）随 D8/D10 一步删除——DRecomputeSilentRebuildTest 整类退役。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d19_boundary_layout.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from dataclasses import replace as dataclass_replace

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
from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_BASE,
    INTERVAL_ROLE_CONTEXT,
    INTERVAL_ROLE_GROWTH,
    OBLIGATION_COMMITTED_TRANSFER,
    WAIT_IN_PLACE_CONTINUATION,
    WAIT_NEW_ADMISSION,
    BoundaryFirstTokenState,
    BoundaryKvInterval,
    BoundaryKVLayout,
    InheritedObligation,
    boundary_stage_work,
    boundary_target_wait_ns,
    boundary_wait_kind,
)
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostError,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
    project_end_state,
)
from joint.joint_scheduler import (  # noqa: E402
    select_decode_target_and_action,
)
from joint.test_d18_boundary_decision import (  # noqa: E402
    KV_TOKEN_BYTES,
    _loads,
    _complete_boundary,
    _model,
    _route2,
)


def _load(instance_index, *, queued=0, running=0, active=0):
    return InstanceLoadView(
        instance_index=instance_index,
        queued_task_load_ns=queued,
        running_task_load_ns=running,
        active_decode_task_load_ns=active,
        hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


# ------------------------------------------------------------ §19.1 专项 --


class ThirdTargetSingleSourceTest(unittest.TestCase):
    """完整型单族：第三目标 copy 单来源齐全（§6.5 状态之三）。"""

    def test_complete_family_third_target_single_source(self):
        """完整型第三目标：单来源（i_P 的 H_B 全量）——来源齐全 =
        恰好全部真实来源，不多不少。"""
        layout = BoundaryKVLayout.from_boundary(_complete_boundary())
        self.assertEqual(
            [(holder, interval.replica_id)
             for holder, interval in layout.copy_plan_for(1)],
            [(0, "boundary_context")])
        self.assertEqual(
            layout.copy_bytes_by_rank_for(1, 2),
            (KV_TOKEN_BYTES * 100,) * 2)


# 2026-10-15 recompute 退役：DRecomputeSilentRebuildTest 整类删除
# （静默重建族随 D8/D10 消亡；首 token 消费纪律由
# StageWorkContractTest 与 BoundaryStageViewShellTest 承接）。


# ------------------------------------------------------------ 等待接口 --


class WaitInterfaceSplitTest(unittest.TestCase):
    """等待接口拆分：原地接续不重排队；新接纳不免费插队。"""

    def test_wait_kind_single_criterion(self):
        self.assertEqual(
            boundary_wait_kind(3, 3), WAIT_IN_PLACE_CONTINUATION)
        self.assertEqual(boundary_wait_kind(4, 3), WAIT_NEW_ADMISSION)

    def test_in_place_wait_excludes_new_arrival_backlog(self):
        """原地接续等待基 = 在服流（running+active）；排队积压（新到达）
        不重压到已有请求头上。"""
        load = _load(0, queued=999_000, running=500, active=7_000)
        self.assertEqual(
            boundary_target_wait_ns(load, WAIT_IN_PLACE_CONTINUATION),
            7_500)

    def test_new_admission_pays_full_target_load(self):
        """新接纳等待基 = 目标全部在账负载（不免费 batch 接入）。"""
        load = _load(1, queued=999_000, running=500, active=7_000)
        self.assertEqual(
            boundary_target_wait_ns(load, WAIT_NEW_ADMISSION), 1_006_500)

    def test_unknown_wait_kind_fails_closed(self):
        with self.assertRaises(ValueError):
            boundary_target_wait_ns(_load(0), "queue_tail")

    def test_queued_backlog_no_longer_flips_argmin(self):
        """D19 前的失真形态钉测：i_P 只有排队积压时，原地接续不再被
        当重排队压价——argmin 保持原地（迁移收益不被排队积压制造）。"""
        loads = {
            0: _load(0, queued=10 ** 9),
            1: _load(1),
            2: _load(2),
        }
        model = JointCostModel(
            rates=JointHardwareRates.from_gbps(
                noc_link_gbps=10.0, pool_port_gbps=5.0,
                local_hbm_gbps=100.0, d2d_latency_ns=10,
                pool_latency_ns=100),
            loads=loads, flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2, route_fn=_route2)
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary())
        self.assertEqual(record.chosen.instance_index, 0)
        self.assertTrue(record.chosen.is_continuation)

    def test_switch_still_pays_new_target_occupancy(self):
        """新接纳侧对称钉测：目标实例的在账负载照常进入切换候选代价
        （异地不免费插队）——active 负载使切换严格贵于原地。"""
        loads = {
            0: _load(0),
            1: _load(1, active=10 ** 7),
            2: _load(2),
        }
        model = JointCostModel(
            rates=JointHardwareRates.from_gbps(
                noc_link_gbps=10.0, pool_port_gbps=5.0,
                local_hbm_gbps=100.0, d2d_latency_ns=10,
                pool_latency_ns=100),
            loads=loads, flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2, route_fn=_route2)
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary())
        self.assertEqual(record.chosen.instance_index, 0)
        busy = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1, action="copy")
        self.assertEqual(busy.breakdown.target_wait_ns, 10 ** 7)


# ------------------------------------------------------------ 区间纪律 --


class LayoutDisciplineTest(unittest.TestCase):
    """区间账本纪律（§5.3/§1）。"""

    def test_tp6_whole_head_shards_share_one_token_domain(self):
        bases = (98304, 98304, 81920, 81920, 81920, 81920)
        shards = tuple(41 * basis for basis in bases)
        boundary = _complete_boundary(
            history_tokens=41, kv_token_bytes=bases[0],
            kv_token_bytes_by_tp_rank=bases,
            working_bytes_by_tp_rank=shards)
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(layout.intervals[0].tokens, 41)
        self.assertEqual(layout.copy_bytes_by_rank_for(1, 6), shards)

    def test_on_chip_locations_bounded_by_three(self):
        self.assertEqual(
            BoundaryKVLayout.from_boundary(
                _complete_boundary()).on_chip_locations(), (0,))

    def test_committed_receiver_excluded_from_missing(self):
        """已在向 j 传输的区间不重复发送（§5.3）：义务表命中接收端后，
        该区间从其 copy 推导中排除。"""
        boundary = _complete_boundary()
        layout = dataclass_replace(
            BoundaryKVLayout.from_boundary(boundary),
            obligations=(InheritedObligation(
                obligation_id="r0#context_to_1",
                kind=OBLIGATION_COMMITTED_TRANSFER,
                receiver_instance=1,
                replica_id="boundary_context",
                summary="context in flight to third target"),))
        # H_B 上下文已在向实例 1 传输：其缺失集为空、copy 零腿。
        self.assertEqual(
            [interval.replica_id
             for interval in layout.missing_intervals_for(1)],
            [])
        self.assertEqual(layout.copy_bytes_by_rank_for(1, 2), (0, 0))
        # 其他目标不受该义务影响。
        self.assertEqual(
            [interval.replica_id
             for interval in layout.missing_intervals_for(2)],
            ["boundary_context"])
        self.assertEqual(
            layout.copy_bytes_by_rank_for(2, 2),
            (KV_TOKEN_BYTES * 100,) * 2)

    def test_from_boundary_requires_token_basis(self):
        boundary = _complete_boundary(kv_token_bytes=None)
        with self.assertRaises(ValueError):
            BoundaryKVLayout.from_boundary(boundary)

    def test_terminate_and_steps_carried(self):
        layout = BoundaryKVLayout.from_boundary(_complete_boundary())
        self.assertEqual(layout.decode_steps, 8)
        self.assertFalse(layout.terminated)


class StageWorkContractTest(unittest.TestCase):
    """阶段工作结构契约（fail-closed 面）。"""

    def test_invalid_stage_rejected(self):
        from joint.boundary_layout import StageWork
        with self.assertRaises(ValueError):
            StageWork(stage="X", prefill_tokens=0,
                      remaining_forward_steps=4)

    def test_consumed_token_requires_issue_and_advance(self):
        with self.assertRaises(ValueError):
            BoundaryFirstTokenState(
                issued=False, consumed=True, rng_advances=0)
        with self.assertRaises(ValueError):
            BoundaryFirstTokenState(
                issued=True, consumed=True, rng_advances=0)


# ------------------------------------------------------------ 执行侧 --


class BoundaryStageViewShellTest(unittest.TestCase):
    """drain 边界的执行侧材料化：stage_work/义务表落边界决策记录，
    等待接口类别随选中目标披露；switch 成员入目标就绪队列（不进新请求
    队尾、不跳过在飞列车）。"""

    def _scheduler_with_switch_success(self):
        """实例 0 压在服负载（active 成员）→ 边界 argmin 翻转到实例 1
        （容量充足、提交成功）的两实例真 drain 壳。"""
        from types import SimpleNamespace
        from online.test_decision_route_serialization import (
            _GraphStub,
            _decision_shell,
            _make_runtime,
        )
        from joint.test_d18_boundary_decision import _CompletionGraphStub

        hardware = _tiny_hardware()
        model = _tiny_model()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                    topology=topology)
        scheduler.graph = _CompletionGraphStub()
        scheduler.next_request = {}
        scheduler._runtime_index = {}
        scheduler.runtimes = [None]
        scheduler.completed_requests = 0
        scheduler.config = SimpleNamespace(request_queue=[
            SimpleNamespace(next_trigger_type="human",
                            inter_request_interval_ns=1000)])
        runtime = _make_runtime("r0", decode_length=8)
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
        # 在服负载压实例 0（原地接续等待基），实例 1 轻载 → 切换胜出。
        for seq in range(3):
            hog = _make_runtime(f"active0_{seq}")
            scheduler.runtime_by_request_id[f"active0_{seq}"] = hog
            scheduler.instances[0].active_decode.append(hog)
        return scheduler, kv, runtime

    def test_switch_decision_discloses_wait_kind_and_stage_view(self):
        scheduler, kv, runtime = self._scheduler_with_switch_success()
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertFalse(runtime.joint_decode_is_continuation)
        decision = runtime.boundary_decision
        # 等待接口类别 = 选中目标（新接纳）。
        self.assertEqual(
            decision["boundary_wait_kind"], WAIT_NEW_ADMISSION)
        # 阶段工作视图：零新 prefill、n̂_B 剩余前向、首 token 已采样待
        # 消费一次。
        stage_work = decision["stage_work"]
        self.assertEqual(stage_work["stage"], "D")
        self.assertEqual(stage_work["prefill_tokens"], 0)
        # W5 消费切换重锚（W6 §2）：n̂_B = 条件剩余估计器现值。
        self.assertEqual(
            stage_work["remaining_forward_steps"],
            scheduler._joint_length_history_or_default()
            .estimate_conditional(
                session_id=runtime.session_id,
                exposed_progress_steps=runtime.decode_tokens_consumed,
                category=getattr(
                    runtime, "joint_revealed_category", None),
            ).remaining_forward_steps)
        self.assertTrue(stage_work["first_token"]["issued"])
        self.assertFalse(stage_work["first_token"]["consumed"])
        self.assertEqual(stage_work["first_token"]["rng_advances"], 1)
        # 继承义务表：t_B 仍在册的预约如实登记（提交事务随后注销）。
        self.assertEqual(
            [item["kind"] for item in decision["inherited_obligations"]],
            ["capacity_reservation"])
        # 新接纳的执行面：成员进入目标实例就绪队列（KV 就绪栅栏 +
        # busy 门），不进新请求队尾 qp。
        self.assertIn(runtime, scheduler.instances[1].pending_decode_ready)
        self.assertNotIn(runtime, scheduler.instances[1].qp)
        self.assertNotIn(runtime, scheduler.instances[0].qp)
        # 决策行透传 D19 披露键（schema 前向兼容）。
        decode_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "decode"]
        self.assertEqual(len(decode_rows), 1)
        row_boundary = decode_rows[0]["decision"]["boundary_decision"]
        self.assertEqual(
            row_boundary["boundary_wait_kind"], WAIT_NEW_ADMISSION)
        self.assertIn("stage_work", row_boundary)
        self.assertIn("inherited_obligations", row_boundary)

    def test_in_place_decision_discloses_continuation_kind(self):
        from joint.test_d18_boundary_decision import _drain_ready_scheduler
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertTrue(runtime.joint_decode_is_continuation)
        self.assertEqual(
            runtime.boundary_decision["boundary_wait_kind"],
            WAIT_IN_PLACE_CONTINUATION)
        # 原地接续的执行面：进入本实例就绪队列（不重排队）。
        self.assertIn(runtime, scheduler.instances[0].pending_decode_ready)
        self.assertNotIn(runtime, scheduler.instances[0].qp)


# ---------------------------------------- 权威唯一性（token×layer 单元格） --


class IntervalAuthorityUniquenessTest(unittest.TestCase):
    """区间权威唯一性按 token×layer 初等单元格裁决（project_end_state
    代数，角色无关；远读退役前的增量/基础形态用例收缩为通用区间形态）。"""

    def test_partial_suffix_and_base_share_tokens_disjoint_layers(self):
        """层域互补形态：后缀 S 与基础 [0,b) token 重合、层域互补
        ——合法布局（初等单元格不相交），不得按纯 token 交叠误杀。"""
        suffix_tokens, prefix_layers, model_layers = 100, 16, 32
        suffix = (KV_TOKEN_BYTES * suffix_tokens // 2,) * 2
        growth = (KV_TOKEN_BYTES * 40,) * 2
        base = (KV_TOKEN_BYTES * suffix_tokens // 2,) * 2
        intervals = (
            BoundaryKvInterval(
                "work_suffix", INTERVAL_ROLE_GROWTH, 0, True,
                0, suffix_tokens, prefix_layers, model_layers, suffix),
            BoundaryKvInterval(
                "work_growth", INTERVAL_ROLE_GROWTH, 0, True,
                suffix_tokens, suffix_tokens + 40, 0, model_layers,
                growth),
            BoundaryKvInterval(
                "base", INTERVAL_ROLE_BASE, 2, True,
                0, suffix_tokens, 0, prefix_layers, base),
        )
        state = project_end_state(
            intervals=intervals, decode_instance=1, logic_home=2)
        self.assertEqual(
            state.total_bytes, sum(suffix) + sum(growth) + sum(base))

    def test_same_cell_overlap_still_fails_closed(self):
        """真实交叠（token 且层域同时相交；退化层域 (0,0) = 全层）仍
        fail-closed——层域互补放行只针对合法形态。"""
        with self.assertRaises(JointCostError):
            project_end_state(
                intervals=(
                    BoundaryKvInterval(
                        "a", INTERVAL_ROLE_CONTEXT, 0, True,
                        0, 100, 4, 8, (KV_TOKEN_BYTES * 100,) * 2),
                    BoundaryKvInterval(
                        "b", INTERVAL_ROLE_CONTEXT, 1, True,
                        50, 150, 0, 8, (KV_TOKEN_BYTES * 100,) * 2),
                ),
                decode_instance=1, logic_home=0)
        # 退化层域（未披露 = 全层口径）与真实层域 token 交叠 → 仍违例。
        with self.assertRaises(JointCostError):
            project_end_state(
                intervals=(
                    BoundaryKvInterval(
                        "a", INTERVAL_ROLE_CONTEXT, 0, True,
                        0, 100, 0, 0, (KV_TOKEN_BYTES * 100,) * 2),
                    BoundaryKvInterval(
                        "b", INTERVAL_ROLE_CONTEXT, 1, True,
                        50, 150, 0, 8, (KV_TOKEN_BYTES * 100,) * 2),
                ),
                decode_instance=1, logic_home=0)


if __name__ == "__main__":
    unittest.main()
