#!/usr/bin/env python3
"""test_d19_boundary_layout.py -- D19 D 候选专用边界视图单元测试。

验收红线逐条落测（joint修改文档 D19 / v2.0 §5.3/§6.5/§13.1/§13.3/
§19.1 P→D 专项）：

  1. **原地 remote 不变成本地 hit** —— 增量型边界上 (i_P, remote-read)
     的远读基数 = 源端（home）仍保留的基础区间（式 18a），非零、不因
     "目标已有部分区间" 被算成本地命中；
  2. **回 home copy 仅补 U** —— copy 计划只含 i_P 持有的增量区间，
     基础区间（home 已有）不重复搬运；
  3. **第三目标 copy 补 H+U** —— copy 计划逐源齐全（home→H、
     i_P→U），字节 = 两者并集；
  4. **remote 反复读两个真实来源** —— 第三目标的远读区间 = 两个真实
     来源（H@home + U@i_P），逐遍基数 = 两者并集；
  5. **D recompute 静默重建且首 token 只消费一次** —— 阶段工作视图
     rebuild = H_B、抑制输出/采样/RNG；已发首 token 恰消费一次（二次
     消费 fail-closed）；
  6. **等待接口拆分** —— 原地接续不重排队（排队积压不进等待基、不再
     翻转 argmin）；新接纳不免费插队（目标全部在账负载照常计价）；
  7. **区间账本纪律** —— 片上位置 ≤3、字节真值 = 账本向量、已承诺
     接收端排除（已在向 j 传输的内容不重复发送）。

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
    INTERVAL_ROLE_INCREMENT,
    OBLIGATION_COMMITTED_TRANSFER,
    WAIT_IN_PLACE_CONTINUATION,
    WAIT_NEW_ADMISSION,
    BoundaryFirstTokenState,
    BoundaryKvInterval,
    BoundaryKVLayout,
    InheritedObligation,
    boundary_recompute_stage_work,
    boundary_stage_work,
    boundary_target_wait_ns,
    boundary_wait_kind,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
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
    _loads3,
    _increment_boundary,
    _complete_boundary,
    _model,
    _route2,
)

BASE_BYTES = KV_TOKEN_BYTES * 100      # 增量型基础区间（100 token）/rank
WORK_BYTES = KV_TOKEN_BYTES * 40       # 增量型 U 区间（40 token）/rank


def _load(instance_index, *, queued=0, running=0, active=0):
    return InstanceLoadView(
        instance_index=instance_index,
        queued_task_load_ns=queued,
        running_task_load_ns=running,
        active_decode_task_load_ns=active,
        hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


# ------------------------------------------------------------ §19.1 专项 --


class InPlaceRemoteNotLocalHitTest(unittest.TestCase):
    """原地 remote 不变成本地 hit（§19.1 P→D 专项第 1 条）。"""

    def test_in_place_remote_reads_retained_base(self):
        """(i_P, remote-read) 的远读基数 = 源端仍保留的基础区间：布局
        推导与计价腿双双非零——i_P 已持有 U 不使基础区间被当本地。"""
        boundary = _increment_boundary()
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(
            layout.remote_bytes_by_rank_for(0, 2),
            (BASE_BYTES, BASE_BYTES))
        self.assertEqual(
            [i.replica_id for i in layout.remote_intervals_for(0)],
            ["boundary_base"])
        model = _model(_loads3())
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_REMOTE,
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertTrue(candidate.is_continuation)
        # 非本地 hit：远读腿真实存在（式 18a 基数 × n̂_B 遍）。
        self.assertGreater(candidate.breakdown.remote_read_ns, 0)
        self.assertEqual(candidate.breakdown.history_prep_ns, 0)
        self.assertIn("boundary_wait_kind=in_place_continuation",
                      candidate.breakdown.notes)

    def test_in_place_remote_base_identity_with_view(self):
        """布局远读基数与 D18 计价视图的 base 向量逐 rank 恒等（字节
        真值 = 账本向量，区间只补身份）。"""
        boundary = _increment_boundary()
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(
            layout.remote_bytes_by_rank_for(
                boundary.p_instance_index, 2),
            tuple(int(v) for v in boundary.base_bytes_by_tp_rank))


class HomeCopyFillsOnlyUTest(unittest.TestCase):
    """回 home copy 仅补 U（§6.5 状态之二）。"""

    def test_home_copy_plan_is_increment_only(self):
        boundary = _increment_boundary()
        layout = BoundaryKVLayout.from_boundary(boundary)
        plan = layout.copy_plan_for(boundary.home_instance)
        # 只补 i_P 持有的增量区间；home 已有的基础区间不重复搬运。
        self.assertEqual(
            [(holder, interval.replica_id, interval.role)
             for holder, interval in plan],
            [(0, "boundary_increment", INTERVAL_ROLE_INCREMENT)])
        self.assertEqual(
            layout.copy_bytes_by_rank_for(boundary.home_instance, 2),
            (WORK_BYTES, WORK_BYTES))

    def test_home_move_leg_is_u_only_not_full_context(self):
        """回 home/切换的迁移腿按缺失区间（copy 族）——与完整型 H_B
        迁移腿对照（copy 只补目标缺少部分的计价纪律）：
        * 完整型 copy@j = H_B 全量（100 token × 128 B × 2 rank）；
        * 增量型 copy@home = 仅 U（增量区间；home 已有基础不重复搬运）；
        * 增量型 copy@第三目标 = H+U 并集（两腿逐区间）。
        增量型 remote（C4 纯多源）零迁移腿——U 留 i_P 逐遍远读。"""
        model = _model(_loads3())
        complete_move = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1, action="copy",
            remote_enabled=True)
        self.assertIn("boundary_move_bytes=25600",
                      complete_move.breakdown.notes)
        home_copy = model.estimate_D_remaining(
            boundary=_increment_boundary(),
            instance_index=_increment_boundary().home_instance,
            action="copy", remote_enabled=True)
        self.assertIn("boundary_move_bytes=10240",
                      home_copy.breakdown.notes)
        self.assertIn("boundary_copy_legs=1", home_copy.breakdown.notes)
        third_copy = model.estimate_D_remaining(
            boundary=_increment_boundary(), instance_index=1,
            action="copy", remote_enabled=True)
        self.assertIn("boundary_move_bytes=35840",
                      third_copy.breakdown.notes)
        self.assertIn("boundary_copy_legs=2",
                      third_copy.breakdown.notes)
        self.assertIn("boundary_copy_sources=2,0",
                      third_copy.breakdown.notes)
        increment_remote = model.estimate_D_remaining(
            boundary=_increment_boundary(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(
            increment_remote.breakdown.history_prep_ns, 0)
        self.assertLess(
            home_copy.breakdown.history_prep_ns,
            complete_move.breakdown.history_prep_ns)


class ThirdTargetTwoSourceTest(unittest.TestCase):
    """第三目标来源齐全（§6.5 状态之三/四）。"""

    def test_third_target_copy_covers_h_and_u(self):
        boundary = _increment_boundary()
        layout = BoundaryKVLayout.from_boundary(boundary)
        plan = layout.copy_plan_for(1)
        self.assertEqual(
            [(holder, interval.replica_id, interval.role)
             for holder, interval in plan],
            [(2, "boundary_base", INTERVAL_ROLE_BASE),
             (0, "boundary_increment", INTERVAL_ROLE_INCREMENT)])
        self.assertEqual(
            layout.copy_bytes_by_rank_for(1, 2),
            (BASE_BYTES + WORK_BYTES, BASE_BYTES + WORK_BYTES))

    def test_third_target_remote_reads_both_real_sources(self):
        boundary = _increment_boundary()
        layout = BoundaryKVLayout.from_boundary(boundary)
        holders = sorted({
            interval.holder_instance
            for interval in layout.remote_intervals_for(1)})
        self.assertEqual(holders, [0, 2])   # i_P（U）+ home（H）两真实来源
        self.assertEqual(
            layout.remote_bytes_by_rank_for(1, 2),
            (BASE_BYTES + WORK_BYTES,) * 2)
        # 多来源共用同一对有限缓冲的计价口径 = 逐遍一次读取（来源边界
        # 排空），遍数由计价侧乘——布局只给单遍基数。

    def test_complete_family_third_target_single_source(self):
        """完整型第三目标：单来源（i_P 的 H_B 全量）——与增量型两源
        形态区分（来源齐全 = 恰好全部真实来源，不多不少）。"""
        layout = BoundaryKVLayout.from_boundary(_complete_boundary())
        self.assertEqual(
            [(holder, interval.replica_id)
             for holder, interval in layout.copy_plan_for(1)],
            [(0, "boundary_context")])
        self.assertEqual(
            layout.copy_bytes_by_rank_for(1, 2),
            (KV_TOKEN_BYTES * 100,) * 2)


class DRecomputeSilentRebuildTest(unittest.TestCase):
    """D recompute 静默重建且首 token 只消费一次（§19.1 P→D 专项）。"""

    def test_recompute_stage_work_is_silent_rebuild(self):
        first = BoundaryFirstTokenState(
            issued=True, consumed=False, rng_advances=1)
        work = boundary_recompute_stage_work(
            history_tokens=140, decode_steps=8, first_token=first)
        self.assertEqual(work.stage, "D")
        self.assertEqual(work.rebuild_tokens, 140)      # H_B 全量重建
        self.assertTrue(work.rebuild_silent)
        self.assertEqual(work.prefill_tokens, 0)        # 无新 U-prefill
        self.assertEqual(work.remaining_forward_steps, 8)  # n̂_B 归一
        self.assertTrue(work.first_token.rebuild_sampling_suppressed)

    def test_normal_first_step_consumes_issued_token_once(self):
        first = BoundaryFirstTokenState(
            issued=True, consumed=False, rng_advances=1)
        self.assertEqual(first.rng_advances, 1)         # P 末 chunk 采样恰一次
        consumed = first.consume_once()
        self.assertTrue(consumed.consumed)
        with self.assertRaises(ValueError):
            first.consume_once().consume_once()

    def test_silent_rebuild_never_re_samples(self):
        """静默重建的采样/RNG 契约：抑制标志与重建共存，且不产生第二
        次 RNG 推进（rng_advances 只能 0/1，构造层 fail-closed）。"""
        first = BoundaryFirstTokenState(
            issued=True, consumed=False, rng_advances=1)
        work = boundary_recompute_stage_work(
            history_tokens=100, decode_steps=4, first_token=first)
        self.assertEqual(work.first_token.rng_advances, 1)
        with self.assertRaises(ValueError):
            BoundaryFirstTokenState(
                issued=True, consumed=False, rng_advances=2)
        # 消费过一次后 rng_advances 恒 1：重建 + 正常第一步 = 恰一次采样。
        self.assertEqual(first.consume_once().rng_advances, 1)

    def test_boundary_default_work_has_no_new_prefill(self):
        work = boundary_stage_work(
            decode_steps=8,
            first_token=BoundaryFirstTokenState(
                issued=True, consumed=False, rng_advances=1))
        self.assertEqual(work.prefill_tokens, 0)
        self.assertEqual(work.rebuild_tokens, 0)
        self.assertFalse(work.rebuild_silent)


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
            model_layers=4, instance_tp_size=2, route_fn=_route2,
            remote_read_kv_token_bytes=KV_TOKEN_BYTES)
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary(), remote_enabled=True)
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
            model_layers=4, instance_tp_size=2, route_fn=_route2,
            remote_read_kv_token_bytes=KV_TOKEN_BYTES)
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary(), remote_enabled=True)
        self.assertEqual(record.chosen.instance_index, 0)
        busy = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1, action="copy",
            remote_enabled=True)
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
            working_bytes_by_tp_rank=shards,
            base_bytes_by_tp_rank=(0,) * 6)
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(layout.intervals[0].tokens, 41)
        self.assertEqual(layout.copy_bytes_by_rank_for(1, 6), shards)

    def test_partial_working_rectangles_keep_suffix_and_increment_domains(self):
        bases = (98304, 98304, 81920, 81920, 81920, 81920)
        suffix = tuple(100 * basis // 2 for basis in bases)
        increment = tuple(40 * basis for basis in bases)
        intervals = (
            BoundaryKvInterval(
                "suffix", INTERVAL_ROLE_INCREMENT, 0, True,
                0, 100, 16, 32, suffix),
            BoundaryKvInterval(
                "increment", INTERVAL_ROLE_INCREMENT, 0, True,
                100, 140, 0, 32, increment),
        )
        boundary = _increment_boundary(
            history_tokens=140, kv_token_bytes=bases[0],
            kv_token_bytes_by_tp_rank=bases,
            working_bytes_by_tp_rank=tuple(
                s + i for s, i in zip(suffix, increment)),
            base_bytes_by_tp_rank=suffix,
            working_intervals=intervals)
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(layout.intervals[:2], intervals)
        self.assertEqual(
            layout.copy_bytes_by_rank_for(1, 6),
            tuple(140 * basis for basis in bases))
        self.assertEqual(
            layout.copy_bytes_by_rank_for(2, 6),
            boundary.working_bytes_by_tp_rank)

    def test_on_chip_locations_bounded_by_three(self):
        self.assertEqual(
            BoundaryKVLayout.from_boundary(
                _complete_boundary()).on_chip_locations(), (0,))
        self.assertEqual(
            BoundaryKVLayout.from_boundary(
                _increment_boundary()).on_chip_locations(), (0, 2))

    def test_committed_receiver_excluded_from_missing(self):
        """已在向 j 传输的区间不重复发送（§5.3）：义务表命中接收端后，
        该区间从其 copy/remote 推导中排除。"""
        boundary = _increment_boundary()
        layout = dataclass_replace(
            BoundaryKVLayout.from_boundary(boundary),
            obligations=(InheritedObligation(
                obligation_id="r0#base_to_1",
                kind=OBLIGATION_COMMITTED_TRANSFER,
                receiver_instance=1,
                replica_id="boundary_base",
                summary="base in flight to third target"),))
        # 基础区间已在向实例 1 传输：其缺失集只剩 U。
        self.assertEqual(
            [interval.replica_id
             for interval in layout.missing_intervals_for(1)],
            ["boundary_increment"])
        self.assertEqual(
            layout.copy_bytes_by_rank_for(1, 2), (WORK_BYTES,) * 2)
        # 其他目标（home）不受该义务影响。
        self.assertEqual(
            [interval.replica_id
             for interval in layout.missing_intervals_for(2)],
            ["boundary_increment"])

    def test_zero_byte_working_increment_layout_has_no_interval(self):
        boundary = _increment_boundary(working_bytes_by_tp_rank=(0, 0))
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(
            [i.replica_id for i in layout.intervals], ["boundary_base"])

    def test_from_boundary_requires_token_basis(self):
        boundary = _increment_boundary(kv_token_bytes=None)
        with self.assertRaises(ValueError):
            BoundaryKVLayout.from_boundary(boundary)

    def test_terminate_and_steps_carried(self):
        layout = BoundaryKVLayout.from_boundary(_complete_boundary())
        self.assertEqual(layout.decode_steps, 8)
        self.assertFalse(layout.terminated)


class StageWorkContractTest(unittest.TestCase):
    """阶段工作结构契约（fail-closed 面）。"""

    def test_rebuild_requires_silent_flag(self):
        from joint.boundary_layout import StageWork
        with self.assertRaises(ValueError):
            StageWork(stage="D", prefill_tokens=0, rebuild_tokens=100,
                      rebuild_silent=False, remaining_forward_steps=4)
        with self.assertRaises(ValueError):
            StageWork(stage="D", prefill_tokens=0, rebuild_tokens=0,
                      rebuild_silent=True, remaining_forward_steps=4)

    def test_invalid_stage_rejected(self):
        from joint.boundary_layout import StageWork
        with self.assertRaises(ValueError):
            StageWork(stage="X", prefill_tokens=0, rebuild_tokens=0,
                      rebuild_silent=False, remaining_forward_steps=4)

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


# ---------------------------------------- 增量-完整形态共存修复（二期第 1 项） --


class IncrementCompleteFormCoexistenceTest(unittest.TestCase):
    """turn-1 prompt 追加时增量/完整两形态不得并存为交叠权威区间。

    回归锚（2026-10-03 rollout_joint 冒烟根因，session_1@queue_1s）：
    turn-0 增量形态（base [0,11087) 权威 @home）→ turn-1 追加 768 后，
    LOCAL 基（全层驻留）remote-read 的工作副本只含本轮增量——若边界
    视图把仅含 U 的工作副本伪造成 (0, 11855) 完整上下文矩形，from_
    boundary 再追加 boundary_base (0, 11087)，两权威区间 token 交叠，
    project_end_state 以 ``JointCostError`` fail-closed（两后端同崩）。
    修复 = 生产边界按增量矩形披露（见 test_production_boundary_view_
    uses_increment_rectangle）+ 权威唯一性按 token×layer 初等单元格裁决。
    """

    BASE_TOKENS = 100
    INPUT_TOKENS = 40
    HISTORY_TOKENS = BASE_TOKENS + INPUT_TOKENS

    def _bytes_for(self, tokens):
        return (KV_TOKEN_BYTES * tokens,) * 2

    def test_turn1_append_local_base_projects_end_state(self):
        """最小复现：turn-0 增量 boundary → turn-1 追加 → 投影不得抛。"""
        boundary = _increment_boundary(
            history_tokens=self.HISTORY_TOKENS,
            working_bytes_by_tp_rank=self._bytes_for(self.INPUT_TOKENS),
            base_bytes_by_tp_rank=self._bytes_for(self.BASE_TOKENS),
            base_tokens=self.BASE_TOKENS)
        boundary = dataclass_replace(
            boundary,
            working_intervals=(BoundaryKvInterval(
                replica_id="boundary_work_increment",
                role=INTERVAL_ROLE_INCREMENT,
                holder_instance=boundary.p_instance_index,
                authority=True,
                token_start=self.BASE_TOKENS,
                token_end=self.HISTORY_TOKENS,
                layer_start=0,
                layer_end=32,
                bytes_by_tp_rank=self._bytes_for(self.INPUT_TOKENS)),))
        layout = BoundaryKVLayout.from_boundary(boundary)
        roles = tuple(
            (interval.replica_id, interval.token_start, interval.token_end)
            for interval in layout.intervals)
        # 增量 [H, H+U) 与基础 [0, H) token 域分割——无交叠双权威。
        self.assertEqual(
            roles,
            (("boundary_work_increment",
              self.BASE_TOKENS, self.HISTORY_TOKENS),
             ("boundary_base", 0, self.BASE_TOKENS)))
        state = project_end_state(
            intervals=layout.intervals,
            decode_instance=1,
            decode_growth_by_rank=self._bytes_for(8),
            decode_growth_tokens=8,
            history_tokens=self.HISTORY_TOKENS,
            logic_home=2)
        self.assertEqual(
            state.total_bytes,
            sum(self._bytes_for(
                self.HISTORY_TOKENS + 8)))

    def test_partial_suffix_and_base_share_tokens_disjoint_layers(self):
        """PARTIAL 混合形态：后缀 S 与基础 [0,b) token 重合、层域互补
        ——合法布局（初等单元格不相交），不得按纯 token 交叠误杀。"""
        suffix_tokens, prefix_layers, model_layers = 100, 16, 32
        suffix = (KV_TOKEN_BYTES * suffix_tokens // 2,) * 2
        increment = (KV_TOKEN_BYTES * 40,) * 2
        base = (KV_TOKEN_BYTES * suffix_tokens // 2,) * 2
        intervals = (
            BoundaryKvInterval(
                "work_suffix", INTERVAL_ROLE_INCREMENT, 0, True,
                0, suffix_tokens, prefix_layers, model_layers, suffix),
            BoundaryKvInterval(
                "work_increment", INTERVAL_ROLE_INCREMENT, 0, True,
                suffix_tokens, suffix_tokens + 40, 0, model_layers,
                increment),
            BoundaryKvInterval(
                "base", INTERVAL_ROLE_BASE, 2, True,
                0, suffix_tokens, 0, prefix_layers, base),
        )
        state = project_end_state(
            intervals=intervals, decode_instance=1, logic_home=2)
        self.assertEqual(
            state.total_bytes, sum(suffix) + sum(increment) + sum(base))

    def test_same_cell_overlap_still_fails_closed(self):
        """真实交叠（token 且层域同时相交；退化层域 (0,0) = 全层）仍
        fail-closed——修复只放行层域互补的合法形态。"""
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

    def test_production_boundary_view_uses_increment_rectangle(self):
        """生产侧（sh30 _joint_boundary_view）：LOCAL 基 remote-read 的
        turn-1 边界披露**增量矩形**，不再伪造完整上下文矩形。"""
        from types import SimpleNamespace
        from online.test_decision_route_serialization import _decision_shell
        hardware = _tiny_hardware()
        model = _tiny_model()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                    topology=topology)
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
        # turn-1：remote-read@0（LOCAL 全层基驻留 1），P 已物化增量。
        kv.prepare_prefill(
            session_id="s", target_instance_index=0, history_tokens=5,
            trigger_request_id="r1", action="remote-read")
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=1,
            trigger_request_id="r1")
        runtime = SimpleNamespace(
            session_id="s", history_tokens_before=5, joint_input_tokens=1,
            joint_action="remote-read", request_id="r1",
            decode_length=8, remote_read_preplan=None)
        envelope = scheduler._joint_boundary_view(
            runtime, SimpleNamespace(index=0))
        rectangles = tuple(
            (interval.replica_id, interval.role,
             interval.token_start, interval.token_end)
            for interval in envelope.working_intervals)
        self.assertEqual(
            rectangles,
            (("boundary_work_increment", INTERVAL_ROLE_INCREMENT, 5, 6),))
        layout = BoundaryKVLayout.from_boundary(envelope)
        self.assertEqual(
            tuple(
                (interval.replica_id, interval.token_start, interval.token_end)
                for interval in layout.intervals),
            (("boundary_work_increment", 5, 6), ("boundary_base", 0, 5)))
        state = project_end_state(
            intervals=layout.intervals, decode_instance=1,
            decode_growth_by_rank=(KV_TOKEN_BYTES * 8,) * 2,
            decode_growth_tokens=8, history_tokens=6, logic_home=0)
        self.assertEqual(
            state.total_bytes, sum((KV_TOKEN_BYTES * 14,) * 2))


if __name__ == "__main__":
    unittest.main()
