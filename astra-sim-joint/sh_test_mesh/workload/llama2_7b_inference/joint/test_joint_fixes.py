#!/usr/bin/env python3
"""test_joint_fixes.py -- R1'-R15 修复批的关键语义单元测试（2026-09-14）。

覆盖（对应 kimi 终稿 §5 用例的核心可单测断言）：
  - R1'：动作感知预约足迹单源 + PARTIAL 去钉扎（可行性与物理可行性
    不再被驻留亲和掩码）。（remote-read input-only 足迹与 PARTIAL 远读
  适用性/预约用例已随远读退役删除，2026-10-14。）
  - R13/N9：recompute@home 缺失后缀物化、无工作副本（stay 同构本地
    事务）、merge 零搬运；跨实例 recompute 工作副本语义不变。
  - R4：merge home 容量不足 → 基础前缀自降级（merge_degrade_events
    落账）→ k=0 落 REMOTE 归并（永不失败）；重复 merge 版本键。
  - N3'：容量类异常类型化（KVCapacityError 携带已提交逐出）；合同类
    异常不被类型化吞掉。
  - D18：P→D 边界预约移动事务（同址 no-op/跨实例换址/容量失败保留继续态）。
  - R14（调度器侧）：停滞登记/唤醒键/死锁守卫（__new__ 壳夹具）。
  - R3'：驱逐等待定价区分逐出可解/深缺口（活跃剩余负载峰值）。
  - R15-2/P3：ServiceFactors α 时间衰减公式 + 同时刻汇总 + 拒绝样本。
  - R15-1/F-B：divisor_multi 按全部 TP 并行流链路并集取瓶颈。
  - R12：水印重放的 joint 双驻留原语（stash_base / merge 结算）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_fixes.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    KVCapacityError,
    KVCacheManager,
    kv_cache_shard_bytes_for_tokens,
    kv_cache_shard_bytes_for_layer_range,
    model_weight_shard_bytes_by_tp_rank,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _seed,
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
)
from joint.capacity_summary import (  # noqa: E402
    PHASE_DECODE,
    PhaseFootprint,
)


# ===================================================== R1'/N1/N3'/R4/R13 ==

def _manager(capacity_bytes=1_000_000_000, **kwargs):
    hardware = _tiny_hardware(capacity_bytes)
    topology = _two_instance_topology(hardware)
    return KVCacheManager(
        topology, _tiny_model(4),
        category_mode="typed", layer_policy="minimal_layer_groups",
        **kwargs)


def _make_partial(kv, session_id="s", tokens=10, completion_ns=10):
    """在实例 0 建立 PARTIAL 会话（half 后缀逐出）。"""
    _seed(kv, session_id, 0, tokens, completion_ns, "human")
    kv._evict_suffix(
        kv._sessions[session_id],
        phase="history", reason="test_partial",
        trigger_request_id="seed")
    return kv._sessions[session_id]


class FootprintSingleSourceTest(unittest.TestCase):
    """R1'：预约基数单源（history+input 恒等式已随远读退役内联到
    调用点）+ 去钉扎。"""

    def test_partial_depinning_allows_other_instances(self):
        """PARTIAL 会话不再构成实例亲和掩码：容量允许时其他实例可行。"""
        kv = _manager()
        _make_partial(kv)
        feasible = kv.request_hbm_feasible_instances(
            session_id="s", final_context_tokens=20)
        self.assertEqual(feasible, (True, True))
        eventual = kv.request_hbm_eventually_feasible_instances(
            session_id="s", final_context_tokens=20)
        self.assertEqual(eventual, (True, True))


class RecomputeAtHomeSemanticsTest(unittest.TestCase):
    """R13/N9 本地事务骨架 + D10 空间口径（v2.0 §6.2，2026-09-26）：
    recompute@home = 驻留字节不清零、不建工作副本；缺失后缀 prepare
    入账 = 重建上下文的空间占位（**空间复用，非计算口径**——完整 H+U
    重放不依赖旧 KV，"只重算缺失层"折算已取消，重算量由调度器折入
    span：整份 H + input、span 基 0）。"""

    def test_recompute_at_partial_home_materializes_suffix_only(self):
        kv = _manager()
        session = _make_partial(kv, tokens=10)
        prefix_before = session.resident_prefix_layers
        before = kv.session_snapshot("s")
        _before, transfers, evictions = kv.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=10, trigger_request_id="t1",
            action="recompute")
        # 无传输（重算物化 = 计算量，非搬运）；无工作副本（stay 同构）。
        self.assertEqual(transfers, ())
        after = kv.session_snapshot("s")
        self.assertIsNone(after.working_kind)
        self.assertEqual(after.location, kv.LOCAL_HBM)
        self.assertEqual(after.resident_prefix_layers, 4)  # 全层驻留
        # 物化字节 = 缺失后缀层（prefix_before..L），不多算前缀。
        suffix = kv_cache_shard_bytes_for_tokens(
            kv.model, 10, kv.tp_degree)
        suffix_only = suffix[0] * (4 - prefix_before) // 4
        used0 = kv.hbm_snapshots(0)[0].kv_cache_bytes
        used0_before = before.local_bytes // kv.tp_degree
        self.assertAlmostEqual(
            used0 - used0_before, suffix_only, delta=2)
        # merge：stay 等价零流量（无工作副本）。
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=14,
            trigger_request_id="t1")
        self.assertEqual(
            kv.merge_back(session_id="s", trigger_request_id="t1",
                          new_tokens=4), ())
        kv.mark_complete("s", 20, "human")

    def test_cross_instance_recompute_keeps_working_copy(self):
        kv = _manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1",
            action="recompute")
        working = kv.session_snapshot("s")
        self.assertEqual(working.working_kind, "recompute")
        self.assertEqual(working.context_tokens, 0)
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=13,
            trigger_request_id="t1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=3)
        merged = kv.session_snapshot("s")
        # merge v2（2026-09-17 需求②）：recompute 执行端恒持并集 →
        # 反向零字节翻转——零传输、终态 LOCAL@exec、home 迁移 exec。
        self.assertEqual(transfers, ())
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(merged.location, kv.LOCAL_HBM)
        self.assertEqual(merged.context_tokens, 13)
        # K8（2026-09-23）：原随附 all(kind != "local_hit") 对上方已钉
        # 空元组恒真——删除（空集断言由 assertEqual(transfers, ()) 承载）。


class MergeV2CapacityFallbackTest(unittest.TestCase):
    """merge v2（2026-09-17 裁定④）：R4 自降级/k=0 池归并兜底退役——
    copy/recompute 反向零字节翻转无需任何空间准备（并集已在胜者侧），
    home 灌满活跃占用也照常成功；空间准备腿仅远读两方向存在，已随
    remote-read 退役删除（2026-10-14；改造前双向二选一兜底＋双侧深缺口
    fail-closed 的金样留档见 git 9a95e06）。"""

    def test_copy_flip_needs_no_home_capacity(self):
        # home 灌满活跃占用（不可逐）——copy 零字节翻转照常成功（无空间
        # 准备、无自降级事件——台账冻结恒空）。
        kv = _manager(capacity_bytes=51_328)
        _seed(kv, "s", 0, 10, 10, "human")
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True  # 不可逐的活跃占用
        kv._refresh_capacity_summary(("hog",))  # D11 白盒重索引
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=14,
            trigger_request_id="t1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=4)
        self.assertEqual(transfers, ())
        self.assertEqual(kv.merge_degrade_events, [])
        merged = kv.session_snapshot("s")
        # 零字节翻转终态：全层 LOCAL@exec、home 迁移、home 侧基础释放。
        self.assertEqual(merged.location, kv.LOCAL_HBM)
        self.assertEqual(merged.instance_index, 1)
        self.assertEqual(merged.home_instance, 1)
        self.assertEqual(merged.resident_prefix_layers, 4)
        kv.mark_complete("s", 20, "human")

    def test_duplicate_merge_fails_closed(self):
        kv = _manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=13,
            trigger_request_id="t1")
        kv.merge_back(session_id="s", trigger_request_id="t1", new_tokens=3)
        with self.assertRaises(RuntimeError):
            kv.merge_back(
                session_id="s", trigger_request_id="t1", new_tokens=3)


class TypedCapacityErrorsTest(unittest.TestCase):
    """N3'：容量类类型化；合同类异常原样上抛。"""

    def test_deep_gap_raises_typed_capacity_error(self):
        kv = _manager(capacity_bytes=51_328)
        _seed(kv, "s", 0, 10, 10, "human")
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True
        kv._refresh_capacity_summary(("hog",))  # D11 白盒重索引
        with self.assertRaises(KVCapacityError):
            kv.reserve_request_capacity(
                request_id="r9", session_id="s",
                instance_index=0, final_context_tokens=90)

    def test_contract_duplicate_reservation_not_swallowed(self):
        kv = _manager()
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=10)
        with self.assertRaises(ValueError) as ctx:
            kv.reserve_request_capacity(
                request_id="r1", session_id="s1", instance_index=0,
                final_context_tokens=10)
        self.assertNotIsInstance(ctx.exception, KVCapacityError)

    def test_d18_boundary_reservation_move_transaction(self):
        """D18（C15 重设计，2026-09-26）→ D20 事务面（2026-09-26）：
        P→D 边界预约移动从"同实例 no-op + 跨实例 raise"的 M3 钉死形态，
        经 D18 重设计为真实阶段事务，再由 D20 升级为幂等 epoch 的跨阶段
        预约事务（``commit_stage_reservation_transaction``，v2.0
        §14.5/§16.4）——两段式 move/revert 调用口已整体替换（直接替换，
        无开关并存），本钉测沿原语义断言面重写：

        - 同实例 = 纯 resize 事务（原地继续是合法终态，式 1c 平局优
          先；足迹不变时零逐出零释放）；
        - 跨实例 = 目标实例按 D 阶段足迹先 _ensure_capacity，成功后
          原子换址（占用视图/指标锚单点生效）；
        - 容量失败 = KVCapacityError 且预约留在源实例、epoch 不在案
          （§16.4：改变方案提交失败且无不可逆工作时保留继续方案）。"""
        kv = _manager()
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=10)
        # 工作副本物化（边界真实形态：驻留 10 token = basis 足迹）。
        kv.prepare_prefill(
            session_id="s1", target_instance_index=0,
            history_tokens=0, trigger_request_id="r1")
        kv.expand_prefill(
            session_id="s1", instance_index=0, context_tokens=10,
            trigger_request_id="r1", reservation_request_id="r1")
        # 同实例：原地继续 = 纯 resize 事务（stay 足迹 = 准入足迹，
        # 零逐出零释放、无不可逆工作）。
        txn_stay = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=0, stage_action="stay",
            stage_epoch=1, history_tokens=0, input_tokens=10)
        self.assertEqual(txn_stay.target_instance_index, 0)
        self.assertEqual(txn_stay.evictions, ())
        self.assertEqual(txn_stay.released_shard_bytes, (0,) * kv.tp_degree)
        self.assertFalse(txn_stay.irreversible)
        self.assertFalse(txn_stay.aborted)
        # 跨实例：换址成功 + 占用视图随新实例（源实例零残留预约字节）。
        txn_move = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=2, history_tokens=0, input_tokens=10)
        self.assertEqual(txn_move.source_instance_index, 0)
        self.assertEqual(txn_move.target_instance_index, 1)
        self.assertFalse(txn_move.irreversible)  # 足迹已由准入预扣
        self.assertEqual(kv._reservations["r1"].instance_index, 1)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(0)),
            (0,) * kv.tp_degree)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)),
            kv_cache_shard_bytes_for_tokens(kv.model, 10, kv.tp_degree))
        # 同 epoch 重复提交（重复 P 完成通知的等价直连）：原子重放回执，
        # 不产生第二份换址。
        replay = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=2, history_tokens=0, input_tokens=10)
        self.assertIs(replay, txn_move)
        # epoch 回退/冲突重放 fail-closed。
        with self.assertRaises(ValueError):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=0,
                stage_action="stay", stage_epoch=1,
                history_tokens=0, input_tokens=10)

    def test_d18_boundary_reservation_move_capacity_failure_keeps_source(self):
        """目标容量不足（活跃会话占位 = 非合法 victim 的深缺口）→
        KVCapacityError 且原预约留在源实例、epoch 不在案（可换状态后
        同 epoch 重评）——继续方案保持合法。"""
        kv = _manager(capacity_bytes=100_000)
        # 实例 1 驻留 ACTIVE 会话（未 mark_complete，非合法 victim）。
        kv.prepare_prefill(
            session_id="hog", target_instance_index=1,
            history_tokens=0, trigger_request_id="hog")
        kv.expand_prefill(
            session_id="hog", instance_index=1, context_tokens=690,
            trigger_request_id="hog")
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=10)
        with self.assertRaises(KVCapacityError):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=1,
                stage_action="stay", stage_epoch=1,
                history_tokens=0, input_tokens=10)
        self.assertEqual(kv._reservations["r1"].instance_index, 0)
        self.assertIsNone(kv.last_stage_transaction("r1"))
        self.assertNotIn("r1", kv._stage_epochs)


# ============================================================ R15 模型 ==

def _model(loads, flows=None, factors=None):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads,
        flow_registry=flows or LinkFlowRegistry(),
        service_factors=factors or ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route2)


def _route2(source, target):
    return ((source, target), 1)


def _load(remaining=(10**9, 10**9), reclaimable=None, active_ns=0):
    return InstanceLoadView(
        instance_index=0, queued_task_load_ns=0,
        running_task_load_ns=active_ns, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(
            reclaimable if reclaimable is not None else remaining))


class SpaceAvailablePricingTest(unittest.TestCase):
    """D11（v2.0 §14，2026-09-27）：分阶段足迹 × 容量驱逐摘要 θ 份额
    消费 → 空间可用时刻 A = max(写回完成, 活跃排空)——驱逐决策时刻
    发射（旁路支链）、写回与排队重叠（D03 接口结论不变）。旧单向量
    聚合口径（R3' ``_space_available_ns_estimate``，再前身
    ``_eviction_wait_estimate``）随摘要计价退役：预计写回 = θ·脏向量
    （θ 由最重缺口 rank 决定、TP 耦合同比例），不是缺口全量。
    ``capacity_summary`` 缺席的夹具退 reclaimable 单桶合成锚。"""

    def test_deep_gap_uses_active_drain(self):
        # 缺口 (400,400)、单桶 R=(100,100)：θ=1、预计脏释放 (100,100)
        # → 写回 100+100/5=120；残量 300 → capacity_wait_uncertain；
        # A = max(120, 10^7) = 10^7（活跃排空主导）。
        model = _model(_load())
        notes = []
        space_ready = model._space_ready_ns_estimate(
            _load(remaining=(100, 100), reclaimable=(100, 100),
                  active_ns=10**7),
            PhaseFootprint.single(PHASE_DECODE, (500, 500)), notes,
            instance_index=0)
        self.assertGreaterEqual(space_ready, 10**7)
        self.assertIn("execution_growth_capacity_wait_uncertain", notes)
        self.assertNotIn("execution_growth_deep_gap_unresolved", notes)

    def test_shallow_gap_uses_pool_writeback(self):
        # 缺口 (400,400)、单桶 R=(10^9,10^9)：θ=400/10^9、预计脏释放
        # = θ·R = (400,400)（份额恰覆盖缺口）→ D04 逐 shard 写回：
        # 每 rank 独立池端口（除数 1）→ wall = 100 时延 +
        # max(400/5, 400/100) = 180；无残量 → 无不确定注记。
        load = _load(remaining=(100, 100), reclaimable=(10**9, 10**9),
                     active_ns=10**9)
        model = _model(_load())
        notes = []
        space_ready = model._space_ready_ns_estimate(
            load,
            PhaseFootprint.single(PHASE_DECODE, (500, 500)), notes,
            instance_index=0)
        self.assertEqual(space_ready, 100 + 400 // 5)
        self.assertNotIn("execution_growth_capacity_wait_uncertain", notes)

    def test_deep_gap_takes_writeback_drain_max(self):
        # A = max(写回, 排空) —— 排空 < 写回时 A = 写回（深缺口注记
        # 仍在：两事实分列）。缺口 (400,400)、R=(100,100)：θ=1、预计
        # 脏释放 (100,100)（不是缺口全量 400——耦合释放按桶份额）→
        # 写回 = 100 时延 + 100/5 = 120，活跃排空 50 ⇒ A = 120。
        model = _model(_load())
        notes = []
        space_ready = model._space_ready_ns_estimate(
            _load(remaining=(100, 100), reclaimable=(100, 100),
                  active_ns=50),
            PhaseFootprint.single(PHASE_DECODE, (500, 500)), notes,
            instance_index=0)
        self.assertEqual(space_ready, 100 + 100 // 5)
        self.assertIn("execution_growth_capacity_wait_uncertain", notes)

    def test_writeback_composition_values_in_notes(self):
        # M4：写回估计的构成值随注记落盘（零 schema 变更——notes 恒
        # str 列表）——D11 起为摘要估计构成值：gap_bytes=400（最重缺
        # 口）、beyond_reclaim=全桶消费后残量（400−θ·R = 100）、
        # pool_divisor=1。
        model = _model(_load())
        notes = []
        model._space_ready_ns_estimate(
            _load(remaining=(100, 100), reclaimable=(300, 300)),
            PhaseFootprint.single(PHASE_DECODE, (500, 500)), notes,
            instance_index=0)
        self.assertIn(
            "execution_growth_eviction_writeback_composition"
            "(gap_bytes=400,beyond_reclaim=100,pool_divisor=1)",
            notes)
        # 深缺口路径注记与构成值共存（两事实分列）。
        notes_deep = []
        model._space_ready_ns_estimate(
            _load(remaining=(100, 100), reclaimable=(300, 300),
                  active_ns=10**7),
            PhaseFootprint.single(PHASE_DECODE, (500, 500)), notes_deep,
            instance_index=0)
        self.assertIn(
            "execution_growth_eviction_writeback_composition"
            "(gap_bytes=400,beyond_reclaim=100,pool_divisor=1)",
            notes_deep)
        self.assertIn("execution_growth_capacity_wait_uncertain", notes_deep)


class ServiceFactorsAlphaTest(unittest.TestCase):
    """D14（v2.0 §9.2）：固定 1/16 EWMA 平滑效率 η（首样本同规则、自
    η=1 出发），因子 = 1/η_new；同时刻汇总；非法样本拒绝；亚 roofline
    样本记异常跳过。"""

    def test_fixed_coef_ewma_from_first_sample(self):
        factors = ServiceFactors()
        factors.observe_prefill(
            actual_ns=100, base_ns=100, tick_ns=1000)
        factors.flush()
        # 首样本也按同一更新规则（不再直接初始化）：
        # 15/16·1 + 1/16·1 = 1（样本恰在 roofline 上不动）。
        self.assertEqual(factors.prefill_factor, 1.0)
        self.assertEqual(factors.updates["prefill_factor"], 1)
        factors.observe_prefill(
            actual_ns=300, base_ns=100, tick_ns=2000)
        factors.flush()
        # 固定系数（Δt 不再进入系数）：η_sample = 1/3，
        # η = 15/16 + 1/48 = 23/24，因子 = 24/23。
        self.assertAlmostEqual(factors.prefill_factor, 24.0 / 23.0,
                               places=12)
        self.assertEqual(factors.updates["prefill_factor"], 2)

    def test_same_tick_samples_aggregate(self):
        factors = ServiceFactors()
        factors.observe_decode(
            actual_ns=100, base_ns=100, tick_ns=5)
        factors.observe_decode(
            actual_ns=300, base_ns=100, tick_ns=5)
        factors.flush()
        # 同时刻先汇总：η_sample = Σbase/Σactual = 200/400 = 1/2
        # （一条更新）；η = 15/16 + 1/32 = 31/32，因子 = 32/31。
        self.assertAlmostEqual(factors.decode_factor, 32.0 / 31.0,
                               places=12)
        self.assertEqual(factors.updates["decode_factor"], 1)

    def test_sub_roofline_sample_anomalous_skipped(self):
        # D14：0 < η_sample ≤ 1 有效窗口（除数口径 ratio ≥ 1）外的
        # 亚 roofline 样本（实测快于 roofline）= 记录异常并跳过，
        # 不钳位、不更新（prefill/decode 亦适用）。
        factors = ServiceFactors()
        factors.observe_decode(
            actual_ns=100, base_ns=200, tick_ns=1)
        factors.flush()
        self.assertEqual(factors.decode_factor, 1.0)
        self.assertNotIn("decode_factor", factors.updates)
        self.assertEqual(factors.anomalous.get("decode_factor"), 1)
        self.assertNotIn("decode_factor", factors.rejected)

    def test_invalid_samples_rejected(self):
        factors = ServiceFactors()
        factors.observe_prefill(
            actual_ns=0, base_ns=100, tick_ns=1)
        factors.observe_prefill(
            actual_ns=100, base_ns=0, tick_ns=1)
        self.assertEqual(factors.rejected["prefill_factor"], 2)
        self.assertEqual(factors.prefill_factor, 1.0)


class DivisorMultiTest(unittest.TestCase):
    """R15-1/F-B：非代表路径上的争用被计价（并集瓶颈）。"""

    def test_non_representative_path_contention_counted(self):
        registry = LinkFlowRegistry()
        # 代表对路径 (0->2)；非代表 TP shard 路径 (1->3) 有他流登记。
        registry.register(1, 3)
        divisor = registry.divisor_multi(
            ((0, 2), (1, 3)), include_self=True)
        self.assertEqual(divisor, 2)

    def test_self_multiplicity_on_shared_link(self):
        registry = LinkFlowRegistry()
        # 候选自身两条 TP 路径共享链路 (4,5)：自身份额 = 2。
        divisor = registry.divisor_multi(
            ((4, 5), (4, 5)), include_self=True)
        self.assertEqual(divisor, 2)

    def test_owner_release(self):
        registry = LinkFlowRegistry()
        flow_id = registry.register_path((0, 1, 2), owner="r1#decode")
        self.assertEqual(registry.divisor((0, 1), include_self=False), 1)
        self.assertEqual(
            registry.divisor((0, 1), include_self=True), 2)
        registry.release_owner("r1#decode")
        self.assertEqual(registry.snapshot(), {})
        with self.assertRaises(Exception):
            registry.release_flow(flow_id)


# ======================================================== R14 调度器侧 ==

class DecodeStallMachineryTest(unittest.TestCase):
    """R14：停滞登记 / 唤醒键 / 死锁守卫（__new__ 壳夹具）。"""

    def _scheduler(self):
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState, _OnlineRequestRuntime,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.instances = [_OnlineInstanceState(index=0),
                               _OnlineInstanceState(index=1)]
        scheduler.kv_manager = _manager()
        scheduler._kv_ledger_epoch = 0
        scheduler._stalled_by_instance = {}
        scheduler._batch = {"tick": 0}
        scheduler.online_log_count = 0
        scheduler.decision_log_sink = None
        scheduler.online_log_rows = []
        # 自查 A 后守卫引用的跨 tick 事件源状态（空 = 无未来事件源，
        # 守卫可在全停滞时触发）。
        scheduler._pending_merge_alarms = {}
        scheduler.arrival_heap = []
        scheduler.runtimes = []
        scheduler._arrived_request_count = 0
        runtime = _OnlineRequestRuntime({
            "request_id": "r0", "session_id": "s0", "turn_index": 0,
            "queue_index": 0, "prefill_length": 512, "decode_length": 8,
            "history_tokens_before": 0,
            "prefill_context_tokens": 512,
            "final_context_tokens": 520,
        }, 512)
        runtime.decode_instance_index = 0
        return scheduler, runtime

    def test_stall_wake_and_deadlock_guard(self):
        scheduler, runtime = self._scheduler()
        # 会话在 kv 账本就位（wake 重试读真实 manager）。
        scheduler.kv_manager.prepare_prefill(
            session_id="s0", target_instance_index=0,
            history_tokens=0, trigger_request_id="r0")
        scheduler.kv_manager.expand_prefill(
            session_id="s0", instance_index=0, context_tokens=512,
            trigger_request_id="r0")
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        self.assertTrue(runtime.decode_stalled)
        self.assertIn(0, scheduler._stalled_by_instance)
        # 唤醒键：KV 纪元未变且实例纪元未变 → 不重试（无自旋）。
        key_before = runtime.stall_wake_key
        scheduler._wake_stalled_decodes(0)
        self.assertEqual(runtime.stall_wake_key, key_before)
        self.assertTrue(runtime.decode_stalled)
        # 容量释放（KV 纪元 bump）→ 重试窗口打开（会话上下文已就位则
        # 直接唤醒退出停滞）。
        scheduler._bump_kv_ledger_epoch()
        scheduler._wake_stalled_decodes(0)
        self.assertFalse(runtime.decode_stalled)
        self.assertNotIn(0, scheduler._stalled_by_instance)
        # 重新停滞且无任何事件源 → 死锁守卫显式 fail-closed。
        scheduler.kv_manager._sessions["s0"].context_tokens = 0
        runtime.decode_tokens_consumed = 1
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        scheduler.instances[0].active_decode.append(runtime)
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._check_decode_deadlock()
        self.assertIn("deadlock", str(ctx.exception))

    def test_guard_passes_when_work_remains(self):
        scheduler, runtime = self._scheduler()
        scheduler.instances[0].qp.append(runtime)
        runtime.remaining_chunks = 2
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        # qp 仍有 chunk 工作 → 有事件源，守卫不触发。
        scheduler._check_decode_deadlock()


# =========================================================== R12 水印 ==

class WatermarkJointReplayTest(unittest.TestCase):
    """R12：joint 双驻留重放原语（stash/merge 结算）。"""

    def _replay(self):
        sys.path.insert(0, os.path.join(_WL, "..", "..", "slo_tools"))
        import hbm_watermark
        mapping = hbm_watermark.REPO_VARIANTS["astra-sim-joint"]
        return hbm_watermark.WatermarkReplay(
            "astra-sim-joint", mapping,
            {"requests": {}, "path": "test"}, coef_bytes_per_token=100,
            capacity_per_instance=None)

    def test_stash_and_merge_settle(self):
        replay = self._replay()
        # 实例 0 驻留 1000B → 跨实例工作副本（实例 1 搬入 600B）。
        replay.apply_grow(1, "s", 0, 10, "prefill_grow")
        replay.stash_base("s")
        session = replay.sessions["s"]
        self.assertEqual((session.base_instance, session.base_bytes), (0, 1000))
        self.assertIsNone(session.instance)
        # merge：noc 增量 200B 回 home；执行端释放。
        replay._apply(2, 1, 600)
        session.bytes = 600
        session.instance = 1
        self.assertEqual(replay.occupancy, {0: 1000, 1: 600})
        # 模拟 _joint_completion 的核心结算序。
        replay._apply(3, 1, -600)
        session.bytes = 0
        session.instance = None
        replay._apply(3, 0, 200)  # home 侧只加增量（base 未离开）
        session.instance = 0
        session.bytes = 1200
        session.base_instance = None
        session.base_bytes = 0
        self.assertEqual(replay.occupancy, {0: 1200, 1: 0})

    def test_double_stash_fails_closed(self):
        replay = self._replay()
        replay.apply_grow(1, "s", 0, 10, "prefill_grow")
        replay.stash_base("s")
        import hbm_watermark
        with self.assertRaises(hbm_watermark.SloToolError):
            replay.stash_base("s")


if __name__ == "__main__":
    unittest.main()
