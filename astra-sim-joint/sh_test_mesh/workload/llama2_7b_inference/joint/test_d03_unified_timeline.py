#!/usr/bin/env python3
"""test_d03_unified_timeline.py -- D03 Q/驱逐/准备统一时间轴单测
（v2.0 §8.3/§11.1/§16.2/§22-D03，2026-09-26）。

语义合同（joint_cost_model 两入口关键路径合成）：

* ``target_wait_ns``（Q）为**观测字段**（breakdown 审计位），不再被
  所有动作机械前置相加；Q 只门控计算（式 13 首层 F_0 = Q、A^work =
  A）；
* 空间可用时刻 A（``_space_available_ns_estimate``）成为相对本次决策
  的接口：式(12) A_dest——依赖驱逐后才能写目标的准备腿不得早于 A
  （不能用 max 把尚无空间的写入提前）；式(13) A^work；
* **发射 frontier 核对**（验收红线——不能仅改预测器制造不存在的重
  叠）：P 准入三腿（reserve/prepare/prefill-read）在真实执行侧均决
  策时刻发射（SH 准入事务 at_ns=now、流表准入相登记、请求其后入队
  ——sh30_online_scheduler 准入链 + graph_batch_builder 准入主链/
  旁挂支链），故 P 准备腿自 0 起估、与排队重叠是真实执行顺序；
  D 边界迁移/credit 腿**随加入列车发射**（pending_decode_ready → 下
  一列车，"迁移腿在加入列车才发射，此刻零字节移动"）——自
  max(Q, A) 起估、不与 D 等待重叠；边界逐出旁路支链即时——A 自决策
  时刻起（与 D 等待重叠，真实）；
* 验收两条（v2.0 §22-D03）：Q=100、空间和准备在计算前已可用的例子
  不再重复支付准备时间；依赖驱逐后才能写目标时不用 max 把尚无空间
  的写入提前。

手算锚硬件（与 test_d02 同族）：B_link = 10 B/ns、B_pool = 5 B/ns、
B_HBM = 100 B/ns、单跳时延 10 ns、池端口事务时延 100 ns、流注册表
空、u_port = 0（离线口径）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d03_unified_timeline.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_STAY,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    ServiceFactors,
    SessionKVView,
)


# ============================================================ 夹具构造 ==

def _rates(pool_latency_ns=100, d2d_latency_ns=10):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=d2d_latency_ns, pool_latency_ns=pool_latency_ns)


def _load(index, total=0, running=0, active=0,
          remaining=(10 ** 9, 10 ** 9), reclaimable=None):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=running, active_decode_task_load_ns=active,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(
            remaining if reclaimable is None else reclaimable))


def _session(home=1, resident=1, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


def _route(source, target):
    return ((source, target), 1)


def _model(loads, *, layers=4):
    return JointCostModel(
        rates=_rates(), loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=layers, instance_tp_size=2, route_fn=_route)


def _note_value(breakdown, key):
    prefix = key + "="
    for note in breakdown.notes:
        if note.startswith(prefix):
            return int(note[len(prefix):])
    raise AssertionError(f"note {key!r} missing in {breakdown.notes}")


def _stay_partial(model, loads, total_on_target=0):
    """stay-PARTIAL：缺失后缀 (1000,1000)/rank 池恢复（D04 逐 shard：
    restore = 100 + 1000/5 = 300）；驻留前缀两层无门。"""
    session = _session(
        home=1, resident=1, history_bytes=(1000, 1000),
        missing=(1000, 1000), location="partial_hbm_remote", prefix=2)
    return model.estimate_P_with_continuation(
        session=session, request=_request(), instance_index=1,
        action=ACTION_STAY, remote_enabled=True)


KV_TOKEN_BYTES = 128
DECODE_STEPS = 8
DECODE_COMPUTE_NS = DECODE_STEPS * 2  # decode_ns_per_token = 2.0


def _complete_boundary(**overrides):
    values = dict(
        session_id="s", home_instance=0, p_instance_index=0,
        p_action="stay", history_tokens=100,
        working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2,
        base_bytes_by_tp_rank=(0, 0), decode_steps=DECODE_STEPS,
        base_tokens=100, kv_token_bytes=KV_TOKEN_BYTES)
    values.update(overrides)
    return PDBoundaryView(**values)


def _d_model(loads):
    return JointCostModel(
        rates=_rates(), loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        remote_read_kv_token_bytes=KV_TOKEN_BYTES)


# ===================================================== P 入口时间轴锚 ==

class PObservationFieldTest(unittest.TestCase):
    """Q = target_wait 为观测字段 + 计算门（零门控守恒锚不漂移）。"""

    def test_zero_gate_anchor_q_folds_into_recursion_only(self):
        # stay-LOCAL 零门控：递推退化为 Q + 整段计算；cost 与旧串行
        # 口径逐位一致（5000 + 50 + 20）——守恒锚。
        model = _model({0: _load(0), 1: _load(1, total=5000)})
        candidate = model.estimate_P_with_continuation(
            session=_session(home=1, resident=1),
            request=_request(), instance_index=1,
            action=ACTION_STAY, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.target_wait_ns, 5000)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            5000 + 50)
        self.assertEqual(candidate.cost_ns, 5000 + 50 + 20)
        # A = 0（无缺口）；frontier 前置废除后 cost 无第二次 Q。
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 0)
        self.assertEqual(_note_value(candidate.breakdown, "space_ready_ns"),
                         0)

    def test_prep_issue_frontier_note_disclosed(self):
        model = _model({0: _load(0), 1: _load(1, total=5000)})
        candidate = model.estimate_P_with_continuation(
            session=_session(home=1, resident=1),
            request=_request(), instance_index=1,
            action=ACTION_STAY, remote_enabled=True)
        # 发射 frontier 说明（在线调度/构图核对结论）随候选落盘。
        self.assertIn("prep_issue_frontier=P_prep_at_decision_epoch",
                      candidate.breakdown.notes)


class QOverlapsPreparationTest(unittest.TestCase):
    """验收 1（v2.0 §22-D03）：Q=100、空间和准备在计算前已可用 ⇒ 不
    再重复支付准备时间（可提前的准备自 0 起估、与排队重叠）。"""

    def test_q100_restore_overlaps_drain(self):
        # 手算（D04 逐 shard 池腿）：restore = 100 + 1000/5 = 300
        # （A = 0，容量无缺口）；ready 后缀层 = 0 + 300 = 300（决策相
        # 对绝对量）。F_0 = Q = 100：L1 = 113、L2 = 126、
        # L3 = max(126, 300)+12 = 312、L4 = 324 ⇒ cost = 324 + 20 =
        # 344。
        model = _model({0: _load(0), 1: _load(1, total=100)})
        candidate = _stay_partial(model, model.loads)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.target_wait_ns, 100)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            324)
        self.assertEqual(candidate.cost_ns, 344)
        # 旧串行前置 = frontier(100) + 相对递推(324) + 20 = 444——
        # 重叠 credit = 100（恢复腿真实飞行于 [0,300]，与排队重叠）。
        self.assertLess(candidate.cost_ns, 100 + 324 + 20)

    def test_copy_chunk_chain_absorbed_by_drain(self):
        # copy@异地：历史 (1600,3200)/rank、共享链路除数 2（缺省代表
        # 路径）→ 单 chunk 链 X = 330 / 650、链壁 = 650（A = 0）；
        # Q = 1000 ⊇ 链 ⇒ 递推 = Q + 整段计算 = 1050 ⇒ cost =
        # 1050 + 20 = 1070（旧串行前置 1000 + 650 + 20 = 1670 双计
        # 排队×交接重叠——交接腿真实飞行于排队视界内）。
        model = _model({0: _load(0), 1: _load(1, total=1000)})
        session = _session(
            home=0, resident=0, history_bytes=(1600, 3200),
            missing=(0, 0), location="local_hbm", prefix=4)
        candidate = model.estimate_P_with_continuation(
            session=session, request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(
            _note_value(candidate.breakdown, "copy_handoff_first_chunk_ns"),
            650)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            1000 + 50)
        self.assertEqual(candidate.cost_ns, 1000 + 50 + 20)


class EvictionGatedWriteTest(unittest.TestCase):
    """验收 2（v2.0 §22-D03）：依赖驱逐后才能写目标时，不能用 max 把
    尚无空间的写入提前——式(12) A_dest = A 门控目标写腿。"""

    def test_restore_write_not_earlier_than_space_ready(self):
        # 手算（D11 摘要计价）：stay 足迹 = missing+input+growth =
        # 1000+25+5 = 1030/rank；remaining (20,20) ⇒ gap 1010/rank。
        # reclaimable 缺省 = remaining = (20,20) ⇒ 合成单桶 θ=1、预计
        # 脏释放 (20,20)（桶份额，非缺口全量）⇒ A = 写回 = pool(20) =
        # 100+20/5 = 104；残量 990 → capacity_wait_uncertain（running+
        # active = 0，A 不被排空抬升）。restore ready = A + 300 = 404
        # （**不是** max(104, 300) = 300——后者把尚无空间的写提前）。
        # F_0 = Q = 100：L1 = 113、L2 = 126、L3 = max(126, 404)+12 =
        # 416、L4 = 428 ⇒ cost = 428 + 20 = 448。
        model = _model({0: _load(0), 1: _load(1, total=100,
                                              remaining=(20, 20))})
        candidate = _stay_partial(model, model.loads)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 104)
        self.assertEqual(_note_value(candidate.breakdown, "space_ready_ns"),
                         104)
        self.assertIn("execution_growth_capacity_wait_uncertain",
                      candidate.breakdown.notes)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            428)
        self.assertEqual(candidate.cost_ns, 448)
        # 反证（形态级）：旧 max(驱逐, 恢复) 形态给 300 + 24 = 324
        # （写提前到无空间时刻）——不得回归。
        self.assertNotEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            324)

    def test_deep_gap_space_ready_is_absolute_drain_time(self):
        # 深缺口：needed ⊃ reclaimable ⇒ A = max(写回, 活跃排空
        # running+active = 700)（绝对时刻）；恢复写 = A + 300 =
        # 1000 ≥ A——不得早于排空时刻。
        model = _model({0: _load(0), 1: _load(
            1, total=100, running=300, active=400,
            remaining=(20, 20), reclaimable=(0, 0))})
        candidate = _stay_partial(model, model.loads)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 700)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            700 + 300 + 12 + 12)
        # D11：零可回收 ⇒ 无写回腿、A 全由活跃排空给出、标不确定。
        self.assertIn("execution_growth_capacity_wait_uncertain",
                      candidate.breakdown.notes)
        self.assertNotIn("execution_growth_deep_gap_unresolved",
                         candidate.breakdown.notes)


# ===================================================== D 入口时间轴锚 ==

class DBoundaryTimelineTest(unittest.TestCase):
    """D 边界发射 frontier：迁移/credit 腿随加入列车（自 max(Q, A)
    起、不与 D 等待重叠）；边界逐出旁路即时（A 自决策时刻起、与等待
    重叠）。"""

    def _loads(self, target_total=0, target_remaining=(10 ** 9, 10 ** 9),
               target_reclaimable=None):
        return {
            0: _load(0),
            1: _load(1, total=target_total,
                     remaining=target_remaining,
                     reclaimable=target_reclaimable),
        }

    def test_continuation_writeback_overlaps_wait(self):
        # 原地继续 stay@i_P：growth = 128×8 = 1024/rank；
        # remaining (1000,1000) ⇒ gap 24/rank ⇒ A = pool(24)
        # = 100 + ⌈24/5⌉ = 105（D17：带宽服务向上取整，24/5=4.8）。
        # Q = running+active = 100 ⇒ cost = max(100, 105) + 16 = 121
        # （写回真实飞行于 [0,105] ⊆ 等待视界；旧串行 = 100 + 105 +
        # 16 = 221 双计）。
        model = _d_model({
            0: _load(0, running=100, remaining=(1000, 1000)),
            1: _load(1)})
        boundary = _complete_boundary()
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY,
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertTrue(candidate.is_continuation)
        self.assertEqual(candidate.breakdown.target_wait_ns, 100)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 105)
        self.assertEqual(candidate.cost_ns, 105 + DECODE_COMPUTE_NS)
        self.assertIn(
            "prep_issue_frontier=D_prep_at_train_join,"
            "D_evict_at_decision_epoch",
            candidate.breakdown.notes)

    def test_deep_gap_drain_within_wait_no_double_count(self):
        # 深缺口（reclaimable 0）：A = max(写回, 排空 1e7)；原地 Q =
        # 1e7 ⊇ 排空 ⇒ max(Q, A) = max(Q, 写回) —— K5 抵扣语义由
        # max 合成自然成立（无串行加法即无被覆盖分量双计）。
        model = _d_model({
            0: _load(0, running=10 ** 7, remaining=(1000, 1000),
                     reclaimable=(0, 0)),
            1: _load(1)})
        boundary = _complete_boundary()
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY,
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        # 写回 = pool(24) = 104 < 1e7 ⇒ A = 1e7 = Q ⇒ cost = Q + 16。
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 10 ** 7)
        self.assertEqual(
            candidate.cost_ns, 10 ** 7 + DECODE_COMPUTE_NS)

    def test_move_leg_not_overlapped_with_wait(self):
        # 完整型迁移 copy@j：迁移腿随加入列车发射（发射 frontier =
        # 列车体前）——cost = max(Q, A) + X_move + 计算 + merge 零翻
        # 转。**不得**自决策时刻起发（不虚构与 D 等待的重叠）。
        target_total = 5000
        model = _d_model(self._loads(target_total=target_total))
        boundary = _complete_boundary()
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy",
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertFalse(candidate.is_continuation)
        self.assertEqual(candidate.breakdown.target_wait_ns, target_total)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 0)
        # 迁移腿完整串在 max(Q, A) 之后（重叠未发生——执行侧"迁移腿
        # 在加入列车才发射，此刻零字节移动"）。
        self.assertEqual(
            candidate.cost_ns,
            max(target_total, 0)
            + candidate.breakdown.history_prep_ns
            + DECODE_COMPUTE_NS + 0)
        self.assertGreaterEqual(
            candidate.cost_ns, target_total
            + candidate.breakdown.history_prep_ns)

    def test_move_write_gated_by_space_ready(self):
        # 验收 2（D 侧）：目标空间缺口（working+growth = 13824/rank，
        # remaining (100,100)）⇒ D11 摘要计价：reclaimable 缺省 =
        # remaining = (100,100)，合成单桶 θ=1、预计脏释放 (100,100)
        # （桶份额，非缺口全量 13724）⇒ A = pool(100) = 100+100/5 =
        # 120 > Q = 0（残量 13624 → capacity_wait_uncertain；running+
        # active = 0 不抬升）——迁移腿完成 = max(Q, A) + X（写不得早
        # 于空间可用；旧 max(X, A) 形态把写提前到尚无空间时刻，废除）。
        model = _d_model(self._loads(
            target_remaining=(100, 100)))
        boundary = _complete_boundary()
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy",
            remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 120)
        self.assertGreaterEqual(
            _note_value(candidate.breakdown, "space_ready_ns"), 120 - 1)
        self.assertEqual(
            candidate.cost_ns,
            120 + candidate.breakdown.history_prep_ns
            + DECODE_COMPUTE_NS + 0)
        self.assertEqual(_note_value(candidate.breakdown, "space_ready_ns"),
                         120)


if __name__ == "__main__":
    unittest.main()
