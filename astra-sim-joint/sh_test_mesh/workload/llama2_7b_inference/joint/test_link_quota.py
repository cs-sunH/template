#!/usr/bin/env python3
"""test_link_quota.py -- WP3a 配额模块（link_quota.py）+ JOINT_QUOTA_MODE
开关解析的定向验收单测（C9 卡；零后端、python 级构造性用例）。

覆盖（卡内测试清单）：

* Q_init 硬件派生：rho_eff 与 max(1, floor(rho))（base 配置
  4050/1640 → 2 只是派生结果；rho<1 → 1 = 结构性防死锁下限）；
* 端口平价门：B_HBM/(u_port+1) >= r_hat_KV；忙端口即 0、无 max(1,.)
  保底；headroom 余量；
* 三类流各自判据：realtime（链路门+平价门）/ elastic（链路门+并发
  上限、不套平价门）/ oneshot（链路计数入占用、端口无门、在册收紧
  后续实时流）；
* 预留借还配对：merge 双向各 1 槽 + 双候选胜者侧 bulk 名额、
  N_bulk=Q_init 封顶、service_done 裁决释放败者侧、merge_done 释放
  胜者侧、失败注入（双释放/漏释放/重复预留/部分失败零登记）；
* deferred 重试键：配额代数在流 settle/预留释放/裁决处 bump、
  extend_retry_key 组合；
* δ_adm 近平局带：margin >= delta 才翻转、δ=0（冻结初值）与既有
  argmin 比较组合后零行为变更；
* quota_deferred 回队：全部动作不可行 → 回队记录而非
  JointSchedulerError（对照 joint_scheduler.py:156-159 零候选
  fail-closed 会终局 run）；
* joint_config 解析：off/static/aimd 三值合法、非法值 fail-closed、
  缺省 off（F7）、manifest 含该键、与 combo 预设正交。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_link_quota.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT, os.path.join(_PARENT, "online")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import (  # noqa: E402
    JointConfigError,
    parse_joint_config,
)
from joint.joint_scheduler import JointSchedulerError  # noqa: E402
from joint.link_quota import (  # noqa: E402
    AIMD_ACTION_COMFORT,
    AIMD_ACTION_COMFORT_FROZEN,
    AIMD_ACTION_EXPAND,
    AIMD_ACTION_SHRINK,
    AIMD_SIGNAL_COMFORT,
    AIMD_SIGNAL_SHRINK,
    DELTA_ADM_INITIAL_NS,
    FLOW_ELASTIC,
    FLOW_ONESHOT,
    FLOW_REALTIME,
    MERGE_DIRECTION_FORWARD,
    MERGE_DIRECTION_REVERSE,
    MERGE_RESERVE_OWNER_TEMPLATE,
    QUOTA_AIMD,
    QUOTA_MODES,
    QUOTA_OFF,
    QUOTA_STATIC,
    WAIT_CAPACITY,
    WAIT_QUOTA_LINK,
    WAIT_QUOTA_PORT,
    LinkQuotaError,
    LinkQuotaTracker,
    challenger_flips,
    derive_q_init,
    derive_rho_eff,
    extend_retry_key,
    port_parity_admits,
    port_parity_headroom,
    quota_deferred_requeue,
    reachable_domain,
)

#: base 配置锚点（设计文档 §1.1：D2D 4050 GB/s、HBM 1640 GB/s）。
_BASE_NOC = 4050.0
_BASE_HBM = 1640.0


def _tracker(mode=QUOTA_STATIC, noc=_BASE_NOC, hbm=_BASE_HBM, **kwargs):
    return LinkQuotaTracker(
        mode=mode, noc_link_bytes_per_ns=noc, local_hbm_bytes_per_ns=hbm,
        **kwargs)


# ====================================================== Q_init 硬件派生 ==


class QuotaDerivationTest(unittest.TestCase):
    """Q_init = max(1, floor(rho_eff))：逐配置硬件派生，非固定常数。"""

    def test_base_config_derives_two(self):
        # 4050/1640 ≈ 2.4695 → floor = 2——派生结果，不是"每链路两条流"
        # 的固定常数（F2；设计文档 §1.2"两条并发流不是端到端性能保证"）。
        rho = derive_rho_eff(_BASE_NOC, _BASE_HBM)
        self.assertAlmostEqual(rho, 2.4695121951, places=9)
        self.assertEqual(derive_q_init(rho), 2)
        tracker = _tracker()
        self.assertEqual(tracker.q_init, 2)
        self.assertEqual(tracker.n_bulk, 2)   # N_bulk(port) = Q_init（冻结）
        self.assertEqual(tracker.rho_eff, rho)

    def test_rho_below_one_floors_at_one(self):
        # rho<1 配置（如 1200/1640 ≈ 0.73）：Q_init = 1——max(1,.) 仅防
        # 结构性死锁；配额不关闭，域空化由定价选中率表达（argmin 涌现）。
        self.assertEqual(derive_q_init(0.7317073), 1)
        self.assertEqual(derive_q_init(0.0), 1)

    def test_floor_boundary(self):
        self.assertEqual(derive_q_init(2.0), 2)
        self.assertEqual(derive_q_init(1.9999999), 1)
        self.assertEqual(derive_q_init(3.7), 3)

    def test_invalid_inputs_fail_closed(self):
        for noc, hbm in ((0.0, 1640.0), (-1.0, 1640.0),
                         (float("nan"), 1640.0), (4050.0, 0.0)):
            with self.assertRaises(LinkQuotaError):
                derive_rho_eff(noc, hbm)
        for rho in (-0.1, float("inf")):
            with self.assertRaises(LinkQuotaError):
                derive_q_init(rho)

    def test_tracker_rejects_unknown_mode(self):
        with self.assertRaises(LinkQuotaError):
            _tracker(mode="dynamic")
        self.assertEqual(QUOTA_MODES, ("off", "static", "aimd"))


# ======================================================== 端口平价门 ==


class PortParityGateTest(unittest.TestCase):
    """B_HBM/(u_port+1) >= r_hat_KV；端口无 max(1,.) 保底（忙端口即 0）。"""

    def test_admits_when_rate_sufficient(self):
        # B=100, r̂=50：u_port=1 → 100/2 = 50 >= 50（边界含等号）。
        self.assertTrue(port_parity_admits(100.0, 0, 50.0))
        self.assertTrue(port_parity_admits(100.0, 1, 50.0))
        self.assertFalse(port_parity_admits(100.0, 2, 50.0))

    def test_busy_port_admits_nothing_no_floor(self):
        # 忙端口即 0：u_port 高到分母压平带宽 → 全拒；无任何保底名额。
        self.assertFalse(port_parity_admits(100.0, 100, 50.0))
        # r̂ > B：即使 u_port=0 也无准入（对比链路侧 max(1,.) 下限）。
        self.assertFalse(port_parity_admits(100.0, 0, 150.0))

    def test_headroom_counts_remaining_realtime_slots(self):
        # k_max = floor(B/r̂) − u：u=0/r̂=50 → k=2（B/2=50 含边界）。
        self.assertEqual(port_parity_headroom(100.0, 0, 50.0), 2)
        self.assertEqual(port_parity_headroom(100.0, 1, 50.0), 1)
        self.assertEqual(port_parity_headroom(100.0, 0, 10.0), 10)
        self.assertEqual(port_parity_headroom(100.0, 0, 150.0), 0)

    def test_invalid_parity_inputs_fail_closed(self):
        with self.assertRaises(LinkQuotaError):
            port_parity_admits(100.0, -1, 50.0)
        with self.assertRaises(LinkQuotaError):
            port_parity_admits(0.0, 0, 50.0)
        with self.assertRaises(LinkQuotaError):
            port_parity_admits(100.0, 0, 0.0)


# ======================================================= 三类流判据 ==


class RealtimeFlowTest(unittest.TestCase):
    """实时流（decode remote-read 读流）：链路门 + 端口平价门。"""

    def setUp(self):
        # noc=4, hbm=2 → rho=2 → Q_init=2（小数值派生，同源公式）。
        self.t = _tracker(noc=4.0, hbm=2.0)
        self.assertEqual(self.t.q_init, 2)

    def test_link_gate_fills_then_defers(self):
        v1 = self.t.admit_flow(
            owner="r1#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0, r_hat_kv_bytes_per_ns=0.5)
        v2 = self.t.admit_flow(
            owner="r2#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0, r_hat_kv_bytes_per_ns=0.5)
        self.assertTrue(v1.admitted and v2.admitted)
        v3 = self.t.admit_flow(
            owner="r3#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0, r_hat_kv_bytes_per_ns=0.5)
        self.assertFalse(v3.admitted)
        self.assertEqual(v3.wait_reason, WAIT_QUOTA_LINK)
        self.assertEqual(v3.resource_kind, "link")
        self.assertEqual(v3.remaining, 0)
        # inapplicable_reason 注明资源种类与余量（实例不掩码——裁决只
        # 标动作，无异常抛出）。
        self.assertIn("quota_link", v3.inapplicable_reason)
        self.assertIn("remaining=0", v3.inapplicable_reason)

    def test_parity_gate_defers_with_quota_port(self):
        # hbm=2.0, r̂=1.5：u=0 → 2.0/1 = 2.0 >= 1.5 过；u=1 → 1.0 < 1.5 拒。
        v1 = self.t.admit_flow(
            owner="r1#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=5, r_hat_kv_bytes_per_ns=1.5)
        self.assertTrue(v1.admitted)
        v2 = self.t.admit_flow(
            owner="r2#decode#0", flow_class=FLOW_REALTIME,
            links=[(1, 2)], port_id=5, r_hat_kv_bytes_per_ns=1.5)
        self.assertFalse(v2.admitted)
        self.assertEqual(v2.wait_reason, WAIT_QUOTA_PORT)
        self.assertEqual(v2.resource_kind, "port")

    def test_u_port_base_from_load_view_tightens_gate(self):
        # 外部活跃 decode 消费流（负载视图）计入分母：u_port_base=1 时
        # 首条实时流即被平价门拒（忙端口即 0）。
        v = self.t.admit_flow(
            owner="r1#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=5, r_hat_kv_bytes_per_ns=1.5,
            u_port_base=1)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)

    def test_realtime_requires_r_hat(self):
        with self.assertRaises(LinkQuotaError):
            self.t.admit_flow(
                owner="r1#decode#0", flow_class=FLOW_REALTIME,
                links=[(0, 1)], port_id=5)

    def test_atomic_no_partial_enrollment_on_failure(self):
        # 链路 (0,1) 已满、链路 (1,2) 空：跨两链路的流整体不借——
        # 两条链路都不得残留登记。
        for owner in ("a#decode#0", "b#decode#0"):
            self.assertTrue(self.t.admit_flow(
                owner=owner, flow_class=FLOW_REALTIME,
                links=[(0, 1)], port_id=0,
                r_hat_kv_bytes_per_ns=0.1).admitted)
        v = self.t.admit_flow(
            owner="c#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1), (1, 2)], port_id=9,
            r_hat_kv_bytes_per_ns=0.1)
        self.assertFalse(v.admitted)
        self.assertEqual(self.t.link_occupancy((1, 2)), 0)
        self.assertEqual(self.t.port_enrolled(9), 0)

    def test_duplicate_owner_fails_closed(self):
        self.assertTrue(self.t.admit_flow(
            owner="r1#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0, r_hat_kv_bytes_per_ns=0.5).admitted)
        with self.assertRaises(LinkQuotaError):
            self.t.admit_flow(
                owner="r1#decode#0", flow_class=FLOW_REALTIME,
                links=[(2, 3)], port_id=1, r_hat_kv_bytes_per_ns=0.5)


class ElasticFlowTest(unittest.TestCase):
    """弹性流（remote-read prefill 首遍读流）：并发上限 ≤ Q，不套平价门。"""

    def setUp(self):
        self.t = _tracker(noc=4.0, hbm=2.0)   # Q_init = 2
        self.assertEqual(self.t.q_init, 2)

    def test_admitted_even_when_parity_would_fail(self):
        # r̂=1.5 下平价门在 u=1 即拒（见 RealtimeFlowTest）；弹性流无
        # r̂ 输入也不套平价门——两条照过（并发上限内）。
        for owner in ("r1#readplan", "r2#readplan"):
            v = self.t.admit_flow(
                owner=owner, flow_class=FLOW_ELASTIC,
                links=[(owner == "r1#readplan") and (0, 1) or (1, 2)],
                port_id=5)
            self.assertTrue(v.admitted)

    def test_concurrency_cap_defers(self):
        for i, owner in enumerate(("r1#readplan", "r2#readplan")):
            self.assertTrue(self.t.admit_flow(
                owner=owner, flow_class=FLOW_ELASTIC,
                links=[(i, i + 1)], port_id=5).admitted)
        v = self.t.admit_flow(
            owner="r3#readplan", flow_class=FLOW_ELASTIC,
            links=[(9, 10)], port_id=5)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)
        self.assertEqual(v.resource_kind, "port")
        self.assertIn("concurrency", v.inapplicable_reason)
        self.assertIn("cap Q_init=2", v.inapplicable_reason)

    def test_elastic_counts_into_u_port_for_later_realtime(self):
        # 弹性流端口入 u_port：占用平价门分母，收紧后续实时流。
        self.assertTrue(self.t.admit_flow(
            owner="r1#readplan", flow_class=FLOW_ELASTIC,
            links=[(0, 1)], port_id=5).admitted)
        # hbm=2.0, r̂=1.5：u_port = 0(外部) + 1(弹性在册) → 1.0 < 1.5 拒。
        v = self.t.admit_flow(
            owner="r2#decode#0", flow_class=FLOW_REALTIME,
            links=[(1, 2)], port_id=5, r_hat_kv_bytes_per_ns=1.5)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)
        self.assertEqual(self.t.port_parity_headroom(
            5, r_hat_kv_bytes_per_ns=1.5), 0)

    def test_elastic_still_gated_by_link_quota(self):
        for i, owner in enumerate(("r1#readplan", "r2#readplan")):
            self.assertTrue(self.t.admit_flow(
                owner=owner, flow_class=FLOW_ELASTIC,
                links=[(0, 1)], port_id=i).admitted)
        v = self.t.admit_flow(
            owner="r3#readplan", flow_class=FLOW_ELASTIC,
            links=[(0, 1)], port_id=9)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_LINK)


class OneshotFlowTest(unittest.TestCase):
    """一次性传输（copy/merge/逐出写回/池恢复）：全额计价、无端口门、
    链路计数入占用；端口在册收紧后续实时流（D4 通用治理同等纳入）。"""

    def setUp(self):
        self.t = _tracker(noc=4.0, hbm=2.0)   # Q_init = 2

    def test_no_port_gate_but_enrolls_into_u_port(self):
        # hbm=2.0, r̂=1.5 下两条在册即拒平价；oneshot 无门、三条照过
        # （并发上限只约束 elastic 类）。
        for i in range(3):
            v = self.t.admit_flow(
                owner=f"victim{i}#writeback", flow_class=FLOW_ONESHOT,
                links=[(i, i + 1)], port_id=5)
            self.assertTrue(v.admitted)
        self.assertEqual(self.t.port_enrolled_by_class(5)[FLOW_ONESHOT], 3)
        # 在册收紧后续实时流：u_port=3 → 2.0/4 = 0.5 < 1.5 拒。
        v = self.t.admit_flow(
            owner="r1#decode#0", flow_class=FLOW_REALTIME,
            links=[(9, 10)], port_id=5, r_hat_kv_bytes_per_ns=1.5)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)

    def test_link_side_counted_into_occupancy(self):
        for owner in ("a#copy", "b#copy"):
            self.assertTrue(self.t.admit_flow(
                owner=owner, flow_class=FLOW_ONESHOT,
                links=[(0, 1)], port_id=0).admitted)
        v = self.t.admit_flow(
            owner="c#copy", flow_class=FLOW_ONESHOT,
            links=[(0, 1)], port_id=0)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_LINK)

    def test_release_flow_restores_credit(self):
        self.assertTrue(self.t.admit_flow(
            owner="a#copy", flow_class=FLOW_ONESHOT,
            links=[(0, 1)], port_id=0).admitted)
        self.assertTrue(self.t.admit_flow(
            owner="b#copy", flow_class=FLOW_ONESHOT,
            links=[(0, 1)], port_id=0).admitted)
        self.assertEqual(self.t.link_remaining((0, 1)), 0)
        self.t.release_flow("a#copy")
        self.assertEqual(self.t.link_remaining((0, 1)), 1)
        self.assertEqual(self.t.port_enrolled(0), 1)

    def test_release_unknown_owner_fails_closed(self):
        with self.assertRaises(LinkQuotaError):
            self.t.release_flow("ghost#copy")

    def test_double_release_fails_closed(self):
        self.assertTrue(self.t.admit_flow(
            owner="a#copy", flow_class=FLOW_ONESHOT,
            links=[(0, 1)], port_id=0).admitted)
        self.t.release_flow("a#copy")
        with self.assertRaises(LinkQuotaError):
            self.t.release_flow("a#copy")


# ====================================================== 预留借还配对 ==


class MergeReservePairingTest(unittest.TestCase):
    """merge 义务预留：双向各 1 槽 + 双候选胜者侧 bulk 名额的借还配对。"""

    def setUp(self):
        self.t = _tracker(noc=4.0, hbm=2.0)   # Q_init = N_bulk = 2
        self.assertEqual(self.t.n_bulk, 2)

    def test_reserve_then_full_release_cycle(self):
        v = self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1), (1, 2)],
            links_reverse=[(2, 1), (1, 0)],
            port_forward=7, port_reverse=3)
        self.assertTrue(v.admitted)
        # 预留计入链路信用余量（非 occupancy），端口占 bulk 名额。
        self.assertEqual(self.t.link_reserved((0, 1)), 1)
        self.assertEqual(self.t.link_reserved((1, 2)), 1)
        self.assertEqual(self.t.link_reserved((2, 1)), 1)
        self.assertEqual(self.t.link_remaining((0, 1)), 1)
        self.assertEqual(self.t.bulk_used(7), 1)
        self.assertEqual(self.t.bulk_used(3), 1)
        # service_done 方向裁决：败者侧（reverse）立即释放。
        self.t.adjudicate_merge_direction("rid-1", MERGE_DIRECTION_FORWARD)
        self.assertEqual(self.t.link_reserved((2, 1)), 0)
        self.assertEqual(self.t.bulk_used(3), 0)
        self.assertEqual(self.t.link_reserved((0, 1)), 1)  # 胜者侧仍持有
        self.assertEqual(self.t.bulk_used(7), 1)
        # merge_done（_on_merge_done）：胜者侧释放，全部归零。
        self.t.release_merge("rid-1")
        self.assertEqual(self.t.link_reserved((0, 1)), 0)
        self.assertEqual(self.t.bulk_used(7), 0)
        self.assertEqual(self.t.link_remaining((0, 1)), 2)

    def test_unadjudicated_release_rolls_back_both_sides(self):
        # 未裁决即整体释放 = 双侧撤销（remote 轮取消路径）。
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=3).admitted)
        self.t.release_merge("rid-1")
        self.assertEqual(self.t.link_reserved((0, 1)), 0)
        self.assertEqual(self.t.bulk_used(7) + self.t.bulk_used(3), 0)

    def test_bulk_cap_defers_with_quota_port(self):
        # N_bulk = Q_init = 2：两个 rid 各占端口 7 的双候选名额后，第三
        # 个预留 quota_deferred（wait_reason=quota_port）。
        for rid in ("rid-1", "rid-2"):
            self.assertTrue(self.t.reserve_merge(
                rid, links_forward=[(0, 1)], links_reverse=[(1, 0)],
                port_forward=7, port_reverse=8).admitted)
        v = self.t.reserve_merge(
            "rid-3", links_forward=[(2, 3)], links_reverse=[(3, 2)],
            port_forward=7, port_reverse=9)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_PORT)
        self.assertEqual(v.resource_kind, "port")
        self.assertIn("N_bulk=2", v.inapplicable_reason)
        # 原子性：失败的预留零登记。
        self.assertEqual(self.t.link_reserved((2, 3)), 0)
        self.assertEqual(self.t.bulk_used(9), 0)

    def test_shared_link_in_both_directions_counts_both_slots(self):
        # 同一链路同时位于前向与反向路径 → 双向各 1 槽 = 计 2。
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0), (0, 1)],
            port_forward=7, port_reverse=3).admitted)
        self.assertEqual(self.t.link_reserved((0, 1)), 2)
        self.assertEqual(self.t.link_remaining((0, 1)), 0)

    def test_reserve_deferred_when_link_credit_exhausted(self):
        for owner in ("a#decode#0", "b#decode#0"):
            self.assertTrue(self.t.admit_flow(
                owner=owner, flow_class=FLOW_REALTIME,
                links=[(0, 1)], port_id=0,
                r_hat_kv_bytes_per_ns=0.1).admitted)
        v = self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=3)
        self.assertFalse(v.admitted)
        self.assertEqual(v.wait_reason, WAIT_QUOTA_LINK)
        # 释放一条实时流（流 settle bump 代数）后预留可行。
        self.t.release_flow("a#decode#0")
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=3).admitted)

    def test_duplicate_reserve_fails_closed(self):
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], port_forward=7).admitted)
        with self.assertRaises(LinkQuotaError):
            self.t.reserve_merge("rid-1", links_forward=[(2, 3)])

    def test_double_adjudication_and_unknown_fails_closed(self):
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=3).admitted)
        self.t.adjudicate_merge_direction("rid-1", MERGE_DIRECTION_FORWARD)
        with self.assertRaises(LinkQuotaError):
            self.t.adjudicate_merge_direction(
                "rid-1", MERGE_DIRECTION_REVERSE)
        with self.assertRaises(LinkQuotaError):
            self.t.adjudicate_merge_direction("ghost", MERGE_DIRECTION_FORWARD)
        with self.assertRaises(LinkQuotaError):
            self.t.adjudicate_merge_direction("rid-1", "sideways")
        # 全释放后再释放 = 双释放 fail-closed。
        self.t.release_merge("rid-1")
        with self.assertRaises(LinkQuotaError):
            self.t.release_merge("rid-1")

    def test_reserve_requires_some_resource(self):
        with self.assertRaises(LinkQuotaError):
            self.t.reserve_merge("rid-1")

    def test_owner_string_is_frozen_template(self):
        self.assertEqual(
            MERGE_RESERVE_OWNER_TEMPLATE.format(rid="r42"),
            "r42#merge-reserve")


# ============================================== 借槽账本守恒断言（O10②） ==


class MergeBorrowLedgerInvariantTest(unittest.TestCase):
    """O10②（2026-09-23 终轮深挖）：N1 同事务借槽账本的守恒不变式
    （borrow ≤ rid 名下该链路占用）在创建/转移后 O(1) 增量断言、sweep
    前残留 fail-closed——合法序列零触发，篡改（变异式）触发对应 raise。
    断言失败 = RuntimeError fail-closed（与既有 pairing 失败同风格的
    硬 raise，账本脱钩不静默）；O14：_release_merge_side 的 desync
    raise（原 pragma: no cover）注入测试。"""

    def _tracker_with_rid_flow(self):
        t = _tracker(noc=4.0, hbm=2.0)          # Q_init = 2
        self.assertTrue(t.admit_flow(
            owner="rid-1#readplan", flow_class=FLOW_ELASTIC,
            links=[(0, 1)], port_id=5, now_ns=0).admitted)
        return t

    def test_full_borrow_cycle_does_not_raise(self):
        # 正常 N1 全序列：创建 → 转移 → 消解（裁决败者侧）→ sweep——
        # 全程无 raise，账本清空、预留归零（守恒断言对合法输入零触发）。
        t = self._tracker_with_rid_flow()
        self.assertTrue(t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8).admitted)
        # 创建断言位：borrow[(0,1)] = min(need=1, occ=1) = 1 ≤ occ。
        self.assertEqual(t._merge_borrow["rid-1"], {(0, 1): 1})
        t.release_flow("rid-1#readplan", now_ns=100)   # 转移 borrow→res
        # 转移断言位：借槽清空（0 ≤ 剩余占用 0），预留接管槽位。
        self.assertEqual(t._merge_borrow, {})
        self.assertEqual(t.link_reserved((0, 1)), 1)
        t.adjudicate_merge_direction("rid-1", MERGE_DIRECTION_REVERSE)
        self.assertEqual(t.link_reserved((0, 1)), 0)   # 败者侧归还
        t.release_merge("rid-1")                       # 胜者侧 + sweep
        self.assertEqual(t._merge_borrow, {})
        self.assertEqual(
            t.link_reserved((0, 1)) + t.link_reserved((1, 0)), 0)
        self.assertEqual(t.bulk_used(7) + t.bulk_used(8), 0)

    def test_tampered_borrow_raises_at_transfer(self):
        # 变异：篡改 borrow > rid 占用，流 settle 触发转移断言——
        # transfer=min 后 borrow=4 > occ=0 ⇒ RuntimeError（fail-closed）。
        t = self._tracker_with_rid_flow()
        t.reserve_merge("rid-1", links_forward=[(0, 1)], port_forward=7)
        t._merge_borrow["rid-1"][(0, 1)] = 5           # 借槽 1 → 5（篡改）
        with self.assertRaisesRegex(
                RuntimeError, r"borrowed=4 > rid occupancy=0"):
            t.release_flow("rid-1#readplan", now_ns=100)

    def test_tampered_borrow_raises_at_creation(self):
        # 变异：上一笔预留 sweep 后篡改造残留挂账，再次预留时合并借槽
        # = 残留 1 + 新借 1 = 2 > occ 1 ⇒ 创建断言 RuntimeError。
        t = self._tracker_with_rid_flow()
        t.reserve_merge("rid-1", links_forward=[(0, 1)], port_forward=7)
        t.release_merge("rid-1")                       # sweep 正常清空
        self.assertEqual(t._merge_borrow, {})
        t._merge_borrow["rid-1"] = {(0, 1): 1}         # 模拟未来编辑残留
        with self.assertRaisesRegex(
                RuntimeError, r"borrowed=2 > rid occupancy=1"):
            t.reserve_merge("rid-1", links_forward=[(0, 1)])

    def test_sweep_residue_raises_runtime_error(self):
        # 变异：消解路径只消解 1 槽（篡改造 borrow 虚增 1），sweep 前
        # 残留非空 ⇒ RuntimeError（替代 N1 静默弃置——脱钩不静默）。
        t = self._tracker_with_rid_flow()
        t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8)
        t._merge_borrow["rid-1"][(0, 1)] += 1          # borrow 1 → 2
        with self.assertRaisesRegex(
                RuntimeError,
                r"desynchronized at release_merge.*residue=\{\(0, 1\): 1\}"):
            t.release_merge("rid-1")

    def test_release_merge_side_desync_raise(self):
        # O14：_release_merge_side 的账本脱钩 raise 注入——借槽被篡改
        # 删除后，预留释放时 res 无登记且 borrow 无条目 ⇒ LinkQuotaError
        # （fail-closed；本测试覆盖后该 raise 的 pragma: no cover 已除）。
        t = self._tracker_with_rid_flow()
        t.reserve_merge("rid-1", links_forward=[(0, 1)], port_forward=7)
        # 借槽链路由本 rid 读流 occupancy 承载（res 无 merge-owner 登记）：
        self.assertEqual(t.link_reserved((0, 1)), 0)
        del t._merge_borrow["rid-1"][(0, 1)]           # 最小篡制造脱钩
        with self.assertRaisesRegex(
                LinkQuotaError,
                r"merge reservation ledger desynchronized at \(0, 1\) "
                r"for 'rid-1#merge-reserve'"):
            t.release_merge("rid-1")


# ====================================================== deferred 重试键 ==


class DeferredRetryKeyTest(unittest.TestCase):
    """配额代数（信用占用代数）：流 settle/预留释放 bump 键；C11 经
    extend_retry_key 并入 SH:1775-1802 的 epoch 键重试门。"""

    def setUp(self):
        self.t = _tracker(noc=4.0, hbm=2.0)

    def test_epoch_stable_without_release(self):
        before = self.t.quota_retry_key()
        self.assertTrue(self.t.admit_flow(
            owner="a#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0,
            r_hat_kv_bytes_per_ns=0.1).admitted)
        # 准入本身不 bump（键变化须对应"信用可得性变化"= 释放侧事件）。
        self.assertEqual(self.t.quota_retry_key(), before)

    def test_flow_settle_bumps_epoch(self):
        self.assertTrue(self.t.admit_flow(
            owner="a#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0,
            r_hat_kv_bytes_per_ns=0.1).admitted)
        before = self.t.quota_retry_key()
        self.t.release_flow("a#decode#0")
        self.assertNotEqual(self.t.quota_retry_key(), before)

    def test_adjudication_and_merge_release_bump_epoch(self):
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=3).admitted)
        before = self.t.quota_retry_key()
        self.t.adjudicate_merge_direction("rid-1", MERGE_DIRECTION_FORWARD)
        mid = self.t.quota_retry_key()
        self.assertNotEqual(mid, before)
        self.t.release_merge("rid-1")
        self.assertNotEqual(self.t.quota_retry_key(), mid)

    def test_extend_retry_key_composition(self):
        base = (3, ((0, 17), (2, 5)))   # SH _current_retry_key 形态
        composed = extend_retry_key(base, (9,))
        self.assertEqual(composed, (3, ((0, 17), (2, 5)), 9))
        self.assertEqual(base, (3, ((0, 17), (2, 5))))  # base 不可变约定
        self.assertEqual(extend_retry_key((), self.t.quota_retry_key()),
                         self.t.quota_retry_key())

    def test_quota_adjust_bumps_epoch_in_aimd(self):
        t = _tracker(mode=QUOTA_AIMD, noc=4.0, hbm=2.0)
        before = t.quota_retry_key()
        t.set_link_quota((0, 1), 1)
        self.assertNotEqual(t.quota_retry_key(), before)


# ======================================================= δ_adm 翻转带 ==


class DeltaAdmFlipBandTest(unittest.TestCase):
    """δ_adm 近平局带：挑战动作须胜在册动作族 >= δ_adm 才翻转（D5）。"""

    def test_initial_value_frozen_at_zero(self):
        self.assertEqual(DELTA_ADM_INITIAL_NS, 0)

    def test_band_requires_margin_at_least_delta(self):
        # K8（2026-09-23）：原 546/548 两行同断言重复（"边界含等号"注释
        # 挂在重复行上）——改为边界等号 + 边界内/外三态实钉。
        self.assertTrue(challenger_flips(90, 100, delta_adm_ns=10))   # margin=10=δ 含等号
        self.assertFalse(challenger_flips(91, 100, delta_adm_ns=10))  # margin=9<δ
        self.assertTrue(challenger_flips(89, 100, delta_adm_ns=10))   # margin=11>δ
        self.assertTrue(challenger_flips(0, 100, delta_adm_ns=100))   # margin=100

    def test_zero_delta_is_open_gate_composing_with_argmin(self):
        # δ=0（冻结初值）时谓词恒开（margin >= 0）；与既有
        # (cost_ns, order_key) argmin 比较组合后零行为变更——平局由
        # order_key 裁决，不借初值夹带翻转语义。
        # F4（复审修复）：原期望式 ``incumbent - challenger >= 0`` 与
        # challenger_flips 实现裁决式同源（恒真恒等，零效）——改为写死
        # 查表，期望值逐对手工推导：语义 = 挑战者不贵于在册
        # （challenger <= incumbent）才允许翻转。
        # (challenger, incumbent, expected_flip)
        # 90 < 100：挑战者更便宜 → 翻转 → True
        # 100 == 100：平局边界（δ=0 含等号；平局终裁归 argmin 的
        #   order_key）→ True
        # 110 > 100：挑战者更贵 → 拒翻转 → False
        # 0 == 0：零成本平局边界 → True
        # 100 > 90：更贵（反向）→ False
        # 0 < 100：全幅胜出 → True
        table = ((90, 100, True), (100, 100, True), (110, 100, False),
                 (0, 0, True), (100, 90, False), (0, 100, True))
        for challenger, incumbent, expected in table:
            self.assertIs(
                challenger_flips(challenger, incumbent, 0), expected,
                (challenger, incumbent))

    def test_composition_with_argmin_at_zero_delta_matches_plain_argmin(self):
        candidates = [("stay", 100), ("remote-read", 90), ("copy", 100),
                      ("recompute", 120)]
        order = {name: i for i, (name, _) in enumerate(candidates)}
        plain = min(candidates, key=lambda c: (c[1], order[c[0]]))
        incumbent = plain
        for name, cost in candidates:
            standard_prefers = (cost, order[name]) < (
                incumbent[1], order[incumbent[0]])
            flips = standard_prefers and challenger_flips(
                cost, incumbent[1], DELTA_ADM_INITIAL_NS)
            if flips:
                incumbent = (name, cost)
        self.assertEqual(incumbent, plain)   # 组合裁决 ≡ 纯 argmin

    def test_nonzero_band_keeps_incumbent_on_near_tie(self):
        # δ=25：挑战者只胜 10 不足翻转；胜 30 翻转。
        self.assertFalse(challenger_flips(90, 100, delta_adm_ns=25))
        self.assertTrue(challenger_flips(70, 100, delta_adm_ns=25))

    def test_negative_delta_fails_closed(self):
        with self.assertRaises(LinkQuotaError):
            challenger_flips(90, 100, delta_adm_ns=-1)

    def test_tracker_carries_shared_switch_value(self):
        # 位于共享决策路径、内部对照臂同开关：单一 δ_adm 值经 tracker
        # 构造参数共享（主臂/对照臂同值）。
        t = _tracker(delta_adm_ns=25)
        self.assertEqual(t.delta_adm_ns, 25)
        self.assertEqual(_tracker().delta_adm_ns, DELTA_ADM_INITIAL_NS)


# ================================================== quota_deferred 回队 ==


class QuotaDeferredRequeueTest(unittest.TestCase):
    """全部动作不可行 → 回 pending_admissions，严禁零候选 fail-closed
    （joint_scheduler.py:156-159 的 JointSchedulerError 会终局 run）。"""

    def _exhausted_verdicts(self):
        t = _tracker(noc=4.0, hbm=2.0)
        for owner in ("a#decode#0", "b#decode#0"):
            t.admit_flow(owner=owner, flow_class=FLOW_REALTIME,
                         links=[(0, 1)], port_id=0,
                         r_hat_kv_bytes_per_ns=0.1)
        verdicts = {}
        for action, links in (("remote-read", [(0, 1)]),
                              ("copy", [(0, 1)])):
            verdicts[action] = t.admit_flow(
                owner=f"r#{action}", flow_class=FLOW_ONESHOT, links=links,
                port_id=0)
            # F4（复审修复）：helper 内裸 assert 在 pytest -O 下会被剥除
            # ——改 self.assertFalse（本方法已是 TestCase 成员，self 在
            # 场）。
            self.assertFalse(
                verdicts[action].admitted,
                f"{action} must be quota-blocked in exhausted fixture")
        return t, verdicts

    def test_all_infeasible_requeues_without_joint_scheduler_error(self):
        t, verdicts = self._exhausted_verdicts()
        # 核心断言：回队路径不触发 JointSchedulerError（也不抛任何
        # 异常）——正确语义 = 请求回 pending_admissions 带扩展重试键。
        try:
            deferred = quota_deferred_requeue(
                "req-1", verdicts, t.quota_retry_key(),
                base_retry_key=(3, ((0, 17),)))
        except JointSchedulerError:
            self.fail("quota_deferred must requeue, not fail-closed")
        except Exception:  # noqa: BLE001 - 永不 raise 合同
            self.fail("quota_deferred_requeue must never raise")
        self.assertTrue(deferred.requeue)
        self.assertEqual(deferred.wait_reason, WAIT_QUOTA_LINK)
        self.assertEqual(
            deferred.retry_key, (3, ((0, 17),), t.quota_retry_key()[0]))
        for action in ("remote-read", "copy"):
            self.assertIn("quota_link", 
                          deferred.inapplicable_reasons[action])

    def test_wait_reason_reports_port_when_port_scarse(self):
        t = _tracker(noc=4.0, hbm=2.0)
        for i in range(2):
            t.admit_flow(owner=f"e{i}#readplan", flow_class=FLOW_ELASTIC,
                         links=[(i, i + 1)], port_id=5)
        verdict = t.admit_flow(
            owner="r#decode#0", flow_class=FLOW_ELASTIC,
            links=[(9, 10)], port_id=5)
        self.assertFalse(verdict.admitted)
        deferred = quota_deferred_requeue(
            "req-2", {"remote-read": verdict}, t.quota_retry_key())
        self.assertEqual(deferred.wait_reason, WAIT_QUOTA_PORT)

    def test_release_reopens_admission_and_bumps_key(self):
        # 释放后仍无动作可行才终端 fail-closed 披露；此处释放后可再准入
        # ——重试门因配额代数 bump 而重开。
        t, verdicts = self._exhausted_verdicts()
        deferred_before = quota_deferred_requeue(
            "req-1", verdicts, t.quota_retry_key())
        t.release_flow("a#decode#0")
        self.assertNotEqual(t.quota_retry_key(), deferred_before.retry_key[-1:])
        v = t.admit_flow(owner="r#copy", flow_class=FLOW_ONESHOT,
                         links=[(0, 1)], port_id=0)
        self.assertTrue(v.admitted)

    def test_admitted_verdict_input_fails_closed(self):
        t = _tracker(noc=4.0, hbm=2.0)
        admitted = t.admit_flow(
            owner="a#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0, r_hat_kv_bytes_per_ns=0.1)
        with self.assertRaises(LinkQuotaError):
            quota_deferred_requeue(
                "req-1", {"remote-read": admitted}, t.quota_retry_key())

    def test_empty_verdicts_defensive_requeue(self):
        t = _tracker()
        deferred = quota_deferred_requeue("req-1", {}, t.quota_retry_key())
        self.assertTrue(deferred.requeue)
        self.assertEqual(deferred.wait_reason, WAIT_CAPACITY)


# ============================================================== 模式 ==


class QuotaModeTest(unittest.TestCase):
    """off/static/aimd 三模式的门旁路与 AIMD 调整原语。"""

    def test_off_bypasses_gates_but_keeps_pairing(self):
        t = _tracker(mode=QUOTA_OFF, noc=4.0, hbm=2.0)
        for i in range(5):   # 远超 Q=2：off 模式全部放行
            v = t.admit_flow(
                owner=f"a{i}#decode#0", flow_class=FLOW_REALTIME,
                links=[(0, 1)], port_id=0,
                r_hat_kv_bytes_per_ns=1.9)
            self.assertTrue(v.admitted)
        # 簿记照记：释放配对语义与模式无关。
        self.assertEqual(t.link_occupancy((0, 1)), 5)
        t.release_flow("a0#decode#0")
        self.assertEqual(t.link_occupancy((0, 1)), 4)

    def test_static_rejects_quota_adjustment(self):
        t = _tracker(mode=QUOTA_STATIC, noc=4.0, hbm=2.0)
        with self.assertRaises(LinkQuotaError):
            t.set_link_quota((0, 1), 1)

    def test_aimd_adjusts_with_structural_floor(self):
        t = _tracker(mode=QUOTA_AIMD, noc=4.0, hbm=2.0)
        t.set_link_quota((0, 1), 1)
        self.assertEqual(t.link_quota((0, 1)), 1)
        t.set_link_quota((0, 1), 0)     # 下限 1 = 防结构性死锁（F2 同源）
        self.assertEqual(t.link_quota((0, 1)), 1)
        self.assertEqual(t.link_quota((9, 10)), 2)  # 未调整链路 = Q_init
        # N_bulk 与端口判据不随链路 Q 调整（冻结）。
        self.assertEqual(t.n_bulk, 2)

    def test_aimd_shrink_grandfathers_occupancy(self):
        # 收缩低于在册占用：不逐出（grandfathered），余量为负期间封新
        # 准入，释放排空后恢复。
        t = _tracker(mode=QUOTA_AIMD, noc=4.0, hbm=2.0)
        for owner in ("a#decode#0", "b#decode#0"):
            self.assertTrue(t.admit_flow(
                owner=owner, flow_class=FLOW_REALTIME,
                links=[(0, 1)], port_id=0,
                r_hat_kv_bytes_per_ns=0.1).admitted)
        t.set_link_quota((0, 1), 1)
        self.assertEqual(t.link_occupancy((0, 1)), 2)   # 不逐出
        self.assertEqual(t.link_remaining((0, 1)), -1)
        v = t.admit_flow(owner="c#decode#0", flow_class=FLOW_REALTIME,
                         links=[(0, 1)], port_id=0,
                         r_hat_kv_bytes_per_ns=0.1)
        self.assertFalse(v.admitted)
        t.release_flow("a#decode#0")
        self.assertEqual(t.link_remaining((0, 1)), 0)
        self.assertFalse(t.admit_flow(
            owner="c#decode#0", flow_class=FLOW_REALTIME,
            links=[(0, 1)], port_id=0,
            r_hat_kv_bytes_per_ns=0.1).admitted)

    def test_snapshot_discloses_link_and_port_state(self):
        t = _tracker(noc=4.0, hbm=2.0)
        t.admit_flow(owner="a#decode#0", flow_class=FLOW_REALTIME,
                     links=[(0, 1)], port_id=5,
                     r_hat_kv_bytes_per_ns=0.5)
        t.reserve_merge("rid-1", links_forward=[(0, 1)], port_forward=5)
        snap = t.snapshot()
        link = snap["links"]["(0, 1)"]
        self.assertEqual(link["occupancy"], 1)
        self.assertEqual(link["reserved"], 1)
        self.assertEqual(link["remaining"], 0)
        port = snap["ports"]["5"]
        self.assertEqual(port["enrolled_by_class"][FLOW_REALTIME], 1)
        self.assertEqual(port["bulk_used"], 1)
        self.assertEqual(port["n_bulk"], 2)


# ======================================== AIMD 扩张冻结窗口（O6②） ==


class AimdExpansionFreezeTest(unittest.TestCase):
    """O6②：observe_telemetry 的 allow_expansion 冻结扩张参数。

    语义钉（docstring 合同）：False = 冻结 additive-increase 与
    expand_capped，并清除此前的舒适 streak；冻结区间不计入解冻后的
    新 streak。冻结期间收缩/保持/comfort/shrink 判定、EWMA、簿记照常；
    True = 现状零行为变更（既有全部测试不改动即为回归锚）。
    """

    def _seeded_tracker(self):
        # noc=240, hbm=100 → Q_init=2；r_KV=10 → 舒适阈 12、ceiling 24。
        # 旁链路喂一个寿命样本 1000 → EWMA=1000 → T_expand=10000。
        t = _tracker(mode=QUOTA_AIMD, noc=240.0, hbm=100.0)
        self.assertEqual(t.q_init, 2)
        t.admit_flow(owner="svc#decode#0", flow_class=FLOW_REALTIME,
                     links=[(9, 10)], port_id=6,
                     r_hat_kv_bytes_per_ns=10.0, now_ns=0)
        t.release_flow("svc#decode#0", now_ns=1000)
        self.assertEqual(t.t_expand_ns, 10000.0)
        return t

    def test_frozen_window_comfort_accumulates_nothing(self):
        t = self._seeded_tracker()
        # (now, dt_ns)：首样本 dt=0 基线，其后每 5000——第三样本非冻结
        # 下必达 T_expand=10000 触发扩张，冻结语义钉其不动。
        samples = ((2000, 0), (7000, 5000), (12000, 5000))
        for now, dt_ns in samples:
            obs = t.observe_telemetry(
                now, {(0, 1): 240.0}, 10.0, allow_expansion=False)
            record = obs["links"]["(0, 1)"]
            # comfort 判定照常（信号不冻），只冻扩张动作：
            self.assertEqual(record["signal"], AIMD_SIGNAL_COMFORT)
            self.assertEqual(record["action"], AIMD_ACTION_COMFORT_FROZEN)
            self.assertEqual(record["dt_ns"], dt_ns)
            self.assertEqual(obs["allow_expansion"], False)
        # 全程 quiet 不进、Q 不动：
        self.assertEqual(t.link_quota((0, 1)), t.q_init)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)

    def test_unfrozen_baseline_expands_on_same_inputs(self):
        # 回归锚：同输入 + allow_expansion=True（= 缺省现状）正常扩张——
        # streak 0 → 5000 → 10000 ≥ T_expand，第三样本 expand，Q 2 → 3。
        t = self._seeded_tracker()
        obs = t.observe_telemetry(2000, {(0, 1): 240.0}, 10.0,
                                  allow_expansion=True)
        self.assertEqual(obs["links"]["(0, 1)"]["action"],
                         AIMD_ACTION_COMFORT)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)  # dt=0
        t.observe_telemetry(7000, {(0, 1): 240.0}, 10.0)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 5000)
        obs = t.observe_telemetry(12000, {(0, 1): 240.0}, 10.0)
        record = obs["links"]["(0, 1)"]
        self.assertEqual(record["action"], AIMD_ACTION_EXPAND)
        self.assertEqual(record["quota_before"], 2)
        self.assertEqual(record["quota_after"], 3)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)  # 清零

    def test_frozen_window_still_shrinks_on_shrink_signal(self):
        # 冻结只冻扩张：超阈（shrink 判定）照常执行——Q 减半、streak 清零。
        t = self._seeded_tracker()
        t.observe_telemetry(2000, {(0, 1): 240.0}, 10.0,
                            allow_expansion=False)
        self.assertEqual(t.link_quota((0, 1)), 2)
        obs = t.observe_telemetry(3000, {(0, 1): 8.0}, 10.0,
                                  allow_expansion=False)
        record = obs["links"]["(0, 1)"]
        self.assertEqual(record["signal"], AIMD_SIGNAL_SHRINK)
        self.assertEqual(record["action"], AIMD_ACTION_SHRINK)
        self.assertEqual(record["quota_after"], 1)
        self.assertEqual(t.link_quota((0, 1)), 1)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)

    def test_resume_after_freeze_requires_fresh_streak(self):
        # 冻结前已累计 5000/10000；空 telemetry 冻结也清 streak，解冻首
        # 样本不把冻结区间 dt 计为新进度，之后从零重新连续累计。
        t = self._seeded_tracker()
        t.observe_telemetry(2000, {(0, 1): 240.0}, 10.0)
        t.observe_telemetry(7000, {(0, 1): 240.0}, 10.0)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 5000)

        frozen = t.observe_telemetry(
            12000, {}, 10.0, allow_expansion=False)
        self.assertEqual(frozen["links"], {})
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 0)
        self.assertEqual(t.link_quota((0, 1)), 2)
        obs = t.observe_telemetry(17000, {(0, 1): 240.0}, 10.0,
                                  allow_expansion=True)
        record = obs["links"]["(0, 1)"]
        self.assertEqual(record["action"], AIMD_ACTION_COMFORT)  # 非 expand
        self.assertFalse(record["contiguous"])
        self.assertEqual(record["quiet_ns"], 0)                  # 不计冻结 dt
        self.assertEqual(t.link_quota((0, 1)), 2)
        obs = t.observe_telemetry(22000, {(0, 1): 240.0}, 10.0)
        self.assertEqual(obs["links"]["(0, 1)"]["quiet_ns"], 5000)
        obs = t.observe_telemetry(27000, {(0, 1): 240.0}, 10.0)
        record = obs["links"]["(0, 1)"]
        self.assertEqual(record["action"], AIMD_ACTION_EXPAND)   # 恢复扩张
        self.assertEqual(record["quota_after"], 3)

    def test_adjacent_unfreeze_does_not_count_frozen_interval(self):
        t = self._seeded_tracker()
        t.observe_telemetry(2000, {(0, 1): 240.0}, 10.0)
        t.observe_telemetry(7000, {(0, 1): 240.0}, 10.0)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 5000)

        frozen = t.observe_telemetry(
            12000, {(0, 1): 240.0}, 10.0, allow_expansion=False)
        self.assertEqual(frozen["links"]["(0, 1)"]["quiet_ns"], 0)
        resumed = t.observe_telemetry(
            17000, {(0, 1): 240.0}, 10.0, allow_expansion=True)
        record = resumed["links"]["(0, 1)"]
        self.assertTrue(record["contiguous"])
        self.assertEqual(record["dt_ns"], 5000)
        self.assertEqual(record["action"], AIMD_ACTION_COMFORT)
        self.assertEqual(record["quiet_ns"], 0)
        self.assertEqual(t.link_quota((0, 1)), 2)

        t.observe_telemetry(22000, {(0, 1): 240.0}, 10.0)
        self.assertEqual(t.aimd_link_state((0, 1))["quiet_ns"], 5000)
        resumed = t.observe_telemetry(27000, {(0, 1): 240.0}, 10.0)
        self.assertEqual(
            resumed["links"]["(0, 1)"]["action"], AIMD_ACTION_EXPAND)

    def test_allow_expansion_type_fail_closed(self):
        t = self._seeded_tracker()
        with self.assertRaises(LinkQuotaError):
            t.observe_telemetry(2000, {(0, 1): 240.0}, 10.0,
                                allow_expansion="no")


# ============================================================ 域（出清） ==


class ReachableDomainTest(unittest.TestCase):
    """域(t) = 经由仍有剩余信用的链路与端口可达的实例集合——出清结果
    非输入，不先验定义"链路区域"。"""

    def _adjacency(self):
        # 0 <-> 1 <-> 2 <-> 3 线形拓扑。
        return {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

    def test_domain_spans_credit_bearing_links(self):
        domain = reachable_domain(
            0, self._adjacency(), link_has_credit=lambda edge: True)
        self.assertEqual(domain, frozenset({0, 1, 2, 3}))

    def test_exhausted_link_cuts_domain(self):
        # 链路 (1,2) 信用耗尽 → 域从 {0..3} 收缩为 {0,1}：域随出清结果
        # 变化（ admissions 改变可达集），不是先验输入。
        domain = reachable_domain(
            0, self._adjacency(), link_has_credit=lambda edge: edge != (1, 2))
        self.assertEqual(domain, frozenset({0, 1}))

    def test_port_admits_gates_instance_entry(self):
        domain = reachable_domain(
            0, self._adjacency(), link_has_credit=lambda edge: True,
            port_admits=lambda node: node != 3)
        self.assertEqual(domain, frozenset({0, 1, 2}))

    def test_domain_tracks_quota_clearing_state(self):
        t = _tracker(noc=4.0, hbm=2.0)
        edges = [(0, 1), (1, 2)]
        full_domain = reachable_domain(
            0, {0: [1], 1: [2]},
            link_has_credit=lambda edge: t.link_remaining(edge) > 0)
        self.assertEqual(full_domain, frozenset({0, 1, 2}))
        for owner in ("a#decode#0", "b#decode#0"):
            t.admit_flow(owner=owner, flow_class=FLOW_REALTIME,
                         links=[(0, 1)], port_id=0,
                         r_hat_kv_bytes_per_ns=0.1)
        shrunk = reachable_domain(
            0, {0: [1], 1: [2]},
            link_has_credit=lambda edge: t.link_remaining(edge) > 0)
        self.assertEqual(shrunk, frozenset({0}))
        t.release_flow("a#decode#0")
        recovered = reachable_domain(
            0, {0: [1], 1: [2]},
            link_has_credit=lambda edge: t.link_remaining(edge) > 0)
        self.assertEqual(recovered, frozenset({0, 1, 2}))


# ================================================== JOINT_QUOTA_MODE 解析 ==


class JointConfigQuotaModeTest(unittest.TestCase):
    """开关解析：off/static/aimd 合法、非法 fail-closed、缺省 off（F7）、
    manifest 含键、与 combo 正交（仿 remote_actions 独立解析模式）。"""

    def test_default_is_off(self):
        config = parse_joint_config({})
        self.assertEqual(config.quota_mode, "off")
        self.assertFalse(config.quota_enabled)

    def test_all_modes_parse(self):
        for mode in QUOTA_MODES:
            config = parse_joint_config({"JOINT_QUOTA_MODE": mode})
            self.assertEqual(config.quota_mode, mode)
            self.assertEqual(config.quota_enabled, mode != "off")

    def test_invalid_values_fail_closed(self):
        for raw in ("dynamic", "OFF", "Static", " off", "off ", "",
                    "aimd ", "1"):
            with self.assertRaises(JointConfigError, msg=repr(raw)):
                parse_joint_config({"JOINT_QUOTA_MODE": raw})

    def test_direct_construction_validates(self):
        from joint.joint_config import JointMechanismConfig
        with self.assertRaises(JointConfigError):
            JointMechanismConfig(
                category_mode="typed", scheduler_mode="joint",
                layer_policy="adaptive", remote_actions="on",
                quota_mode="sometimes")

    def test_manifest_carries_quota_mode(self):
        for env in ({}, {"JOINT_QUOTA_MODE": "static"},
                    {"JOINT_ABLATION_COMBO": "none"}):
            manifest = parse_joint_config(env).manifest_dict()
            self.assertIn("quota_mode", manifest)
            self.assertIn("quota_enabled", manifest)
        manifest = parse_joint_config(
            {"JOINT_QUOTA_MODE": "aimd"}).manifest_dict()
        self.assertEqual(manifest["quota_mode"], "aimd")
        self.assertTrue(manifest["quota_enabled"])
        self.assertEqual(
            manifest["source_env"]["JOINT_QUOTA_MODE"], "aimd")

    def test_default_not_recorded_in_source_env(self):
        manifest = parse_joint_config({}).manifest_dict()
        self.assertNotIn("JOINT_QUOTA_MODE", manifest["source_env"])

    def test_orthogonal_to_combo_and_mechanism_switches(self):
        # 配额与三机制/remote 正交：可与 combo 预设及其余开关共存，
        # 不触发预设-显式互斥（互斥只覆盖三机制开关）。
        config = parse_joint_config({
            "JOINT_ABLATION_COMBO": "TJE",
            "JOINT_QUOTA_MODE": "static",
            "JOINT_REMOTE_ACTIONS": "off",
        })
        self.assertEqual(config.combo, "TJE")
        self.assertEqual(config.quota_mode, "static")
        self.assertFalse(config.remote_enabled)
        # 缺省三机制语义不受配额开关影响。
        self.assertEqual(
            (config.category_mode, config.scheduler_mode,
             config.layer_policy), ("typed", "joint", "adaptive"))


if __name__ == "__main__":
    unittest.main()
