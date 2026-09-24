#!/usr/bin/env python3
"""test_joint_telemetry_divisor.py -- C7（WP2，仅 JCM 半）遥测族 +
N2/M1/N3 复核口径 + O 批补漏（O7③/O10③）的零后端单测。

覆盖（N2/M1/N3 现行口径）：
  1. divisor_effective 数值断言（N2 叠加口径）：旧流并发两口径
     （注册 flows / 遥测换算或 M1 流数）取大者为 base、候选自身
     self_overlap 恒叠加；C7 max 而非相加去重保留（§4.1——同一流量
     不重复计数）：注册表漏计时实测抬升（floor > registered → 取
     floor）、多计时不放大（registered > floor → 取 registered）；
  2. 空窗口/未开遥测 → collective_coverage 条件不满足（telemetry_
     enabled_whole_run=False / 窗口序列空洞 / 空序列三分支 + 全闲
     epoch 零长窗合法锚）；
  3. transfer_factor EWMA 更新与样本纪律：首样本直接初始化、α 时间
     衰减公式复算、同 tick 多链路 Σactual/Σbase 单条更新、零字节/
     零活跃样本拒绝计数、名义速率非法 fail-closed；
  4. 冻结交接接口（C8 联合冻结 kwarg 名）：``link_telemetry_rates =
    {link_id: 实测有效速率(B/ns)}`` 传入 JointCostModel 构造——None/
    空字典 = 零漂移（与无字段构造同值），遥测在链路上抬升 remote 计
    价、breakdown 披露位保守取整；整型 LinkId 键端点无关消费（速率
    查询/披露计数，不参与端点合并），畸形键 fail-closed；
  5. M1 流数口径与混合形态（O7③，A19'(g)③）：``link_flow_counts``
     在场取时间加权活跃流数为物理除数（divisor_effective 的
     _flow_counts 优先腿）；混合形态（仅流数链 + 速率链并存）下
     ``_effective`` 键集 = rates ∪ flow_counts 并集——_union_floor/
     telemetry_links/max_effective_divisor 计入仅流数链路；披露双口径
     可辨认（N3：flow_count_links / legacy_rate_divisor_links）；
  6. O10③（A19'(j)③）：LinkFlowRegistry ``leaked_owners()`` 公共
     泄漏视图——注册→全释放 = 空、注册→漏释放 = 返回该 owner 的在
     途流 id（release_flow 单独释放的账本残留不算泄漏）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_telemetry_divisor.py   （或 pytest 同路径）
"""
import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostError,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
    TelemetryLinkFlowView,
)


# ============================================================ 夹具构造 ==

def _rates(noc=10.0, hbm=100.0, lat=10, pool=5.0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=pool, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=100)


def _load(index=0, total=0):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25),
             prefill_scan_passes=None):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes)


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（rank 对 (2s,2t)/(2s+1,2t+1)，各 1 跳）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _route(source, target):
    return ((source, target), 1)


def _model(link_telemetry_rates=None, *, rates=None):
    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads={0: _load(0), 1: _load(1)},
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=_disjoint_paths,
        link_telemetry_rates=link_telemetry_rates)


# ================================================= 1. divisor_effective ==


class DivisorEffectiveTest(unittest.TestCase):
    """divisor_effective = max(divisor_registered, B_link/实测速率)。"""

    def test_undercount_lifted_by_telemetry(self):
        # 漏计分支：注册表只登记 1 流，实测速率 = B/3（物理上 3 条流
        # 均分）→ 实测除数 3 把注册除数 1 抬起来。
        registry = LinkFlowRegistry()
        registry.register(3, 5)
        view = registry.with_effective_rates(
            {(3, 5): 10.0 / 3.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.divisor_effective((3, 5)), 3.0)

    def test_overcount_not_amplified(self):
        # 多计分支：注册表 4 流（保守多计），实测速率 B/1.2（物理上
        # ~1.2 流均分）→ max 取 4，不放大到 4×1.2。
        registry = LinkFlowRegistry()
        for _ in range(4):
            registry.register(3, 5)
        view = registry.with_effective_rates(
            {(3, 5): 10.0 / 1.2}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.divisor_effective((3, 5)), 4.0)

    def test_max_not_sum_dedup(self):
        # §4.1 去重：注册 2 流 + 实测除数 3 → 3（非 5）——同一流量
        # 不重复计数。
        registry = LinkFlowRegistry()
        registry.register(3, 5)
        registry.register(3, 5)
        view = registry.with_effective_rates(
            {(3, 5): 10.0 / 3.0}, link_capacity_bytes_per_ns=10.0)
        merged = view.divisor_effective((3, 5))
        self.assertEqual(merged, 3.0)
        self.assertNotEqual(merged, 5.0)

    def test_no_sample_degrades_to_registered(self):
        # 字典中不出现的链路 = 无测量：注册表值即有效值。
        registry = LinkFlowRegistry()
        registry.register(3, 5)
        view = registry.with_effective_rates(
            {(7, 9): 5.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.divisor_effective((3, 5)), 1.0)
        self.assertIsNone(view.measured_effective_rate((3, 5)))
        self.assertIsNone(view.effective_divisor((3, 5)))
        self.assertEqual(view.measured_effective_rate((7, 9)), 5.0)
        self.assertAlmostEqual(view.effective_divisor((7, 9)), 2.0)

    def test_include_self_registered_leg(self):
        # N2 叠加口径：旧流并发两口径（注册/遥测换算）取大 + 候选
        # self_overlap 恒叠加。基础 1 流 + self_overlap 2：换算 2.5 →
        # max(1, 2.5)+2 = 4.5；换算 4.5 → max(1, 4.5)+2 = 6.5。
        registry = LinkFlowRegistry()
        registry.register(3, 5)
        view_a = registry.with_effective_rates(
            {(3, 5): 4.0}, link_capacity_bytes_per_ns=10.0)
        view_b = registry.with_effective_rates(
            {(3, 5): 10.0 / 4.5}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(
            view_a.divisor_effective((3, 5), include_self=True,
                                     self_overlap=2), 4.5)
        self.assertEqual(
            view_b.divisor_effective((3, 5), include_self=True,
                                     self_overlap=2), 6.5)

    def test_rate_above_capacity_floored_at_one(self):
        # 非物理输入（速率超容量 → 名义除数 < 1）不放大带宽：合并下界
        # 被 max(1, ·) 截住。
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {(3, 5): 20.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.divisor_effective((3, 5)), 1.0)


class TelemetryKeySpaceTest(unittest.TestCase):
    """冻结接口键两形：端点元组/"src->dst" 合并；整型 LinkId 端点无关
    消费；畸形键 fail-closed。"""

    def test_string_key_equivalent_to_tuple(self):
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {"3->5": 10.0 / 3.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.divisor_effective((3, 5)), 3.0)
        self.assertEqual(view.divisor_effective("3->5"), 3.0)

    def test_bare_int_link_id_endpoint_free_consumption(self):
        # 裸整型 LinkId（C8 _ingest_link_telemetry 原样键）：合法接收，
        # 速率/有效除数可查、divisor_effective 给纯遥测腿（无注册腿可
        # 言——键空间不相交）；不参与端点合并（(3,5) 不被抬升）；
        # disclosure 的 opaque_link_id_entries 计数披露（可见缺口，非
        # 静默）。
        registry = LinkFlowRegistry()
        registry.register(3, 5)
        view = registry.with_effective_rates(
            {17: 5.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.measured_effective_rate(17), 5.0)
        self.assertAlmostEqual(view.effective_divisor(17), 2.0)
        self.assertEqual(view.divisor_effective(17), 2.0)
        self.assertEqual(view.divisor_effective((3, 5)), 1.0)
        report = view.disclosure()
        self.assertEqual(report["opaque_link_id_entries"], 1)
        self.assertEqual(report["telemetry_links"], 0)
        # 未采样 LinkId 的 divisor_effective = fail-closed（非静默 1）。
        with self.assertRaises(JointCostError):
            view.divisor_effective(99)

    def test_malformed_keys_rejected(self):
        # 非链路键形（浮点/布尔/非 "src->dst" 串/None）fail-closed。
        registry = LinkFlowRegistry()
        for bad_key in (1.5, True, "link17", "a->b", None, (3, 5, 7)):
            with self.assertRaises(JointCostError):
                registry.with_effective_rates(
                    {bad_key: 5.0}, link_capacity_bytes_per_ns=10.0)

    def test_invalid_rate_rejected(self):
        registry = LinkFlowRegistry()
        for bad in (0.0, -1.0, float("nan"), float("inf"), True, "5"):
            with self.assertRaises(JointCostError):
                registry.with_effective_rates(
                    {(3, 5): bad}, link_capacity_bytes_per_ns=10.0)

    def test_invalid_capacity_rejected(self):
        registry = LinkFlowRegistry()
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(JointCostError):
                registry.with_effective_rates(
                    {(3, 5): 1.0}, link_capacity_bytes_per_ns=bad)


class TelemetryUnionDivisorTest(unittest.TestCase):
    """divisor_multi 的 max 层间合并（≡ 逐链路合并后取瓶颈）。"""

    def test_union_floor_lifts_bottleneck(self):
        # 并集 {(0,1),(1,2)}（N2 叠加口径逐链路）：(0,1) 注册 2 流无遥
        # 测 → 2+自身 1 = 3；(1,2) 注册 1 + 遥测换算 5 → max+自身 1
        # = 6 → 并集瓶颈 6。include_self=False：(1,2) 侧 max(1,5)=5。
        registry = LinkFlowRegistry()
        registry.register(0, 1)
        registry.register(0, 1)
        registry.register(1, 2)
        view = registry.with_effective_rates(
            {(1, 2): 2.0}, link_capacity_bytes_per_ns=10.0)
        paths = ((0, 1, 2),)
        self.assertEqual(registry.divisor_multi(paths, include_self=True), 3)
        self.assertEqual(view.divisor_multi(paths, include_self=True), 6.0)
        # include_self=False：候选不叠加，(1,2) 侧退 max(1,5)=5。
        self.assertEqual(
            view.divisor_multi(paths, include_self=False), 5.0)

    def test_registered_bottleneck_dominates(self):
        # 注册瓶颈 6 > 遥测 floor 2 → 不放大回 6（多计不放大在并集
        # 层同样成立）。
        registry = LinkFlowRegistry()
        for _ in range(5):
            registry.register(0, 1)
        view = registry.with_effective_rates(
            {(0, 1): 5.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(
            view.divisor_multi(((0, 1),), include_self=True), 6.0)

    def test_view_divisor_single_path_merges(self):
        # N2：(1,2) 注册 1 + 换算 7 → max + 候选 1 = 8（旧口径
        # max(注册含候选 2, 换算 7) = 7 漏候选叠加）。
        registry = LinkFlowRegistry()
        registry.register(1, 2)
        view = registry.with_effective_rates(
            {(1, 2): 10.0 / 7.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(
            view.divisor((0, 1, 2), include_self=True), 8.0)

    def test_registry_state_untouched_by_view(self):
        # 只读快照纪律：视图构造与查询不落注册表状态。
        registry = LinkFlowRegistry()
        registry.register_path((3, 4, 5), owner="o1")
        before_flows = registry.snapshot()
        before_divisor = registry.divisor_multi(((3, 4, 5),))
        view = registry.with_effective_rates(
            {(4, 5): 1.0}, link_capacity_bytes_per_ns=10.0)
        self.assertEqual(view.divisor_effective((4, 5)), 10.0)
        self.assertEqual(registry.snapshot(), before_flows)
        self.assertEqual(
            registry.divisor_multi(((3, 4, 5),)), before_divisor)
        self.assertFalse(registry.collective_coverage)

    def test_disclosure_and_coverage_passthrough(self):
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {(3, 5): 10.0 / 3.0, (7, 9): 5.0},
            link_capacity_bytes_per_ns=10.0)
        report = view.disclosure()
        self.assertEqual(report["telemetry_links"], 2)
        self.assertAlmostEqual(report["max_effective_divisor"], 3.0)
        self.assertEqual(report["link_capacity_bytes_per_ns"], 10.0)
        self.assertFalse(report["collective_coverage"])
        registry.collective_coverage = True   # 翻转接线在 SH 侧 C8
        self.assertTrue(view.collective_coverage)
        self.assertTrue(view.disclosure()["collective_coverage"])


# ==================================== O10③ LinkFlowRegistry 泄漏视图 ==


class LinkFlowRegistryLeakedOwnersTest(unittest.TestCase):
    """O10③（A19'(j)③，2026-09-23）：``leaked_owners()`` 公共视图——
    与 HbmPortFlowRegistry.leaked_owners 同款语义与返回形态（空=干净），
    供 SH run 尾断言消费（接口名冻结，接线在 SH 侧）。
    """

    def test_register_then_release_all_is_clean(self):
        registry = LinkFlowRegistry()
        registry.register_path((1, 2, 3), owner="r1")
        registry.register_path((4, 5), owner="r1")
        self.assertEqual(
            sorted(registry.leaked_owners()), ["r1"])
        released = registry.release_owner("r1")
        self.assertEqual(released, 2)
        self.assertEqual(registry.leaked_owners(), {})
        self.assertEqual(registry.snapshot(), {})

    def test_unreleased_owner_reported_with_live_flow_ids(self):
        registry = LinkFlowRegistry()
        flow_a = registry.register_path((1, 2, 3), owner="r1")
        registry.register_path((4, 5), owner="r2")
        leaked = registry.leaked_owners()
        self.assertEqual(list(leaked), ["r1", "r2"])
        self.assertEqual(leaked["r1"], (flow_a,))
        # 部分释放：r2 全释放后只剩 r1。
        registry.release_owner("r2")
        self.assertEqual(registry.leaked_owners(), {"r1": (flow_a,)})

    def test_release_flow_residual_not_counted_as_leak(self):
        # release_flow 单独释放的流 id 在 _owner_flows 的残留不算
        # 泄漏（物理链路已注销；leaked_owners 只统计仍实际在途的流）。
        registry = LinkFlowRegistry()
        flow_a = registry.register_path((1, 2, 3), owner="r1")
        registry.register_path((4, 5), owner="r1")
        registry.release_flow(flow_a)
        self.assertEqual(registry.leaked_owners(), {"r1": (1,)})


# ==================================== O7③ 混合形态仅流数链路并集 ==


class MixedFormFlowCountUnionTest(unittest.TestCase):
    """O7③（A19'(g)③，2026-09-23）：混合形态（一条仅流数链+一条速率
    链）下 ``_effective`` 键集 = rates ∪ flow_counts 并集——
    _union_floor/telemetry_links/max_effective_divisor 计入仅流数链路
    （修前漏计：探针 (1,2) 仅流数 7.0 时 divisor_effective=8.0 正确、
    _union_floor 只给 1.0、披露键集不含）。divisor_effective 经
    _flow_counts 优先本已正确（回归锚），纯仅流数形态（__post_init__
    假速率分支）不回退。
    """

    def test_union_floor_counts_flow_count_only_link(self):
        # 探针形态复现：链路 (1,2) 仅流数 7.0（不在 rates 键集）、
        # 链路 (3,4) 仅速率（B=10、rate=2.5 ⇒ 除数 4.0）。
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {(3, 4): 2.5}, link_capacity_bytes_per_ns=10.0,
            link_flow_counts={(1, 2): 7.0})
        # divisor_effective 回归锚：流数腿 7+自身 1=8；速率腿
        # max(注册 0, 4)=4。
        self.assertEqual(
            view.divisor_effective((1, 2), include_self=True,
                                   self_overlap=1), 8.0)
        self.assertEqual(view.divisor_effective((3, 4)), 4.0)
        # 修前：_union_floor 只扫 rates 键集 ⇒ 1.0（漏仅流数链路）。
        self.assertEqual(view._union_floor(((1, 2, 9),)), 7.0)
        # 披露同步：键集并集计数与下界。
        report = view.disclosure()
        self.assertEqual(report["telemetry_links"], 2)
        self.assertEqual(report["max_effective_divisor"], 7.0)
        self.assertEqual(report["flow_count_links"], 1)
        self.assertEqual(report["legacy_rate_divisor_links"], 1)

    def test_pure_flow_count_form_unchanged(self):
        # 纯仅流数形态（JointCostModel.__post_init__ 假速率分支同式
        # 直构）不回退：_union_floor/披露与混合形态同口径。
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {(1, 2): 1.0}, link_capacity_bytes_per_ns=10.0,
            link_flow_counts={(1, 2): 7.0})
        self.assertEqual(view.divisor_effective((1, 2)), 7.0)
        self.assertEqual(view._union_floor(((1, 2, 9),)), 7.0)
        report = view.disclosure()
        self.assertEqual(report["telemetry_links"], 1)
        self.assertEqual(report["max_effective_divisor"], 7.0)
        self.assertEqual(report["flow_count_links"], 1)
        self.assertEqual(report["legacy_rate_divisor_links"], 0)


# ============================================ 2. JointCostModel 接线 ==


class ModelTelemetryWiringTest(unittest.TestCase):
    """冻结交接接口：{link_id: 实测速率} 入构造；None/空 = 零漂移。"""

    @staticmethod
    def _remote_candidate(model):
        return model.estimate_action(
            session=_session(history_bytes=(1000, 1000)),
            request=_request(decode=3), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)

    def test_none_and_empty_dict_zero_drift(self):
        # 无遥测 = 原 Registry 模型：None、空字典、缺省三态同值。
        base = self._remote_candidate(_model()).breakdown
        none_case = self._remote_candidate(
            _model(link_telemetry_rates=None)).breakdown
        empty_case = self._remote_candidate(
            _model(link_telemetry_rates={})).breakdown
        self.assertEqual(base, none_case)
        self.assertEqual(base, empty_case)
        self.assertEqual(base.contention_divisor, 1)

    def test_telemetry_lifts_remote_pricing(self):
        # resident=0 → instance=1 的逐 rank 路径 ((0,2),(1,3))；对
        # (0,2) 喂实测速率 B/2（物理 2 流均分）→ N2 叠加口径 noc 腿
        # 除数 = 旧流 2 + 候选 1 = 3、remote_read_ns 三倍；breakdown
        # 披露位保守取整 = 3。
        rates = _rates(noc=10.0, lat=0)
        base = self._remote_candidate(_model(rates=rates)).breakdown
        lifted = self._remote_candidate(_model(
            link_telemetry_rates={(0, 2): 5.0}, rates=rates)).breakdown
        self.assertEqual(base.contention_divisor, 1)
        self.assertEqual(lifted.contention_divisor, 3)
        self.assertEqual(
            lifted.remote_read_ns - base.remote_read_ns,
            2 * base.remote_read_ns)

    def test_telemetry_disclosure_channels(self):
        self.assertEqual(
            _model().link_telemetry_disclosure(), {"telemetry": False})
        model = _model(link_telemetry_rates={(0, 2): 5.0})
        report = model.link_telemetry_disclosure()
        self.assertTrue(report["telemetry"])
        self.assertEqual(report["telemetry_links"], 1)
        self.assertEqual(report["opaque_link_id_entries"], 0)
        self.assertAlmostEqual(report["max_effective_divisor"], 2.0)

    def test_c8_feed_shape_int_keys_accepted_no_pricing_drift(self):
        # G2 联集成单点：C8 _ingest_link_telemetry 的原样喂入形态
        # （{int LinkId: B/ns}，dataclass 软门自动接通）——构造成功、
        # 计价零漂移（整型键不参与端点合并）、opaque 条目披露可见。
        rates = _rates(noc=10.0, lat=0)
        base = self._remote_candidate(_model(rates=rates)).breakdown
        fed = self._remote_candidate(_model(
            link_telemetry_rates={17: 5.0, 23: 10.0 / 3.0},
            rates=rates)).breakdown
        self.assertEqual(base, fed)
        report = _model(
            link_telemetry_rates={17: 5.0, 23: 10.0 / 3.0},
            rates=rates).link_telemetry_disclosure()
        self.assertEqual(report["opaque_link_id_entries"], 2)
        self.assertEqual(report["telemetry_links"], 0)

    def test_fractional_divisor_ceil_in_breakdown(self):
        # 换算 2.5（非整数）+ 候选 1（N2 叠加）→ 计价腿用原值 3.5、
        # 披露位 ceil=4（保守侧）。
        rates = _rates(noc=10.0, lat=0)
        model = _model(link_telemetry_rates={(0, 2): 4.0}, rates=rates)
        lifted = self._remote_candidate(model).breakdown
        self.assertEqual(lifted.contention_divisor, 4)
        # 计价腿仍是 3.5 除数：bytes/有效速率 与 3.5 除数解析一致
        # （1000 B/rank × 4 遍 = 4000 B；4000/(10/3.5)=1400 ns；TP
        # 并行取 max → 两 rank 同值，wall = 1400）。
        self.assertEqual(lifted.remote_read_ns, 1400)


# ======================================== 3. collective_coverage 条件 ==


class CollectiveCoverageTest(unittest.TestCase):
    """翻 True 条件 = 全程遥测开启 ∧ 逐 epoch 窗口序列无空洞。"""

    def test_telemetry_disabled_whole_run(self):
        # 未开遥测（或中途开启）→ 条件不满足。
        flag, report = LinkFlowRegistry.evaluate_collective_coverage(
            telemetry_enabled_whole_run=False,
            window_bounds=[(0, 10), (10, 20)])
        self.assertFalse(flag)
        self.assertFalse(report["telemetry_enabled_whole_run"])

    def test_contiguous_windows_cover(self):
        flag, report = LinkFlowRegistry.evaluate_collective_coverage(
            telemetry_enabled_whole_run=True,
            window_bounds=[(0, 10), (10, 25), (25, 40)],
            run_end_ns=40)
        self.assertTrue(flag)
        self.assertEqual(report["window_count"], 3)
        self.assertEqual(report["hole_count"], 0)
        self.assertEqual(report["covered_span_ns"], 40)

    def test_gap_hole_detected(self):
        # 空洞：epoch 缺席/掉线 → 相邻窗 start != prev end。
        flag, report = LinkFlowRegistry.evaluate_collective_coverage(
            telemetry_enabled_whole_run=True,
            window_bounds=[(0, 10), (15, 25)])
        self.assertFalse(flag)
        self.assertEqual(report["hole_count"], 1)
        self.assertEqual(report["holes"][0]["kind"], "gap")
        self.assertEqual(report["holes"][0]["expected_start_ns"], 10)

    def test_start_and_end_anchors(self):
        flag_start, report_start = (
            LinkFlowRegistry.evaluate_collective_coverage(
                telemetry_enabled_whole_run=True,
                window_bounds=[(5, 10), (10, 20)]))
        self.assertFalse(flag_start)
        self.assertEqual(report_start["holes"][0]["kind"], "start")
        flag_end, report_end = (
            LinkFlowRegistry.evaluate_collective_coverage(
                telemetry_enabled_whole_run=True,
                window_bounds=[(0, 10), (10, 20)],
                run_end_ns=50))
        self.assertFalse(flag_end)
        self.assertEqual(report_end["holes"][-1]["kind"], "end")

    def test_empty_window_sequence_not_covered(self):
        # 空窗口序列（零 delivery 证据）→ 不满足。
        flag, report = LinkFlowRegistry.evaluate_collective_coverage(
            telemetry_enabled_whole_run=True, window_bounds=[])
        self.assertFalse(flag)
        self.assertEqual(report["window_count"], 0)
        self.assertEqual(report["covered_span_ns"], 0)

    def test_all_idle_epoch_zero_length_window_legal(self):
        # 全闲 epoch（数组为空但 epoch 存在，界值取 delivery tick）：
        # 零长窗 [t, t) 合法，不构成空洞。
        flag, report = LinkFlowRegistry.evaluate_collective_coverage(
            telemetry_enabled_whole_run=True,
            window_bounds=[(0, 10), (10, 10), (10, 20)],
            run_end_ns=20)
        self.assertTrue(flag)
        self.assertEqual(report["hole_count"], 0)

    def test_negative_window_flagged(self):
        flag, report = LinkFlowRegistry.evaluate_collective_coverage(
            telemetry_enabled_whole_run=True,
            window_bounds=[(0, 10), (12, 5)])
        self.assertFalse(flag)
        kinds = {hole["kind"] for hole in report["holes"]}
        self.assertIn("negative_window", kinds)
        self.assertIn("gap", kinds)


# ============================================ 4. transfer_factor EWMA ==


class TransferFactorEwmaTest(unittest.TestCase):
    """观测入口 observe_transfer_from_link_window 的样本纪律。"""

    def test_first_sample_initializes(self):
        # served 1000 B、active 2000 ns、名义 1 B/ns → base 1000、
        # actual 2000 → 样本比 = 名义/实测 = 2.0（拥胀方向 ≥ 1）。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1000, active_ns=2000,
            nominal_rate_bytes_per_ns=1.0, tick_ns=100)
        factors.flush()
        self.assertAlmostEqual(factors.transfer_factor, 2.0)
        self.assertEqual(factors.updates["transfer_factor"], 1)
        self.assertEqual(factors.rejected.get("transfer_factor", 0), 0)

    def test_ewma_alpha_decay_replicated(self):
        # 两样本 α 公式复算：α = 1 − exp(−Δt/τ)，τ = 上一有效样本的
        # 正服务时长（§5.4/P3 原文），非首样本 EWMA 推进。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1000, active_ns=2000,
            nominal_rate_bytes_per_ns=1.0, tick_ns=100)
        factors.observe_transfer_from_link_window(
            served_bytes=1000, active_ns=4000,
            nominal_rate_bytes_per_ns=1.0, tick_ns=10_000)
        factors.flush()
        alpha = 1.0 - math.exp(-(10_000 - 100) / 2000)
        expected = (1 - alpha) * 2.0 + alpha * 4.0
        self.assertAlmostEqual(factors.transfer_factor, expected,
                               places=9)
        self.assertEqual(factors.updates["transfer_factor"], 2)

    def test_same_tick_links_aggregate_single_update(self):
        # 同 tick 两链路样本：Σactual/Σbase 单条更新（_record 既有
        # 时刻汇总纪律）——(2000+3000)/(1000+3000) = 1.25。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1000, active_ns=2000,
            nominal_rate_bytes_per_ns=1.0, tick_ns=50)
        factors.observe_transfer_from_link_window(
            served_bytes=3000, active_ns=3000,
            nominal_rate_bytes_per_ns=1.0, tick_ns=50)
        factors.flush()
        self.assertAlmostEqual(factors.transfer_factor, 1.25)
        self.assertEqual(factors.updates["transfer_factor"], 1)

    def test_zero_sample_rejected_not_updated(self):
        # 零字节/零活跃窗口（C6 对 idle 链路不发射，但空窗防御性拒绝
        # 计数披露）：不更新、factor 保持冷启动 1.0。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=0, active_ns=5,
            nominal_rate_bytes_per_ns=1.0, tick_ns=10)
        factors.observe_transfer_from_link_window(
            served_bytes=100, active_ns=0,
            nominal_rate_bytes_per_ns=1.0, tick_ns=10)
        factors.flush()
        self.assertEqual(factors.transfer_factor, 1.0)
        self.assertEqual(factors.updates.get("transfer_factor", 0), 0)
        self.assertEqual(factors.rejected["transfer_factor"], 2)

    def test_congestion_direction_above_one(self):
        # 方向锚：实测速率低于名义（拥胀）→ 因子 > 1（与因子族
        # actual/base、"1.0 冷启动乐观"口径一致）。
        factors = ServiceFactors()
        factors.observe_transfer_from_link_window(
            served_bytes=1000, active_ns=4000,
            nominal_rate_bytes_per_ns=1.0, tick_ns=10)
        factors.flush()
        self.assertGreater(factors.transfer_factor, 1.0)

    def test_invalid_nominal_rate_fail_closed(self):
        factors = ServiceFactors()
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(JointCostError):
                factors.observe_transfer_from_link_window(
                    served_bytes=1000, active_ns=2000,
                    nominal_rate_bytes_per_ns=bad, tick_ns=10)


if __name__ == "__main__":
    unittest.main()
