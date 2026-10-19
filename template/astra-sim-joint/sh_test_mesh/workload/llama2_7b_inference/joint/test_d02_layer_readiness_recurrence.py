#!/usr/bin/env python3
"""test_d02_layer_readiness_recurrence.py -- D02 逐层数据/空间就绪递推
（v2.0 式(12)/(13)/§11，2026-09-26）的零后端单测。

语义合同（joint_cost_model estimate_P_with_continuation 三段重构）：

* "历史准备—计算—关键路径合成"改为逐层就绪递推：一次性传输
  （copy handoff chunk / 池恢复）按 式(12) 供给链
  只写它**真实消费的层**的 ``ready_data`` 门（copy 通道内按层消费序
  串行；池后缀与 NoC 前缀不同通道并行）；聚合 prefill 计算按固定层
  数均分（余数前置、总和守恒）为固定数量算子段；式(13) 递推输出
  prefill 首前向完成时刻；
* 验收红线（v2.0 §11.3 的 110 ns 例）：第二层数据晚到 100、其计算
  10 ⇒ 结果 = max(F_1, 100) + 10 = 110——**不存在整体 max 跳过到达
  后计算**（旧 max(尾流, 计算) / max(prefill 腿, suffix_restore) /
  first_credit + max(stream, compute) 三种整体合成全部废除、无开关）；
* prefill 结束前所有需要层就绪：递推覆盖全部 L 层，末层完成 ≥ 末层
  数据到达 + 其算子份额；
* 纯算术接口：``estimate_first_forward``（式 13；远读 tile 到达闭式
  与重复段折叠已随远读退役删除）；
* 守恒锚：零门控动作（stay-LOCAL）递推 = 整段计算，与旧
  口径逐位一致。

手算锚硬件：B_link = 10 B/ns、
B_pool = 5 B/ns、B_HBM = 100 B/ns、单跳时延 10 ns、池端口事务时延
100 ns、流注册表空、u_port = 0（离线口径）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d02_layer_readiness_recurrence.py（或 pytest）
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
    JointCostError,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
    InstanceLoadView,
    _copy_handoff_layer_ranges,
    _split_ns_by_layer,
    estimate_first_forward,
)


# ============================================================ 夹具构造 ==

def _rates(pool_latency_ns=100, d2d_latency_ns=10):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=d2d_latency_ns, pool_latency_ns=pool_latency_ns)


def _load(index, total=0, remaining=(10**9, 10**9)):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1000, 1000),
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


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（除数 1——noc 有效率 = 10 B/ns）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _model(loads, *, layers=4, route_paths_fn=None, d2d_latency_ns=10):
    return JointCostModel(
        rates=_rates(d2d_latency_ns=d2d_latency_ns),
        loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=layers, instance_tp_size=2, route_fn=_route,
        route_paths_fn=route_paths_fn)


def _note_value(breakdown, key):
    prefix = key + "="
    for note in breakdown.notes:
        if note.startswith(prefix):
            return int(note[len(prefix):])
    raise AssertionError(f"missing disclosure note {key!r}")


# ================================================= 式(13) 纯递推单测 ==

class FirstForwardRecurrenceTest(unittest.TestCase):
    """estimate_first_forward：式(13) 逐层递推的直译锚。"""

    def test_110ns_late_layer_example_v2_section_11_3(self):
        # v2.0 §11.3 原例：第一层数据已到、计算 1 ns；第二层数据到达
        # 100 ns、计算 10 ns ⇒ max(1, 100) + 10 = 110 ns。
        result = estimate_first_forward(
            layer_count=2,
            pre_ns_by_layer=(0, 0),
            attn_ns_by_layer=(1, 10),
            post_ns_by_layer=(0, 0),
            ready_data_by_layer=[(0,), (100,)],
            ready_space_ns=0,
            start_ns=0)
        self.assertEqual(result, 110)

    def test_whole_max_form_would_skip_arrival_compute(self):
        # 同例对照整体 max 形态：max(全部数据 100, 全部计算 11) = 100
        # ——漏掉数据到达后必需的计算；递推必须严格大于它。
        result = estimate_first_forward(
            layer_count=2,
            pre_ns_by_layer=(0, 0),
            attn_ns_by_layer=(1, 10),
            post_ns_by_layer=(0, 0),
            ready_data_by_layer=[(0,), (100,)])
        self.assertEqual(result, 110)
        self.assertGreater(result, max(100, 11))

    def test_three_segment_gates_v2_equation_13(self):
        # 式(13) 三段门逐位：P_g = max(F_{g-1}, A^work) + Cpre（空间门
        # 修前段、数据门不修前段）；F_g = max(P_g, R_g) + Cattn + Cpost。
        # 夹具：start 10、space 40、L1 = max(10,40)+5 = 45 → 45+7+3 =
        # 55；L2 = max(55,40)+5 = 60 → max(60, 90)+10 = 100。
        result = estimate_first_forward(
            layer_count=2,
            pre_ns_by_layer=(5, 5),
            attn_ns_by_layer=(7, 7),
            post_ns_by_layer=(3, 3),
            ready_data_by_layer=[(0,), (90,)],
            ready_space_ns=40,
            start_ns=10)
        self.assertEqual(result, 100)
        # 空间门抬高前段：space 200 → P1 = 205、F1 = 215、
        # P2 = max(215,200)+5 = 220、F2 = max(220,90)+10 = 230。
        lifted = estimate_first_forward(
            layer_count=2,
            pre_ns_by_layer=(5, 5),
            attn_ns_by_layer=(7, 7),
            post_ns_by_layer=(3, 3),
            ready_data_by_layer=[(0,), (90,)],
            ready_space_ns=200,
            start_ns=10)
        self.assertEqual(lifted, 230)

    def test_zero_gating_conserves_total(self):
        # 零门控（ready 全零、space 0）：F_L = start + Σ份额（守恒锚
        # ——与旧整段口径逐位一致；均分余数前置）。
        shares = _split_ns_by_layer(70, 16)
        self.assertEqual(sum(shares), 70)
        result = estimate_first_forward(
            layer_count=16,
            pre_ns_by_layer=(0,) * 16,
            attn_ns_by_layer=shares,
            post_ns_by_layer=(0,) * 16,
            ready_data_by_layer=[() for _ in range(16)])
        self.assertEqual(result, 70)
        self.assertEqual(
            estimate_first_forward(
                layer_count=16, pre_ns_by_layer=(0,) * 16,
                attn_ns_by_layer=shares, post_ns_by_layer=(0,) * 16,
                ready_data_by_layer=[() for _ in range(16)],
                start_ns=1234),
            1234 + 70)

    def test_per_rank_data_gate_takes_rank_max(self):
        # 式(13) F_g = max_r[...]：逐 rank 到取 max（TP 同步）；不均衡
        # 到达由慢 rank 决定层数据门。
        result = estimate_first_forward(
            layer_count=2,
            pre_ns_by_layer=(0, 0),
            attn_ns_by_layer=(5, 5),
            post_ns_by_layer=(0, 0),
            ready_data_by_layer=[(10, 30), (40, 20)])
        # L1 = max(10,30)+5 = 35；L2 = max(35, max(40,20))+5 = 45。
        self.assertEqual(result, 45)

    def test_validation_fail_closed(self):
        with self.assertRaises(JointCostError):
            estimate_first_forward(
                layer_count=2, pre_ns_by_layer=(0,),
                attn_ns_by_layer=(1, 1), post_ns_by_layer=(0, 0),
                ready_data_by_layer=[(0,), (0,)])
        with self.assertRaises(JointCostError):
            estimate_first_forward(
                layer_count=1, pre_ns_by_layer=(-1,),
                attn_ns_by_layer=(1,), post_ns_by_layer=(0,),
                ready_data_by_layer=[(0,)])
        with self.assertRaises(JointCostError):
            estimate_first_forward(
                layer_count=1, pre_ns_by_layer=(0,),
                attn_ns_by_layer=(1,), post_ns_by_layer=(0,),
                ready_data_by_layer=[(-5,)])
        with self.assertRaises(JointCostError):
            estimate_first_forward(
                layer_count=0, pre_ns_by_layer=(), attn_ns_by_layer=(),
                post_ns_by_layer=(), ready_data_by_layer=[])


# ============================================== 重复段/到达闭式单测 ==

class CopyLayerGateIntegrationTest(unittest.TestCase):
    """copy handoff chunk 只门控其层区间；末 chunk 到达后的计算由递推
    保留（110 ns 性质的生产尺度形态）。"""

    def _copy_candidate(self, history_bytes):
        # lat=0（d2d 时延零——test_joint_shard_pricing copy 夹具同族，
        # 每 chunk 事务 X = 纯流送腿）；disjoint 路径 → 并集除数 1。
        model = _model(
            {0: _load(0), 1: _load(1)}, layers=16,
            route_paths_fn=_disjoint_paths, d2d_latency_ns=0)
        candidate = model.estimate_P_with_continuation(
            session=_session(
                history_bytes=history_bytes, prefix=16, resident=0),
            request=_request(input_tokens=50, decode=10),
            instance_index=1, action=ACTION_COPY)
        self.assertTrue(candidate.applicable)
        return model, candidate

    def test_copy_late_chunk_compute_preserved(self):
        # 手算（lat=0、disjoint 除数 1）：chunk 字节 800/rank → 每 chunk
        # X = 80；通道串行链 80 → 160；prefill 计算 50 按层均分
        # (4,4,3×14)：F7 = 88+3×6 = 106 → F8 = max(106,160)+3 = 163 →
        # F15 = 163+21 = 184；decode 20 ⇒ cost = 204。旧整体 max 形态
        # = 80 + max(80, 70) = 160——末 chunk 到达（160）后的 8 层计算
        # 被吸收；递推逐位保留（§11.3 验收的生产尺度形态）。
        model, candidate = self._copy_candidate((1600, 1600))
        self.assertEqual(candidate.breakdown.history_prep_ns, 160)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            184)
        self.assertEqual(candidate.cost_ns, 184 + 20)
        self.assertGreater(
            candidate.cost_ns,
            80 + max(80, 70))  # 整体 max 形态（已废除）的值
        self.assertEqual(
            _copy_handoff_layer_ranges(16), ((0, 8), (8, 16)))

    def test_copy_last_chunk_arrival_bounds_prefill(self):
        # prefill 结束前所有需要层就绪：首前向 ≥ 末 chunk 到达，且
        # ≤ 末 chunk 到达 + 末 chunk 层区间的份额总和（无虚构等待）。
        _, candidate = self._copy_candidate((1600, 1600))
        first_forward = _note_value(
            candidate.breakdown, "prefill_first_forward_ns")
        last_arrival = 160
        tail_shares = sum(_split_ns_by_layer(50, 16)[8:])
        self.assertGreaterEqual(first_forward, last_arrival)
        self.assertLessEqual(first_forward, last_arrival + tail_shares)
        self.assertEqual(first_forward, 184)
        self.assertEqual(tail_shares, 24)

    def test_copy_tp_skew_propagates_via_rank_max(self):
        # 逐 rank 供给链（式 12 R_{g,r}）+ 式(13) max_r：rank1 字节加倍
        # （3200）→ 每 chunk X = 160、链 160→320；层门 = max_r。F7 =
        # 88+3×6 = 106（首 chunk 门 = max(80,160) = 160）→ F8 =
        # max(106,320)+3 = 323 → F15 = 344；decode 20 ⇒ cost = 364。
        model, candidate = self._copy_candidate((1600, 3200))
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            344)
        self.assertEqual(candidate.cost_ns, 344 + 20)
        self.assertEqual(
            _note_value(candidate.breakdown, "copy_handoff_first_chunk_ns"),
            160)


class StayPoolSuffixLayerGateTest(unittest.TestCase):
    """stay-PARTIAL：池恢复是一次性传输——只门控缺失后缀的层
    [p, L)；驻留前缀层的计算与恢复并行（旧 prep = max(restore, ev)
    整段串行形态废除）。"""

    def test_pool_suffix_gates_only_consuming_layers(self):
        # 手算（D04 逐 shard 池腿）：restore = 100 + 1000/5 = 300；前缀
        # 两层无门 → 13、26；后缀两层 = max(26, 300)+12、+12 → 324；
        # decode 20 ⇒ cost = 344。旧整段串行形态 = 300 + 70 = 370——
        # 前缀计算 26 被递推与恢复重叠。
        model = _model({0: _load(0), 1: _load(1)})
        session = _session(
            home=1, resident=1, history_bytes=(1000, 1000),
            missing=(1000, 1000), location="partial_hbm_remote", prefix=2)
        candidate = model.estimate_P_with_continuation(
            session=session, request=_request(), instance_index=1,
            action=ACTION_STAY)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.history_prep_ns, 300)
        self.assertEqual(
            _note_value(candidate.breakdown, "prefill_first_forward_ns"),
            324)
        self.assertEqual(candidate.cost_ns, 324 + 20)
        self.assertLess(candidate.cost_ns, 300 + 70)

    def test_pool_rate_lifts_only_through_suffix_gate(self):
        # 池端口时延放大 → 后缀层门等量抬升（restore 300 →
        # 10^6+200）；递推把抬高传递到 cost（消费层真实等待）。
        base = _model({0: _load(0), 1: _load(1)})
        slow = JointCostModel(
            rates=_rates(pool_latency_ns=10**6), loads=base.loads,
            flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2, route_fn=_route)
        session = _session(
            home=1, resident=1, history_bytes=(1000, 1000),
            missing=(1000, 1000), location="partial_hbm_remote", prefix=2)
        base_c = base.estimate_P_with_continuation(
            session=session, request=_request(), instance_index=1,
            action=ACTION_STAY)
        slow_c = slow.estimate_P_with_continuation(
            session=session, request=_request(), instance_index=1,
            action=ACTION_STAY)
        self.assertEqual(base_c.breakdown.history_prep_ns, 300)
        self.assertEqual(
            slow_c.breakdown.history_prep_ns, 10**6 + 200)
        self.assertGreater(slow_c.cost_ns, base_c.cost_ns)
        # 抬升量逐位 = 池时延增量（后缀层门全程绑定恢复完成）。
        self.assertEqual(
            slow_c.cost_ns - base_c.cost_ns, 10**6 + 200 - 300)


class ConservationAnchorTest(unittest.TestCase):
    """零门控动作守恒锚：递推退化为整段计算（与旧口径逐位一致）。"""

    def test_stay_local_frontier_plus_compute(self):
        model = _model({0: _load(0), 1: _load(1, total=5000)})
        candidate = model.estimate_P_with_continuation(
            session=_session(home=1, resident=1),
            request=_request(), instance_index=1,
            action=ACTION_STAY)
        self.assertTrue(candidate.applicable)
        # target 5000 串行 + prefill 50 + decode 20（无 merge：home ==
        # 目标）；递推零门控 = 整段（无 D02 漂移）。
        self.assertEqual(candidate.breakdown.target_wait_ns, 5000)
        self.assertEqual(candidate.cost_ns, 5000 + 50 + 20)


if __name__ == "__main__":
    unittest.main()
